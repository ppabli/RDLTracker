import rclpy
import open3d as o3d
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2

def ros2_msg_to_o3d_xyz(ros_cloud):
	"""
	Convert the (x, y, z) ROS2 PointCloud2 message to an Open3D Tensor-based (x,y,z) PointCloud.
	"""

	device = o3d.core.Device("CPU:0")
	dtype = o3d.core.float32

	dtype_np = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
	cloud_array = np.array(list(pc2.read_points(ros_cloud, field_names=("x", "y", "z"), skip_nans=True)), dtype=dtype_np)

	points = np.stack((cloud_array['x'], cloud_array['y'], cloud_array['z']), axis=-1)

	o3d_cloud = o3d.t.geometry.PointCloud(device)
	o3d_cloud.point.positions = o3d.core.Tensor(points, dtype, device)

	return o3d_cloud

def o3d_to_ros_msg_xyz(o3d_cloud, frame_id):
	"""
	Convert an Open3D Tensor-based PointCloud (x, y, z) to a ROS2 PointCloud2 message efficiently.
	"""

	points = o3d_cloud.point.positions.numpy().astype(np.float32)

	structured_array = np.zeros(len(points), dtype=[
		('x', np.float32), ('y', np.float32), ('z', np.float32),
		('intensity', np.float32), ('rgb', np.float32), ('label', np.uint32)
	])

	structured_array['x'], structured_array['y'], structured_array['z'] = points.T

	header = Header()
	header.stamp = rclpy.time.Time().to_msg()
	header.frame_id = frame_id

	fields = [
		PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
		PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
		PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
		PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
		PointField(name="rgb", offset=16, datatype=PointField.FLOAT32, count=1),
		PointField(name="label", offset=20, datatype=PointField.UINT32, count=1),
	]

	pc2_msg = pc2.create_cloud(header, fields, structured_array)

	return pc2_msg

def filter_points_downsample(cloud, voxel_size=0.1):
	"""
	Downsample a point cloud using voxel grid downsampling.
	"""

	cloud_downsampled = cloud.voxel_down_sample(voxel_size)

	return cloud_downsampled

def filter_points_by_distance(cloud, distance_threshold):
	"""
	Filter points in a point cloud based on a distance threshold from the origin.
	"""

	positions_np = cloud.point.positions.numpy()

	distances = np.linalg.norm(positions_np, axis=1)

	mask = distances < distance_threshold
	cloud = cloud.select_by_index(np.where(mask)[0])

	return cloud

def crop_y(cloud, amplitude_threshold=0.5):
	"""
	Crop a point cloud based on the y-axis margin.
	"""

	positions_np = cloud.point.positions.numpy()

	mask = np.abs(positions_np[:, 1]) < amplitude_threshold
	cloud = cloud.select_by_index(np.where(mask)[0])

	return cloud

def filter_points_floor(cloud, distance_threshold=0.05, ransac_n=10, num_iterations=50):
	"""
	Filter the floor from a point cloud using RANSAC plane segmentation.
	"""

	plane_model, inliers = cloud.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)

	cloud_no_floor = cloud.select_by_index(inliers, invert=True)

	return cloud_no_floor

def filter_points_objects(cloud, eps=0.25, min_points=20, object_limit=10):
	"""
	Filter the objects from a point cloud using DBSCAN clustering.
	"""

	points = cloud.point.positions.numpy()
	clusters = cloud.cluster_dbscan(eps=eps, min_points=min_points).numpy()

	if clusters.max() < 0:
		return o3d.t.geometry.PointCloud(cloud.device), []

	unique_clusters, cluster_counts = np.unique(clusters[clusters >= 0], return_counts=True)

	valid_clusters = unique_clusters[cluster_counts >= min_points]

	if valid_clusters.size == 0:

		return o3d.t.geometry.PointCloud(cloud.device), []

	mask = np.isin(clusters, valid_clusters)
	valid_points = points[mask]
	valid_clusters = clusters[mask]

	centroids = np.array([
		valid_points[valid_clusters == cid].mean(axis=0) for cid in np.unique(valid_clusters)
	])

	distances = np.linalg.norm(centroids[:, :2], axis=1)
	sorted_indices = np.argsort(distances)[:object_limit]

	selected_clusters = valid_clusters[np.isin(valid_clusters, valid_clusters[sorted_indices])]

	filtered_cloud = o3d.t.geometry.PointCloud(cloud.device)
	filtered_cloud.point.positions = o3d.core.Tensor(valid_points[np.isin(valid_clusters, selected_clusters)], dtype=o3d.core.float32, device=cloud.device)

	return filtered_cloud, [
		{
			'distance': distances[idx],
			'centroid': centroids[idx],
			'points': valid_points[valid_clusters == valid_clusters[sorted_indices[idx]]]
		}
		for idx in range(len(sorted_indices))
	]

def get_bounding_box_dimensions(bounding_box, use_oriented=False):
	"""
	Get the dimensions of a bounding box.
	"""

	extents = bounding_box.extent if use_oriented else bounding_box.get_extent()

	return extents[0], extents[1], extents[2]

