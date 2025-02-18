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

def filter_points_downsample(cloud, voxel_size=0.075):
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

def filter_points_floor(cloud, distance_threshold=0.05, ransac_n=10, num_iterations=75):
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

	if (max_cluster := clusters.max()) < 0:

		empty_cloud = o3d.t.geometry.PointCloud(cloud.device)
		empty_cloud.point.positions = o3d.core.Tensor(np.zeros((0, 3)), dtype=o3d.core.float32, device=cloud.device)

		return empty_cloud, []

	unique_clusters = np.arange(max_cluster + 1)
	cluster_points = [points[clusters == cid] for cid in unique_clusters]
	centroids = np.array([points[clusters == cid].mean(axis=0) for cid in unique_clusters])
	distances = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)

	sorted_indices = np.argsort(distances)[:object_limit]
	kept_clusters = unique_clusters[sorted_indices]

	final_mask = np.isin(clusters, kept_clusters)
	filtered_points = points[final_mask]

	filtered_cloud = o3d.t.geometry.PointCloud(cloud.device)
	filtered_cloud.point.positions = o3d.core.Tensor(filtered_points, dtype=o3d.core.float32, device=cloud.device)

	return filtered_cloud, [
		{
			'distance': distances[idx],
			'centroid': centroids[idx],
			'points': cluster_points[cid]
		}
		for idx, cid in enumerate(kept_clusters)
	]

def get_bounding_box_dimensions(bounding_box, use_oriented=False):
	"""
	Get the dimensions of a bounding box.
	"""

	extents = bounding_box.extent if use_oriented else bounding_box.get_extent()

	return extents[0], extents[1], extents[2]

