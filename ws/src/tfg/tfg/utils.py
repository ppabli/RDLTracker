import numpy as np
import open3d as o3d
import rclpy
import requests
import sensor_msgs_py.point_cloud2 as pc2
import torch
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from tfg.model import normalize_point_cloud_tensor
from tfg.tracked_object import TrackedObject
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from tfg.constants import O3D_DEVICE, TORCH_DEVICE, O3D_DTYPE, TORCH_DTYPE, NP_DTYPE


def ros2_msg_to_o3d_xyz(ros_cloud):
	"""
	Convert the (x, y, z) ROS2 PointCloud2 message to an Open3D Tensor-based (x,y,z) PointCloud.
	"""

	dtype_np = [('x', NP_DTYPE), ('y', NP_DTYPE), ('z', NP_DTYPE)]
	cloud_array = np.array(pc2.read_points(ros_cloud, field_names=("x", "y", "z"), skip_nans=True), dtype=dtype_np)

	points = np.stack((cloud_array['x'], cloud_array['y'], cloud_array['z']), axis=-1)

	o3d_cloud = o3d.t.geometry.PointCloud(O3D_DEVICE)
	o3d_cloud.point.positions = o3d.core.Tensor(points, O3D_DTYPE, O3D_DEVICE)

	return o3d_cloud

def o3d_to_ros_msg_xyz(o3d_cloud, frame_id):
	"""
	Convert an Open3D Tensor-based PointCloud (x, y, z) to a ROS2 PointCloud2 message efficiently.
	"""

	points = o3d_cloud.point.positions.numpy().astype(NP_DTYPE)

	structured_array = np.zeros(len(points), dtype=[
		('x', NP_DTYPE), ('y', NP_DTYPE), ('z', NP_DTYPE),
		('intensity', NP_DTYPE), ('rgb', NP_DTYPE), ('label', np.uint32)
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

def crop_x(cloud, amplitude_threshold=0.5):
	"""
	Crop a point cloud based on the x-axis margin.
	"""

	positions_np = cloud.point.positions.numpy()

	mask = np.abs(positions_np[:, 0]) < amplitude_threshold
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

def crop_z(cloud, amplitude_threshold=0.5):
	"""
	Crop a point cloud based on the z-axis margin.
	"""

	positions_np = cloud.point.positions.numpy()

	mask = np.abs(positions_np[:, 2]) < amplitude_threshold
	cloud = cloud.select_by_index(np.where(mask)[0])

	return cloud

def filter_points_floor(cloud, distance_threshold=0.2, ransac_n=3, num_iterations=50):
	"""
	Filter the floor from a point cloud using RANSAC plane segmentation.
	"""

	_, inliers = cloud.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)

	cloud_no_floor = cloud.select_by_index(inliers, invert=True)

	return cloud_no_floor

def filter_points_outliers(cloud, neighbors=20, std_ratio=2.0):
	"""
	Filter the outliers from a point cloud using statistical outlier removal.
	"""

	cloud_filtered, _ = cloud.remove_statistical_outliers(nb_neighbors=neighbors, std_ratio=std_ratio)

	return cloud_filtered

def filter_points_objects(cloud, timestamp, eps=0.5, min_points=40):
	"""
	Filter the objects from a point cloud
	"""

	if len(cloud.point.positions) == 0:

		return []

	clusters = cloud.cluster_dbscan(eps=eps, min_points=min_points).numpy()

	if clusters.max() < 0:

		return []

	result_objects = []

	for idx in range(clusters.max() + 1):

		cluster_indices = np.where(clusters == idx)[0]

		cluster_cloud = cloud.select_by_index(cluster_indices)

		torch_centroid = torch.tensor(cluster_cloud.get_center().numpy(), dtype=TORCH_DTYPE, device=TORCH_DEVICE)
		torch_features = torch.tensor(get_features_fpfh(cluster_cloud).numpy(), dtype=TORCH_DTYPE, device=TORCH_DEVICE)

		result_objects.append(
			TrackedObject(
				points=cluster_cloud,
				centroid=torch_centroid,
				features=torch_features,
				timestamp=timestamp
			)
		)

	return result_objects

def objects_to_point_cloud(objects):
	"""
	Convert a list of point cloud to a single point cloud.
	"""

	cloud = o3d.t.geometry.PointCloud(O3D_DEVICE)

	if len(objects) == 0:

		cloud.point.positions = o3d.core.Tensor(np.zeros((0, 3), dtype=NP_DTYPE), dtype=O3D_DTYPE, device=O3D_DEVICE)
		return cloud

	points = np.concatenate([obj.points.point.positions.numpy() for obj in objects], axis=0)
	cloud.point.positions = o3d.core.Tensor(points, dtype=O3D_DTYPE, device=O3D_DEVICE)

	return cloud

def get_bounding_box_dimensions(bounding_box):
	"""
	Get the dimensions of a bounding box.
	"""

	extents = bounding_box.get_extent()
	extents = extents.numpy()

	return extents[0], extents[1], extents[2]

def get_features_fpfh(cloud, radius=0.03, max_nn=50):
	"""
	Get features from a cluster of points using FPFH descriptors.
	"""

	cloud.estimate_normals(max_nn=20, radius=radius)

	camera_location = o3d.core.Tensor([0.0, 0.0, 0.0], dtype=O3D_DTYPE, device=O3D_DEVICE)
	cloud.orient_normals_to_align_with_direction(camera_location)

	fpfh = o3d.t.pipelines.registration.compute_fpfh_feature(
		cloud,
		radius=radius,
		max_nn=max_nn
	)

	return fpfh

def classify_objects_by_model(objects, model, batch_size=16):
	"""
	Classify objects in a point cloud using a pre-trained model.
	"""

	if not objects:

		return objects

	all_points = []

	for obj in objects:

		points = obj.points.point.positions.numpy()
		normalized_points = normalize_point_cloud_tensor(torch.tensor(points, dtype=TORCH_DTYPE, device=TORCH_DEVICE))
		all_points.append(normalized_points)

	predictions = []

	with torch.no_grad():

		for i in range(0, len(all_points), batch_size):

			batch = all_points[i:i + batch_size]

			batch_tensor = torch.stack(batch).to(model.device)

			outputs = model.model(batch_tensor)

			batch_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
			predictions.extend(batch_predictions)

	for idx, obj in enumerate(objects):

		obj.label = predictions[idx]

	return objects

def filter_objects_by_label(objects, filter_labels={0, 1, 2, 3, 4, 5, 6, 7, 8}):

	labels = np.array([obj.label for obj in objects])
	mask = np.isin(labels, list(filter_labels))

	return [obj for i, obj in enumerate(objects) if mask[i]]

def get_openstreetmap_data(latitude, longitude, radius=50):
	"""
	Request OpenStreetMap data for a given latitude and longitude.
	"""

	url = "https://overpass-api.de/api/interpreter"
	query = f"""
	[out:json];
	(
		way(around:{radius},{latitude},{longitude})["maxspeed"];
		way(around:{radius},{latitude},{longitude})["maxheight"];
		way(around:{radius},{latitude},{longitude})["maxwidth"];
		way(around:{radius},{latitude},{longitude})["maxlength"];
		way(around:{radius},{latitude},{longitude})["maxweight"];
	);
	out;
	"""

	response = requests.get(url, params={"data": query})
	data = response.json()

	return data['elements'][0]['tags']

def objects_to_bounding_boxes(objects, frame_id):
	"""
	Create marker array messages for all objects' bounding boxes.
	"""

	marker_array = MarkerArray()

	# Remove all previous markers using a DELETEAll action
	marker = Marker()
	marker.header.frame_id = frame_id
	marker.action = Marker.DELETEALL
	marker_array.markers.append(marker)

	for i, obj in enumerate(objects):

		# Compute bounding box
		bbox = obj.compute_bounding_box()

		# Create marker
		marker = Marker()
		marker.header.frame_id = frame_id
		marker.header.stamp = rclpy.time.Time().to_msg()
		marker.ns = "bounding_boxes"
		marker.id = obj.id
		marker.type = Marker.CUBE
		marker.action = Marker.ADD

		# Set position (center of bounding box)
		center = bbox.get_center().numpy()

		marker.pose.position.x = float(center[0])
		marker.pose.position.y = float(center[1])
		marker.pose.position.z = float(center[2])

		marker.pose.orientation.w = 1.0

		# Set dimensions
		x, y, z = get_bounding_box_dimensions(bbox)

		marker.scale.x = float(x)
		marker.scale.y = float(y)
		marker.scale.z = float(z)

		# Set color based on object label or other properties
		# Here using a simple color scheme: static objects are green, moving are red
		if obj.is_static():

			marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.3)

		else:

			marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.3)

		# Add text label with ID and other info
		text_marker = Marker()
		text_marker.header = marker.header
		text_marker.ns = "object_labels"
		text_marker.id = obj.id
		text_marker.type = Marker.TEXT_VIEW_FACING
		text_marker.action = Marker.ADD
		text_marker.pose.position.x = float(center[0])
		text_marker.pose.position.y = float(center[1])
		text_marker.pose.position.z = float(center[2]) + 0.5
		text_marker.scale.z = 0.15	# Text size
		text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
		text_marker.text = f"ID: {obj.id} | {obj.label} | {obj.speed:.2f} m/s"

		# Add markers to the array
		marker_array.markers.append(marker)
		marker_array.markers.append(text_marker)

	return marker_array