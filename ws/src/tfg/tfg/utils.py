from collections import deque

import numpy as np
import open3d as o3d
import rclpy
import requests
import sensor_msgs_py.point_cloud2 as pc2
import torch
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from tfg.model import normalize_point_cloud_tensor
from tfg.tracked_object import TrackedObject

DEVICE = o3d.core.Device("CPU:0")

def ros2_msg_to_o3d_xyz(ros_cloud):
	"""
	Convert the (x, y, z) ROS2 PointCloud2 message to an Open3D Tensor-based (x,y,z) PointCloud.
	"""

	dtype = o3d.core.float32

	dtype_np = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
	cloud_array = np.array(list(pc2.read_points(ros_cloud, field_names=("x", "y", "z"), skip_nans=True)), dtype=dtype_np)

	points = np.stack((cloud_array['x'], cloud_array['y'], cloud_array['z']), axis=-1)

	o3d_cloud = o3d.t.geometry.PointCloud(DEVICE)
	o3d_cloud.point.positions = o3d.core.Tensor(points, dtype, o3d_cloud.device)

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

def filter_points_floor(cloud, distance_threshold=0.2, ransac_n=3, num_iterations=100):
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

def euclidean_clustering(points, eps=0.75, min_samples=30):
	"""
	Perform K-D tree-based Euclidean clustering on a set of points.
	"""

	tree = cKDTree(points)
	labels = -np.ones(points.shape[0], dtype=int)
	cluster_id = 0

	for i in range(points.shape[0]):

		if labels[i] == -1:

			neighbors = tree.query_ball_point(points[i], eps)

			if len(neighbors) < min_samples:

				continue

			queue = deque(neighbors)
			labels[i] = cluster_id

			while queue:

				idx = queue.popleft()

				if labels[idx] == -1:

					labels[idx] = cluster_id
					queue.extend(tree.query_ball_point(points[idx], eps))

			cluster_id += 1

	return labels

def filter_points_objects_ai(cloud, pipeline):

	points = cloud.point.positions.numpy()
	intensity = np.zeros((points.shape[0], 1), dtype=np.float32)

	final_points = np.concatenate([points, intensity], axis=1)

	data = {
		"point": final_points,
	}

	predictions = pipeline.run_inference(data)

	return predictions

def filter_points_objects(cloud, eps=0.75, min_points=50):
	"""
	Filter the objects from a point cloud using DBSCAN clustering.
	"""

	clusters = cloud.cluster_dbscan(eps=eps, min_points=min_points).numpy()

	if clusters.max() < 0:

		return []

	result_objects = []

	for idx in range(clusters.max() + 1):

		cluster_indices = np.where(clusters == idx)[0]

		cluster_cloud = cloud.select_by_index(cluster_indices)

		filtered_cluster = filter_points_outliers(cluster_cloud)

		result_objects.append(
			TrackedObject(
				points=filtered_cluster,
				centroid=filtered_cluster.get_center(),
				features=get_features_fpfh(filtered_cluster)
			)
		)

	return result_objects

def objects_to_point_cloud(objects):
	"""
	Convert a list of point cloud to a single point cloud.
	"""

	cloud = o3d.t.geometry.PointCloud(DEVICE)

	if len(objects) == 0:

		cloud.point.positions = o3d.core.Tensor(np.zeros((0, 3), dtype=np.float32), dtype=o3d.core.float32, device=cloud.device)
		return cloud

	points = np.concatenate([obj.points.point.positions.numpy() for obj in objects], axis=0)
	cloud.point.positions = o3d.core.Tensor(points, dtype=o3d.core.float32, device=cloud.device)

	return cloud

def filter_objects_by_volume(objects, min_vol=2, max_vol=50.0, use_oriented_bounding_boxes=False):
	"""
	Filter the objects based on their volume.
	"""

	boxes = [obj.compute_bouding_box(use_oriented_bounding_boxes) for obj in objects]
	volumes = [get_bounding_box_volume(box, use_oriented_bounding_boxes) for box in boxes]

	mask = np.logical_and(np.array(volumes) > min_vol, np.array(volumes) < max_vol)
	objects = [obj for obj, m in zip(objects, mask) if m]

	return objects

def get_bounding_box_volume(bounding_box, use_oriented=False):
	"""
	Get the volume of a bounding box.
	"""

	extents = bounding_box.extent if use_oriented else bounding_box.get_extent()

	return extents[0] * extents[1] * extents[2]

def get_bounding_box_dimensions(bounding_box, use_oriented=False):
	"""
	Get the dimensions of a bounding box.
	"""

	extents = bounding_box.extent if use_oriented else bounding_box.get_extent()

	return extents[0], extents[1], extents[2]

def get_features_fpfh(cloud, radius=0.03, max_nn=50):
	"""
	Get features from a cluster of points using FPFH descriptors.
	"""

	cloud.estimate_normals(max_nn=20, radius=radius)

	camera_location = o3d.core.Tensor([0.0, 0.0, 0.0], dtype=o3d.core.Dtype.Float32, device=cloud.device)
	cloud.orient_normals_to_align_with_direction(camera_location)

	fpfh = o3d.t.pipelines.registration.compute_fpfh_feature(
		cloud,
		radius=radius,
		max_nn=max_nn
	)

	return fpfh

def classify_objects_by_features(objects, use_oriented=False):
	"""
	Classify objects in a point cloud based on their volume.
	"""

	for obj in objects:

		bounding_box = obj.compute_bouding_box(use_oriented)

		volume = get_bounding_box_volume(bounding_box, use_oriented)

		features_norm = np.linalg.norm(obj.features, axis=1)
		features_norm = features_norm[features_norm > 0]

		feature_variance = np.var(obj.features, axis=0)

		complexity = np.mean(feature_variance)

		#TODO Fix to proper classification

	return objects

def classify_objects_by_model(objects, model, batch_size=16):
	"""
	Classify objects in a point cloud using a pre-trained model.
	"""

	if not objects:

		return objects

	all_points = []

	for obj in objects:

		points = obj.points.point.positions.numpy()
		normalized_points = normalize_point_cloud_tensor(torch.tensor(points, dtype=torch.float32))
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

def filter_objects_by_speed(objects, min_speed=0.0, max_speed=10.0):
	"""
	Filter the objects based on their speed.
	"""

	return [obj for obj in objects if obj.speed > min_speed and obj.speed < max_speed]

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
