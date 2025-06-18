import json
import os
import time
import traceback
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from scipy.optimize import linear_sum_assignment
from sensor_msgs.msg import PointCloud2
from tfg.pointnet import PointNet
from tfg.pointnet_pp import PointNetPlusPlus
from tfg.tracked_object import TrackedObject
from tfg.utils import *
from tfg.model_wrapper import ModelWrapper
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String
from tfg.constants import YELLOW, RESET
from tfg.constants import LABEL_MAP

class LidarProcessor(Node):
	"""
	Node for processing LiDAR point clouds, detecting and tracking objects.
	"""

	def __init__(self):

		super().__init__('lidar_node')

		self.setup_parameters()
		self.setup_objects()
		self.setup_qos_and_topics()

		self.get_logger().info("Lidar Processor node has been initialized.")

	def setup_parameters(self):
		"""Load and initialize parameters from ROS params"""

		# Performance parameters
		self.real_time_constraint = self.declare_parameter('real_time_constraint', 0.1).value
		self.debug_mode = self.declare_parameter('debug_mode', False).value

		# Topic and frame configuration
		self.input_topic = self.declare_parameter('input_topic', '/livox/lidar').value
		self.frame_id = self.declare_parameter('frame_id', 'livox_frame').value

		# Cropping parameters
		self.crop_x = self.declare_parameter('crop_x', 0.0).value
		self.crop_y = self.declare_parameter('crop_y', 0.0).value
		self.crop_z = self.declare_parameter('crop_z', 0.0).value

		# Object tracking parameters
		self.max_tracked_objects = self.declare_parameter('max_tracked_objects', 10).value
		self.max_tracked_objects_age = self.declare_parameter('max_tracked_objects_age', 1.0).value

		# Feature flags
		self.generate_bounding_boxes = self.declare_parameter('generate_bounding_boxes', False).value
		self.calculate_speed = self.declare_parameter('calculate_speed', False).value

		# Notification settings
		self.notify_on_speed = self.declare_parameter('notify_on_speed', False).value
		self.notify_on_width = self.declare_parameter('notify_on_width', False).value
		self.notify_on_height = self.declare_parameter('notify_on_height', False).value
		self.notify_on_length = self.declare_parameter('notify_on_length', False).value
		self.notify_on_weight = self.declare_parameter('notify_on_weight', False).value

		# Location parameters
		self.gps_coordinates = self.declare_parameter('gps_coordinates', [0.0, 0.0]).value

		# Association weights for tracking
		self.position_weight = self.declare_parameter('position_weight', 0.3).value
		self.feature_weight = self.declare_parameter('feature_weight', 0.7).value
		self.max_association_cost = self.declare_parameter('max_association_cost', 1.75).value

		# Model and tracking objects
		self.use_classification_model = self.declare_parameter('use_classification_model', False).value
		self.classification_model_weights_path = self.declare_parameter('classification_model_weights_path', '/home/pablo/Desktop/pointnet/output/pointnet_best.pth').value

		# Log parameters for debugging
		params = {
			"Real-time constraint": f"{self.real_time_constraint} seconds",
			"Debug mode": self.debug_mode,
			"Input topic": self.input_topic,
			"Frame ID": self.frame_id,
			"Crop X": self.crop_x,
			"Crop Y": self.crop_y,
			"Crop Z": self.crop_z,
			"Max tracked objects": self.max_tracked_objects,
			"Max tracked objects age": f"{self.max_tracked_objects_age} seconds",
			"Generate bounding boxes": self.generate_bounding_boxes,
			"Calculate speed": self.calculate_speed,
			"Notify on speed": self.notify_on_speed,
			"Notify on width": self.notify_on_width,
			"Notify on height": self.notify_on_height,
			"Notify on length": self.notify_on_length,
			"Notify on weight": self.notify_on_weight,
			"GPS coordinates": self.gps_coordinates,
			"Position weight": self.position_weight,
			"Feature weight": self.feature_weight,
			"Max association cost": self.max_association_cost,
			"Use classification model": self.use_classification_model,
			"Classification model weights path": self.classification_model_weights_path
		}

		self.get_logger().info(f"Node parameters: {json.dumps(params, indent=4)}")

	def setup_qos_and_topics(self):
		"""Setup QoS profiles and ROS topics"""

		qos_profile = QoSProfile(
			reliability=QoSReliabilityPolicy.BEST_EFFORT,
			history=QoSHistoryPolicy.KEEP_LAST,
			depth=10
		)

		self.subscription = self.create_subscription(
			PointCloud2,
			self.input_topic,
			self.listener_callback_hybrid,
			qos_profile
		)

		self.publisher = self.create_publisher(
			PointCloud2,
			'/filtered_pointcloud',
			10
		)

		self.bbox_publisher = self.create_publisher(
			MarkerArray,
			'/detected_objects',
			10
		)

		self.notification_publisher = self.create_publisher(
			String,
			'/notifications',
			10
		)

	def setup_objects(self):
		"""Initialize ML models and tracking objects"""

		try:

			if self.use_classification_model:

				classification_model_path = self.classification_model_weights_path

				model_name = os.path.basename(classification_model_path).split('_')[0]

				if model_name == "pointnet":

					model = PointNet(num_classes=len(LABEL_MAP))

				elif model_name == "pointnetpp":

					model = PointNetPlusPlus(num_classes=len(LABEL_MAP))

				else:

					self.get_logger().warning(f"Unknown model name: {model_name}")
					return

				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

				checkpoint = torch.load(classification_model_path, map_location=device)
				model.load_state_dict(checkpoint['model_state_dict'], strict=False)

				model = model.to(device)

				self.classification_model = ModelWrapper(model_name, model, device)
				self.get_logger().info("Classification model loaded successfully.")

			else:

				self.classification_model = None

		except Exception as e:

			self.get_logger().error(f"Failed to load classification model: {e}")
			self.classification_model = None

		self.last_assigned_id = -1
		self.tracked_objects = []

		try:

			if all(coord != 0.0 for coord in self.gps_coordinates):

				map_data = get_openstreetmap_data(self.gps_coordinates[0], self.gps_coordinates[1])

				self.max_speed = float(map_data.get('maxspeed', np.inf))
				self.max_width = float(map_data.get('maxwidth', np.inf))
				self.max_height = float(map_data.get('maxheight', np.inf))
				self.max_length = float(map_data.get('maxlength', np.inf))
				self.max_weight = float(map_data.get('maxweight', np.inf))

				self.get_logger().info(f"Map data loaded: {json.dumps(map_data, indent=4)}")

			else:

				self.get_logger().info("No valid GPS coordinates provided, skipping map data loading.")

		except Exception as e:

			self.get_logger().warning(f"Failed to load map data: {e}")

	def listener_callback_hybrid(self, msg):
		"""
		Main callback for processing incoming point cloud messages.
		"""

		try:

			times = {}
			start_total = time.time()

			# Convert ROS message to Open3D point cloud
			start = time.time()
			cloud = ros2_msg_to_o3d_xyz(msg)
			times['ros_to_o3d'] = time.time() - start

			start = time.time()
			process_times = self.process_point_cloud(cloud, return_times=True)
			times.update(process_times)
			times['process_point_cloud'] = time.time() - start

			start = time.time()
			filtered_cloud = objects_to_point_cloud(self.tracked_objects)
			times['objects_to_point_cloud'] = time.time() - start

			# Convert back to ROS message
			start = time.time()
			filtered_msg = o3d_to_ros_msg_xyz(filtered_cloud, self.frame_id)
			times['o3d_to_ros'] = time.time() - start

			# Publish detected objects as bounding boxes
			if self.generate_bounding_boxes and self.tracked_objects:

				start = time.time()
				marker_array = objects_to_bounding_boxes(self.tracked_objects, self.frame_id)
				times['objects_to_bounding_boxes'] = time.time() - start
				self.bbox_publisher.publish(marker_array)

			# Publish filtered point cloud
			self.publisher.publish(filtered_msg)

			times['total'] = time.time() - start_total

			# Print and send debug information if enabled
			if self.debug_mode:

				self.print_debug(times)
				msg_str = json.dumps({k: f"{v:.4f}s" for k, v in times.items()}, indent=2)
				self.notification_publisher.publish(format_message(msg_str))

			if self.notify_on_speed and hasattr(self, 'max_speed') and self.max_speed != np.inf:

				filtered_objects = [
					obj for obj in self.tracked_objects if obj.get_label() != -1 and obj.speed > self.max_speed
				]

				for obj in filtered_objects:

					notification = f"Object {obj.id} exceeds speed limit: {obj.speed:.2f} m/s"
					self.notification_publisher.publish(format_message(notification))

		except Exception as e:

			self.get_logger().error(f"Error processing point cloud: {e}")
			print(traceback.format_exc())

	def process_point_cloud(self, cloud, return_times=False):
		"""
		Process the point cloud through a pipeline of filters. Si return_times=True, devuelve los tiempos de cada bloque.
		"""

		times = {}

		# Extract timestamp from current time
		timestamp = time.time()

		start = time.time()
		filtered_cloud = filter_points_downsample(cloud)
		times['downsample'] = time.time() - start

		start = time.time()
		filtered_cloud = filter_points_floor(filtered_cloud)
		times['floor'] = time.time() - start

		start = time.time()
		filtered_cloud = filter_points_outliers(filtered_cloud)
		times['outliers'] = time.time() - start

		start = time.time()
		filtered_cloud = crop_x(filtered_cloud, self.crop_x)
		times['crop_x'] = time.time() - start

		start = time.time()
		filtered_cloud = crop_y(filtered_cloud, self.crop_y)
		times['crop_y'] = time.time() - start

		start = time.time()
		filtered_cloud = crop_z(filtered_cloud, self.crop_z)
		times['crop_z'] = time.time() - start

		start = time.time()
		filtered_cloud = filter_points_by_distance(filtered_cloud, 18)
		times['distance'] = time.time() - start

		start = time.time()
		objects = filter_points_objects(filtered_cloud, timestamp)
		times['objects_extraction'] = time.time() - start

		# Classify objects if model is available
		if self.classification_model is not None:

			start = time.time()
			objects = classify_objects_by_model(objects, self.classification_model)
			times['classification'] = time.time() - start

			start = time.time()
			objects = filter_objects_by_label(objects)
			times['label_filter'] = time.time() - start

		start = time.time()
		self.track_objects(objects, timestamp, w_position=self.position_weight, w_features=self.feature_weight, max_association_cost=self.max_association_cost)
		times['tracking'] = time.time() - start

		if return_times:

			return times

		return

	def print_debug(self, times):
		"""Print debug information to the console."""

		self.get_logger().info("--- Debug Information ---")

		for k, v in times.items():

			self.get_logger().info(f"{k}: {v:.4f} s")

		self.get_logger().info("-----------------------")

	def track_objects(self, objects, timestamp, w_position=0.3, w_features=0.7, max_association_cost=1.75):
		"""
		Track objects across frames using the Hungarian algorithm.
		"""

		# If we have no tracked objects yet, initialize with current detections
		if not self.tracked_objects:

			for i, obj in enumerate(objects[:self.max_tracked_objects]):

				self.last_assigned_id += 1
				obj.id = self.last_assigned_id

				self.tracked_objects.append(obj)

			return

		# If we have no detections in current frame, update timestamps and remove old tracks
		if not objects:

			updated_tracked_objects = []

			for obj in self.tracked_objects:

				if timestamp - obj.timestamp < self.max_tracked_objects_age:

					updated_tracked_objects.append(obj)

			self.tracked_objects = updated_tracked_objects

			return

		cost_matrix = np.zeros((len(self.tracked_objects), len(objects)))

		for i, tracked_obj in enumerate(self.tracked_objects):

			# Use Kalman filter to predict position at current timestamp
			time_since_last_update = timestamp - tracked_obj.timestamp
			predicted_position = tracked_obj.predict_position(time_since_last_update)

			for j, cluster in enumerate(objects):

				position_distance = np.linalg.norm(predicted_position - cluster.centroid).item()

				feature_distance = self._calculate_feature_distance(tracked_obj.features, cluster.features)

				cost_matrix[i, j] = w_position * position_distance + w_features * feature_distance

		# Apply Hungarian algorithm for optimal assignment
		row_indices, col_indices = linear_sum_assignment(cost_matrix)

		assigned_tracked_indices = set()
		assigned_cluster_indices = set()

		# Process matched pairs
		for row_idx, col_idx in zip(row_indices, col_indices):

			if cost_matrix[row_idx, col_idx] < max_association_cost:

				tracked_obj = self.tracked_objects[row_idx]
				new_obj = objects[col_idx]

				# Update object with new measurements
				tracked_obj.update(
					centroid=new_obj.centroid,
					points=new_obj.points,
					features=new_obj.features,
					timestamp=timestamp,
					label=new_obj.label,
					confidence=new_obj.confidence,
				)

				assigned_tracked_indices.add(row_idx)
				assigned_cluster_indices.add(col_idx)

		# Filter tracked objects to remove old ones
		self.tracked_objects = [
			obj for i, obj in enumerate(self.tracked_objects)
			if i in assigned_tracked_indices or timestamp - obj.timestamp < self.max_tracked_objects_age
		]

		# Add new objects if there's room
		available_slots = self.max_tracked_objects - len(self.tracked_objects)

		if available_slots > 0:

			for j in range(len(objects)):

				if j not in assigned_cluster_indices and available_slots > 0:

					self.last_assigned_id += 1

					new_object = TrackedObject(
						id=self.last_assigned_id,
						centroid=objects[j].centroid,
						points=objects[j].points,
						timestamp=timestamp,
						features=objects[j].features,
						label=objects[j].label,
						confidence=objects[j].confidence
					)

					self.tracked_objects.append(new_object)
					available_slots -= 1

	def _calculate_feature_distance(self, tracked_features, cluster_features):
		"""
		Helper method to calculate feature distance/similarity using torch tensors.
		"""

		feature_distance = 1.0

		tracked_features_mean = torch.mean(tracked_features, dim=0)
		cluster_features_mean = torch.mean(cluster_features, dim=0)

		dot_product = torch.sum(tracked_features_mean * cluster_features_mean)

		norm_tracked = torch.norm(tracked_features_mean)
		norm_cluster = torch.norm(cluster_features_mean)

		if norm_tracked > 0 and norm_cluster > 0:

			similarity = dot_product / (norm_tracked * norm_cluster)
			feature_distance = 1.0 - max(0.0, min(1.0, similarity.item()))

		return feature_distance
