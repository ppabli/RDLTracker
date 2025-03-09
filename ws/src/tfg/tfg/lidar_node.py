import json
import time
import traceback
import numpy as np
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from scipy.optimize import linear_sum_assignment
from sensor_msgs.msg import PointCloud2
from tfg.model import PointNetTrainer, LABEL_MAP
from tfg.tracked_object import TrackedObject
from tfg.utils import *
from visualization_msgs.msg import MarkerArray
from tfg.constants import YELLOW, RESET


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

		# Object tracking parameters
		self.max_tracked_objects = self.declare_parameter('max_tracked_objects', 10).value
		self.max_tracked_objects_age = self.declare_parameter('max_tracked_objects_age', 1.0).value

		# Feature flags
		self.generate_bounding_boxes = self.declare_parameter('generate_bounding_boxes', False).value
		self.calculate_speed = self.declare_parameter('calculate_speed', True).value

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

		# Log parameters for debugging
		params = {
			"Real-time constraint": f"{self.real_time_constraint} seconds",
			"Debug mode": self.debug_mode,
			"Input topic": self.input_topic,
			"Frame ID": self.frame_id,
			"Max tracked objects": self.max_tracked_objects,
			"Max tracked objects age": f"{self.max_tracked_objects_age} seconds",
			"Generate bounding boxes": f"{self.generate_bounding_boxes}",
			"Calculate speed": f"{self.calculate_speed}",
			"Notify on speed": f"{self.notify_on_speed}",
			"Notify on width": f"{self.notify_on_width}",
			"Notify on height": f"{self.notify_on_height}",
			"Notify on length": f"{self.notify_on_length}",
			"Notify on weight": f"{self.notify_on_weight}",
			"GPS coordinates": self.gps_coordinates,
			"Position weight": self.position_weight,
			"Feature weight": self.feature_weight,
			"Max association cost": self.max_association_cost
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

	def setup_objects(self):
		"""Initialize ML models and tracking objects"""

		try:

			classification_model_path = "/home/pablo/Desktop/Model/multi_object_model.pth"
			classification_model = PointNetTrainer(num_classes=len(LABEL_MAP))
			classification_model.load_model(classification_model_path)
			classification_model.model.eval()

			self.classification_model = classification_model
			self.get_logger().info("Classification model loaded successfully.")

		except Exception as e:

			self.get_logger().error(f"Failed to load classification model: {e}")
			self.classification_model = None

		self.last_assigned_id = -1
		self.tracked_objects = []

		try:

			if all(coord != 0.0 for coord in self.gps_coordinates):

				map_data = get_openstreetmap_data(self.gps_coordinates[0], self.gps_coordinates[1])

				self.max_speed = map_data.get('maxspeed', np.inf)
				self.max_width = map_data.get('maxwidth', np.inf)
				self.max_height = map_data.get('maxheight', np.inf)
				self.max_length = map_data.get('maxlength', np.inf)
				self.max_weight = map_data.get('maxweight', np.inf)

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

			start_processing = time.time()

			# Convert ROS message to Open3D point cloud
			start_time = time.time()
			cloud = ros2_msg_to_o3d_xyz(msg)
			ros_to_o3d_time = time.time() - start_time

			# Process the point cloud
			start_time = time.time()

			# Apply sequential filtering operations
			self.process_point_cloud(cloud)

			# Prepare output point cloud
			filtered_cloud = objects_to_point_cloud(self.tracked_objects)
			process_time = time.time() - start_time

			# Convert back to ROS message
			start_time = time.time()
			filtered_msg = o3d_to_ros_msg_xyz(filtered_cloud, self.frame_id)
			o3d_to_ros_time = time.time() - start_time

			# Publish detected objects as bounding boxes
			if self.generate_bounding_boxes and self.tracked_objects:

				marker_array = objects_to_bounding_boxes(self.tracked_objects, self.frame_id)
				self.bbox_publisher.publish(marker_array)

			# Publish filtered point cloud
			self.publisher.publish(filtered_msg)

			# Calculate total processing time
			processing_time = time.time() - start_processing

			# Print debug information if enabled
			if self.debug_mode:

				self.print_debug(ros_to_o3d_time, process_time, o3d_to_ros_time, processing_time)

		except Exception as e:

			self.get_logger().error(f"Error processing point cloud: {e}")
			print(traceback.format_exc())

	def process_point_cloud(self, cloud):
		"""
		Process the point cloud through a pipeline of filters.
		"""

		# Extract timestamp from current time if we don't have it from the message
		timestamp = time.time()

		# Apply filters sequentially
		filtered_cloud = filter_points_downsample(cloud)
		filtered_cloud = filter_points_floor(filtered_cloud)
		filtered_cloud = filter_points_outliers(filtered_cloud)
		filtered_cloud = crop_y(filtered_cloud, 6)
		#filtered_cloud = filter_points_by_distance(filtered_cloud, 18)

		# Extract objects from the filtered cloud
		objects = filter_points_objects(filtered_cloud, timestamp)

		# Classify objects if model is available
		if self.classification_model is not None:

			objects = classify_objects_by_model(objects, self.classification_model)

		# Filter objects by label
		filtered_objects = filter_objects_by_label(objects)

		# Track objects across frames
		self.track_objects(filtered_objects, timestamp, w_position=self.position_weight, w_features=self.feature_weight, max_association_cost=self.max_association_cost)

		return

	def print_debug(self, ros_to_o3d_time, process_time, o3d_to_ros_time, processing_time):
		"""Print debug information about processing times"""

		self.get_logger().info(f"Processing time: {processing_time:.4f} seconds")

		if processing_time > self.real_time_constraint:

			self.get_logger().warn(
				f'{YELLOW}Processing time {processing_time:.6f} seconds exceeded '
				f'the constraint by {processing_time - self.real_time_constraint:.6f} seconds{RESET}'
			)

		self.get_logger().info(f"Time for ROS to Open3D conversion: {ros_to_o3d_time:.4f} seconds")
		self.get_logger().info(f"Time for processing: {process_time:.4f} seconds")
		self.get_logger().info(f"Time for Open3D to ROS conversion: {o3d_to_ros_time:.4f} seconds")
		self.get_logger().info(f"-------------------------------")

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
					timestamp=timestamp
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
						label=objects[j].label
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
