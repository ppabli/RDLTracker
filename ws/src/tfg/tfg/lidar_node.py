import json
import time

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


class LidarProcessor(Node):

	def __init__(self):
		"""
		Initialize the node.
		"""

		super().__init__('lidar_node')

		self.setup_parameters()
		self.setup_objects()
		self.setup_qos_and_topics()

		self.get_logger().info("Lidar Processor node has been initialized.")

	def setup_parameters(self):
		"""
		Set up parameters for the node.
		"""

		self.real_time_constraint = self.declare_parameter('real_time_constraint', 0.1).value
		self.debug_mode = self.declare_parameter('debug_mode', False).value
		self.input_topic = self.declare_parameter('input_topic', '/livox/lidar').value
		self.frame_id = self.declare_parameter('frame_id', 'livox_frame').value
		self.max_tracked_objects = self.declare_parameter('max_tracked_objects', 10).value
		self.max_tracked_objects_age = self.declare_parameter('max_tracked_objects_age', 1).value
		self.generate_bounding_boxes = self.declare_parameter('generate_bounding_boxes', False).value
		self.use_oriented_bounding_boxes = self.declare_parameter('use_oriented_bounding_boxes', False).value
		self.calculate_speed = self.declare_parameter('calculate_speed', False).value
		self.notify_on_speed = self.declare_parameter('notify_on_speed', False).value
		self.notify_on_width = self.declare_parameter('notify_on_width', False).value
		self.notify_on_height = self.declare_parameter('notify_on_height', False).value
		self.notify_on_length = self.declare_parameter('notify_on_length', False).value
		self.notify_on_weight = self.declare_parameter('notify_on_weight', False).value
		self.gps_coordinates = self.declare_parameter('gps_coordinates', [0.0, 0.0]).value

		params = {
			"Real-time constraint": f"{self.real_time_constraint} seconds",
			"Debug mode": self.debug_mode,
			"Input topic": self.input_topic,
			"Frame ID": self.frame_id,
			"Max tracked objects": self.max_tracked_objects,
			"Max tracked objects age": f"{self.max_tracked_objects_age} seconds",
			"Generate bounding boxes": f"{self.generate_bounding_boxes}",
			"Use oriented bounding boxes": f"{self.use_oriented_bounding_boxes}",
			"Calculate speed": f"{self.calculate_speed}",
			"Notify on speed": f"{self.notify_on_speed}",
			"Notify on width": f"{self.notify_on_width}",
			"Notify on height": f"{self.notify_on_height}",
			"Notify on length": f"{self.notify_on_length}",
			"Notify on weight": f"{self.notify_on_weight}",
			"GPS coordinates": self.gps_coordinates
		}

		self.get_logger().info(f"Node parameters: {json.dumps(params, indent=4)}")

	def setup_qos_and_topics(self):
		"""
		Configure the Quality of Service (QoS) profile and set up the ROS 2 topics.
		"""

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

		self.publisher = self.create_publisher(PointCloud2, '/filtered_pointcloud', 10)

	def setup_objects(self):
		"""
		Set up the objects needed for the processing.
		"""

		cfg_path = "/home/pablo/Downloads/ml3d_configs/pointpillars_kitti.yml"
		ckpt_path = "/home/pablo/Downloads/pointpillars_kitti_202012221652utc.pth"

		classification_model_path = "/home/pablo/Desktop/Model/model2_best_model.pth"

		cfg = _ml3d.utils.Config.load_from_file(cfg_path)

		open3d_model = ml3d.models.PointPillars(**cfg.model, device="cpu")
		pipeline = ml3d.pipelines.ObjectDetection(open3d_model, device="cpu", **cfg.pipeline)
		pipeline.load_ckpt(ckpt_path=ckpt_path)

		classification_model = PointNetTrainer(num_classes=len(LABEL_MAP))
		classification_model.load_model(classification_model_path)

		classification_model.model.eval()

		self.pipeline = pipeline
		self.classification_model = classification_model

		self.last_assigned_id = -1
		self.tracked_objects = []

		return

		map_data = get_openstreetmap_data(self.gps_coordinates[0], self.gps_coordinates[1])

		self.max_speed = map_data.get('maxspeed', np.inf)
		self.max_width = map_data.get('maxwidth', np.inf)
		self.max_height = map_data.get('maxheight', np.inf)
		self.max_length = map_data.get('maxlength', np.inf)
		self.max_weight = map_data.get('maxweight', np.inf)

		self.get_logger().info(f"Object detection pipeline has been loaded.")
		self.get_logger().info(f"Classification model has been loaded.")
		self.get_logger().info(f"Map data: {json.dumps(map_data, indent=4)}")

	def listener_callback_hybrid(self, msg):

		start_processing = time.time()

		start_time = time.time()
		cloud = ros2_msg_to_o3d_xyz(msg)
		ros_to_o3d_time = time.time() - start_time

		start_time = time.time()

		filtered_cloud = filter_points_downsample(cloud)
		filtered_cloud = filter_points_floor(filtered_cloud)
		filtered_cloud = filter_points_outliers(filtered_cloud)

		filtered_cloud = crop_y(filtered_cloud, 6)
		#filtered_cloud = filter_points_by_distance(filtered_cloud, 6)

		objects = filter_points_objects(filtered_cloud)

		objects = classify_objects_by_model(objects, self.classification_model)

		filtered_objects = filter_objects_by_label(objects)

		timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
		self.track_objects(filtered_objects, timestamp)

		for obj in self.tracked_objects:
			self.get_logger().info(f"{obj}")

		filtered_cloud = objects_to_point_cloud(self.tracked_objects)

		process_time = time.time() - start_time

		start_time = time.time()
		filtered_msg = o3d_to_ros_msg_xyz(filtered_cloud, self.frame_id)
		o3d_to_ros_time = time.time() - start_time

		self.publisher.publish(filtered_msg)

		processing_time = time.time() - start_processing

		if self.debug_mode:

			self.print_debug(ros_to_o3d_time, process_time, o3d_to_ros_time, processing_time)

	def print_debug(self, ros_to_o3d_time, process_time, o3d_to_ros_time, processing_time):
		"""
		Print debug information about the processing time and the point clouds.
		"""

		YELLOW = '\033[93m'
		RESET = '\033[0m'

		self.get_logger().info(f"Processing time: {processing_time:.4f} seconds")

		if processing_time > self.real_time_constraint:
			self.get_logger().warn(f'{YELLOW}Processing time {processing_time:.6f} seconds exceeded the constraint by {processing_time - self.real_time_constraint:.6f} seconds{RESET}')

		self.get_logger().info(f"Time for ROS to Open3D conversion: {ros_to_o3d_time:.4f} seconds")
		self.get_logger().info(f"Time for procesing: {process_time:.4f} seconds")
		self.get_logger().info(f"Time for Open3D to ROS conversion: {o3d_to_ros_time:.4f} seconds")
		self.get_logger().info(f"-------------------------------")

	def track_objects(self, objects, timestamp, w_position=0.3, w_features=0.7, max_association_cost=1.75, min_movement_threshold=0.5, max_movement_threshold=30.0, w_new_speed=0.9, w_old_speed=0.1):
		"""
		Track objects in the scene using cluster information.
		"""

		if not self.tracked_objects:

			for i, obj in enumerate(objects[:self.max_tracked_objects]):

				self.last_assigned_id += 1
				obj.id = self.last_assigned_id
				obj.timestamp = timestamp
				self.tracked_objects.append(obj)

			return

		if not objects:

			updated_tracked_objects = []

			for obj in self.tracked_objects:

				if timestamp - obj.timestamp < self.max_tracked_objects_age:

					obj.timestamp = timestamp
					updated_tracked_objects.append(obj)

			self.tracked_objects = updated_tracked_objects

			return

		cost_matrix = np.zeros((len(self.tracked_objects), len(objects)))

		for i, tracked_obj in enumerate(self.tracked_objects):

			predicted_position = tracked_obj.centroid

			if tracked_obj.speed > min_movement_threshold:

				time_since_last_update = timestamp - tracked_obj.timestamp
				predicted_position += tracked_obj.direction * tracked_obj.speed * time_since_last_update

			for j, cluster in enumerate(objects):

				position_distance = np.linalg.norm(predicted_position - cluster.centroid).item()

				feature_distance = self._calculate_feature_distance(tracked_obj.features, cluster.features)

				cost_matrix[i, j] = w_position * position_distance + w_features * feature_distance

		row_indices, col_indices = linear_sum_assignment(cost_matrix)

		assigned_tracked_indices = set()
		assigned_cluster_indices = set()

		for row_idx, col_idx in zip(row_indices, col_indices):

			if cost_matrix[row_idx, col_idx] < max_association_cost:

				self._update_tracked_object(
					self.tracked_objects[row_idx],
					objects[col_idx],
					timestamp,
					min_movement_threshold,
					max_movement_threshold,
					w_new_speed,
					w_old_speed
				)

				assigned_tracked_indices.add(row_idx)
				assigned_cluster_indices.add(col_idx)

		self.tracked_objects = [
			obj for i, obj in enumerate(self.tracked_objects)
			if i in assigned_tracked_indices or timestamp - obj.timestamp < self.max_tracked_objects_age
		]

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
		Helper method to calculate feature distance/similarity.
		"""

		feature_distance = 1.0

		tracked_features_mean = np.mean(tracked_features.numpy(), axis=0)
		cluster_features_mean = np.mean(cluster_features.numpy(), axis=0)

		dot_product = np.sum(tracked_features_mean * cluster_features_mean)

		norm_tracked = np.linalg.norm(tracked_features_mean)
		norm_cluster = np.linalg.norm(cluster_features_mean)

		if norm_tracked > 0 and norm_cluster > 0:

			similarity = dot_product / (norm_tracked * norm_cluster)
			feature_distance = 1.0 - max(0.0, min(1.0, similarity))

		return feature_distance

	def _update_tracked_object(self, tracked_obj, new_obj, timestamp, min_movement_threshold, max_movement_threshold, w_new_speed, w_old_speed, decay_rate=0.5):
		"""
		Helper method to update a tracked object with new data.
		"""

		old_centroid = tracked_obj.centroid
		new_centroid = new_obj.centroid

		distance_moved = np.linalg.norm(new_centroid - old_centroid).item()
		time_since_last_update = timestamp - tracked_obj.timestamp

		if distance_moved > min_movement_threshold and distance_moved < max_movement_threshold:

			direction = (new_centroid - old_centroid) / distance_moved

			if self.calculate_speed:

				new_speed = distance_moved / time_since_last_update

				tracked_obj.update_speed(
					new_speed=w_new_speed * new_speed + w_old_speed * tracked_obj.speed,
					new_direction=direction
				)

		else:

			if self.calculate_speed:

				current_speed = tracked_obj.speed

				if current_speed > 0:

					tracked_obj.update_speed(
						new_speed=current_speed * decay_rate,
						new_direction=tracked_obj.direction
					)

		tracked_obj.update(
			centroid=new_centroid,
			points=new_obj.points,
			timestamp=timestamp,
			features=new_obj.features
		)