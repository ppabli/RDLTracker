from tfg.utils import *
from tfg.tracked_object import TrackedObject
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2
import time
import json
import numpy as np

class LidarProcessor(Node):

	def __init__(self):
		"""
		Initialize the node.
		"""

		super().__init__('lidar_node')

		self.setup_parameters()
		self.setup_qos_and_topics()
		self.setup_objects()

		self.get_logger().info("Lidar Processor node has been initialized.")

	def setup_parameters(self):
		"""
		Set up parameters for the node.
		"""

		self.real_time_constraint = self.declare_parameter('real_time_constraint', 0.1).value
		self.debug_mode = self.declare_parameter('debug_mode', False).value
		self.input_topic = self.declare_parameter('input_topic', '/velodyne_points').value
		self.frame_id = self.declare_parameter('frame_id', 'velodyne').value
		self.max_tracked_objects = self.declare_parameter('max_tracked_objects', 10).value
		self.max_tracked_objects_age = self.declare_parameter('max_tracked_objects_age', 1).value
		self.generate_bounding_boxes = self.declare_parameter('generate_bounding_boxes', False).value
		self.use_oriented_bounding_boxes = self.declare_parameter('use_oriented_bounding_boxes', False).value
		self.calculate_speed = self.declare_parameter('calculate_speed', False).value

		#TODO: Add more parameters for the filtering process. Distances, eps, min_points, etc. This parameters can be set in the launch file and can be used to configure the filtering process instead of hardcoding the values. This is for temporal use only.

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
			self.listener_callback,
			qos_profile
		)

		self.publisher = self.create_publisher(PointCloud2, '/filtered_pointcloud', 10)

	def setup_objects(self):
		"""
		Set up the objects needed for the processing.
		"""

		self.tracked_objects = []

	def associate_objects(self, clusters_info, timestamp):
		"""
		Asociate the clusters with the already tracked objects.
		"""

		# Clean the objects that are too old
		self.tracked_objects = [obj for obj in self.tracked_objects if timestamp - obj.timestamp < self.max_tracked_objects_age]

		# Associate the clusters with the tracked objects
		for cluster in clusters_info:

			# Check if the cluster is already associated with an object
			associated = False

			for obj in self.tracked_objects:

				delta_x = np.linalg.norm(obj.centroid - cluster['centroid'])

				if delta_x < 0.5: #TODO This value should be a reviewed

					obj.update(cluster['centroid'], cluster['points'], timestamp, self.calculate_speed, delta_x)
					associated = True
					break

			# If the cluster is not associated with any object, create a new object
			if not associated and len(self.tracked_objects) < self.max_tracked_objects:

				new_obj = TrackedObject(cluster['centroid'], cluster['points'], timestamp)
				self.tracked_objects.append(new_obj)

		# Print the tracked objects
		if self.debug_mode:
			for obj in self.tracked_objects:
				self.get_logger().info(str(obj))

	def listener_callback(self, msg):
		"""
		Callback function triggered when a new PointCloud2 message is received.
		"""

		start_processing = time.time()

		start_time = time.time()
		cloud = ros2_msg_to_o3d_xyz(msg)
		ros_to_o3d_time = time.time() - start_time

		start_time = time.time()

		filtered_cloud = filter_points_downsample(cloud)

		#TODO Crop the point cloud due to recording issues, this may not be necessary in future recordings
		filtered_cloud = crop_y(filtered_cloud, 2.75)

		filtered_cloud = filter_points_by_distance(filtered_cloud, 6) #TODO This value should be a parameter
		filtered_cloud = filter_points_floor(filtered_cloud)

		filtered_cloud, clusters_info = filter_points_objects(filtered_cloud, object_limit=self.max_tracked_objects)

		self.get_logger().info(f"Number of clusters: {len(clusters_info)}")

		timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
		self.associate_objects(clusters_info, timestamp)

		if self.generate_bounding_boxes:

			boxes = [track_obj.compute_bouding_box(self.use_oriented_bounding_boxes) for track_obj in self.tracked_objects]

			if self.debug_mode:

				for box in boxes:

					self.get_logger().info(f"Bounding box | Dim: {get_bounding_box_dimensions(box, self.use_oriented_bounding_boxes)}")

		process_time = time.time() - start_time

		start_time = time.time()
		filtered_msg = o3d_to_ros_msg_xyz(filtered_cloud, self.frame_id)
		o3d_to_ros_time = time.time() - start_time

		self.publisher.publish(filtered_msg)

		processing_time = time.time() - start_processing

		if self.debug_mode:

			self.print_debug(cloud, filtered_cloud, ros_to_o3d_time, process_time, o3d_to_ros_time, processing_time)

	def print_debug(self, cloud, filtered_cloud, ros_to_o3d_time, process_time, o3d_to_ros_time, processing_time):
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
