from tfg.utils import *
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2
import time

YELLOW = '\033[93m'
RESET = '\033[0m'

class LidarProcessor(Node):

	def __init__(self):
		"""
		Initialize the node.
		"""

		super().__init__('lidar_node')

		self.setup_parameters()
		self.setup_qos_and_topics()

	def setup_parameters(self):
		"""
		Set up parameters for the node.
		"""

		self.real_time_constraint = self.declare_parameter('real_time_constraint', 0.1).value
		self.debug_mode = self.declare_parameter('debug_mode', False).value
		self.input_topic = self.declare_parameter('input_topic', '/velodyne_points').value
		self.frame_id = self.declare_parameter('frame_id', 'velodyne').value
		self.generate_bounding_boxes = self.declare_parameter('generate_bounding_boxes', False).value
		self.use_oriented_bounding_boxes = self.declare_parameter('use_oriented_bounding_boxes', True).value

		#TODO: Add more parameters for the filtering process. Distances, eps, min_points, etc. This parameters can be set in the launch file and can be used to configure the filtering process instead of hardcoding the values. This is for temporal use only.

		params = {
			"Real-time constraint": f"{self.real_time_constraint} seconds",
			"Debug mode": self.debug_mode,
			"Input topic": self.input_topic,
			"Frame ID": self.frame_id,
			"Generate bounding boxes": f"{self.generate_bounding_boxes}",
			"Use oriented bounding boxes": f"{self.use_oriented_bounding_boxes}"
		}

		self.get_logger().info(f"Node parameters: {params}")

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

		# Crop the point cloud due to recording issues, this may not be necessary in future recordings
		filtered_cloud = crop_y(filtered_cloud, 2.75)

		filtered_cloud = filter_points_by_distance(filtered_cloud, 5.5)
		filtered_cloud = filter_points_floor(filtered_cloud)

		filtered_cloud, clusters_info = filter_points_objects(filtered_cloud)

		if self.generate_bounding_boxes:

			boxes = compute_bounding_boxes(clusters_info, self.use_oriented_bounding_boxes)

			self.get_logger().info("\n".join([
				f"Bounding box {i} | Dimensions: {box.extent if self.use_oriented_bounding_boxes else box.get_extent()}"
				for i, box in enumerate(boxes)
			]))

		filter_time = time.time() - start_time

		start_time = time.time()
		filtered_msg = o3d_to_ros_msg_xyz(filtered_cloud, self.frame_id)
		o3d_to_ros_time = time.time() - start_time

		self.publisher.publish(filtered_msg)

		processing_time = time.time() - start_processing

		if self.debug_mode:

			self.print_debug(cloud, filtered_cloud, ros_to_o3d_time, filter_time, o3d_to_ros_time, processing_time)

	def print_debug(self, cloud, filtered_cloud, ros_to_o3d_time, filter_time, o3d_to_ros_time, processing_time):
		"""
		Print debug information about the processing time and the point clouds.
		"""

		self.get_logger().info(f"Processing time: {processing_time:.4f} seconds")

		if processing_time > self.real_time_constraint:
			self.get_logger().warn(f'{YELLOW}Processing time {processing_time:.6f} seconds exceeded the constraint by {processing_time - self.real_time_constraint:.6f} seconds{RESET}')

		self.get_logger().info(f"Time for ROS to Open3D conversion: {ros_to_o3d_time:.4f} seconds")
		self.get_logger().info(f"Time for procesing: {filter_time:.4f} seconds")
		self.get_logger().info(f"Time for Open3D to ROS conversion: {o3d_to_ros_time:.4f} seconds")
		self.get_logger().info(f"-------------------------------")
