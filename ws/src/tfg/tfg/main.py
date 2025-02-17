import rclpy
from rclpy.executors import MultiThreadedExecutor
from tfg.lidar_node import LidarProcessor

def main(args=None):

	# Initialize ROS2 and create an instance of LidarProcessor node
	rclpy.init(args=args)
	node = LidarProcessor()

	# Use MultiThreadedExecutor to handle the callbacks
	executor = MultiThreadedExecutor()
	executor.add_node(node)

	# Run the executor to handle node operations
	try:

		executor.spin()

	except KeyboardInterrupt:

		pass

	# Cleanup the node and shut down ROS2
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':

	main()