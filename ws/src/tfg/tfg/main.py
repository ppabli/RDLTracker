import rclpy
from rclpy.executors import MultiThreadedExecutor
from tfg.lidar_node import LidarProcessor

def main(args=None):

	rclpy.init(args=args)
	node = LidarProcessor()

	executor = MultiThreadedExecutor()
	executor.add_node(node)

	try:

		executor.spin()

	except KeyboardInterrupt:

		pass

	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':

	main()