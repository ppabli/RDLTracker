from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

	return LaunchDescription([

		Node(
			package='tfg',
			executable='lidar_node',
			name='lidar_processor',
			output='screen',
			parameters=[
				{'real_time_constraint': 0.1},
				{'debug_mode': True},
				{'input_topic': '/livox/lidar'},
				{'frame_id': 'livox_frame'},
				{'generate_bounding_boxes': True},
				{'use_oriented_bounding_boxes': True},
				#TODO: Add more parameters for the filtering process. Distances, eps, min_points, etc. This parameters can be set in the launch file and can be used to configure the filtering process instead of hardcoding the values. This is for temporal use only.
			]
		)

	])
