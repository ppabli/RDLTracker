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
				{'max_tracked_objects': 10},
				{'max_tracked_objects_age': 1},
				{'generate_bounding_boxes': True},
				{'use_oriented_bounding_boxes': False},
				{'calculate_speed': False},
				#TODO: Add more parameters for the filtering process. Distances, eps, min_points, etc. This parameters can be set in the launch file and can be used to configure the filtering process instead of hardcoding the values. This is for temporal use only.
			]
		)

	])
