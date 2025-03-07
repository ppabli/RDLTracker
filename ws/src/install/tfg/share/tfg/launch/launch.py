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
				{'max_tracked_objects': 100},
				{'max_tracked_objects_age': 1},
				{'generate_bounding_boxes': True},
				{'use_oriented_bounding_boxes': True},
				{'calculate_speed': True},
				{'notify_on_speed': True},
				{'notify_on_width': True},
				{'notify_on_height': True},
				{'notify_on_length': True},
				{'notify_on_weight': True},
				{'gps_coordinates': [42.886232, -8.547737]},
			]
		)

	])
