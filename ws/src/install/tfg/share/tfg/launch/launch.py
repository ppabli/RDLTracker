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
				{'crop_x': 0.0},
				{'crop_y': 6.0},
				{'crop_z': 0.0},
				{'max_tracked_objects': 100},
				{'max_tracked_objects_age': 0.5},
				{'generate_bounding_boxes': True},
				{'calculate_speed': True},
				{'notify_on_speed': True},
				{'notify_on_width': False},
				{'notify_on_height': False},
				{'notify_on_length': False},
				{'notify_on_weight': False},
				{'gps_coordinates': [42.886232, -8.547737]},
				{'use_classification_model': True},
				{'classification_model_weights_path': '/home/pablo/Desktop/pointnet/output/pointnet_best.pth'},
			]
		)

	])
