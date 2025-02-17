from setuptools import find_packages, setup

package_name = 'tfg'

setup(
	name=package_name,
	version='0.0.1',
	packages=find_packages(exclude=['test']),
	data_files=[
		('share/' + package_name + '/launch', ['launch/launch.py']),
		('share/ament_index/resource_index/packages', ['resource/' + package_name]),
		('share/' + package_name, ['package.xml']),
	],
	install_requires=['setuptools', 'open3d', 'rclpy', 'sensor_msgs'],
	zip_safe=True,
	maintainer='Pablo Liste Cancela',
	maintainer_email='pablo.liste@rai.usc.es',
	description='TFG',
	license='MIT',
	tests_require=['pytest'],
	entry_points={
		'console_scripts': [
			'lidar_node = tfg.main:main',
		],
	},
)
