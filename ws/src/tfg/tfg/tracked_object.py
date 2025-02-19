import numpy as np
import open3d as o3d

class TrackedObject:
	"""
	Class to represent a tracked object.
	"""

	def __init__(self, centroid, points, timestamp):
		"""
		Constructor for the TrackedObject class.
		"""

		self.centroid = centroid
		self.points = points

		self.timestamp = timestamp
		self.speed = 0

	def __str__(self):
		"""
		Return a string representation of the object.
		"""

		return f"Object | Centroid: {self.centroid} | Speed: {self.speed}"

	def update(self, centroid, points, timestamp, calculate_speed=False, delta_x=0):
		"""
		Update the object with new information.
		"""

		self.centroid = centroid
		self.points = points
		self.timestamp = timestamp

	def update_speed(self, delta_x):
		"""
		Update the speed of the object.
		"""

		self.speed = delta_x / 0.1

	def compute_bouding_box(self, use_oriented=False):
		"""
		Compute the bounding box of the object
		"""

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(self.points)

		bbox = pcd.get_oriented_bounding_box() if use_oriented else pcd.get_axis_aligned_bounding_box()

		return bbox