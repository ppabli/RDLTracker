import numpy as np
import open3d as o3d

class TrackedObject:
	"""
	Class to represent a tracked object.
	"""

	def __init__(self, centroid, points, features, id=-1, timestamp=None, label='unknown'):
		"""
		Constructor for the TrackedObject class.
		"""

		self.centroid = centroid
		self.points = points

		self.timestamp = timestamp
		self.features = features # Tensor
		self.label = label

		self.speed = 0 # m/s
		self.direction = None # Normalized tensor

		self.id = id

	def __str__(self):
		"""
		Return a string representation of the object.
		"""

		return f"Object | ID:{self.id} | Label: {self.label} | Speed: {self.speed:.2f} m/s"

	def update(self, centroid, points, features, timestamp):
		"""
		Update the object with new information.
		"""

		self.centroid = centroid
		self.points = points
		self.timestamp = timestamp
		self.features = features

	def update_speed(self, new_speed, new_direction):
		"""
		Update the speed of the object based on movement over time.
		"""

		self.speed = new_speed
		self.direction = new_direction

	def compute_bouding_box(self, use_oriented=False):
		"""
		Compute the bounding box of the object
		"""

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(self.points)

		bbox = pcd.get_oriented_bounding_box() if use_oriented else pcd.get_axis_aligned_bounding_box()

		return bbox