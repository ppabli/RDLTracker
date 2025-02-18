import numpy as np
import open3d as o3d

class TrackedObject:
	"""
	Class to represent a tracked object.
	"""

	def __init__(self, cluster_info, timestamp):
		"""
		Constructor for the TrackedObject class.
		"""

		self.centroid = cluster_info['centroid']
		self.points = cluster_info['points']

		self.timestamps = [timestamp]
		self.speeds = []

	def __str__(self):
		"""
		Return a string representation of the object.
		"""

		return f"Object | Centroid: {self.centroid} | Speed: {self.speeds[-1] if self.speeds else 0}"

	def update(self, cluster_info, timestamp, calculate_speed=False):
		"""
		Update the object with new information.
		"""

		if len(self.timestamps) > 1 and calculate_speed:

			delta_t = self.timestamps[-1] - self.timestamps[-2]
			delta_x = np.linalg.norm(cluster_info['centroid'] - self.centroid)

			temp_speed = delta_x / delta_t

			if temp_speed > 0.30:

				self.speeds.append(temp_speed)

		self.centroid = cluster_info['centroid']
		self.points = cluster_info['points']

		self.timestamps.append(timestamp)

		# Limit the number of timestamps to 10 and the number of speeds to 10
		self.timestamps = self.timestamps[-10:]
		self.speeds = self.speeds[-10:]

	def compute_bouding_box(self, use_oriented=False):
		"""
		Compute the bounding box of the object
		"""

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(self.points)

		bbox = pcd.get_oriented_bounding_box() if use_oriented else pcd.get_axis_aligned_bounding_box()

		return bbox