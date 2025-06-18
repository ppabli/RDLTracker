from tfg.constants import TORCH_DEVICE, TORCH_DTYPE
import torch

class TrackedObject:
	"""
	Class to represent a tracked object with improved velocity estimation using Open3D tensors.
	"""

	def __init__(self, centroid, points, features, timestamp, id=-1, label=-1, confidence=1.0, max_history_length=10):
		"""
		Constructor for the TrackedObject class.
		"""

		# Store the original Open3D PointCloud object directly
		self.points = points

		# Store centroids and features using PyTorch tensors
		self.centroid = centroid
		self.features = features

		self.timestamp = timestamp
		self.id = id
		self.max_history_length = max_history_length

		self.speed = 0.0
		self.direction = torch.zeros(3, dtype=TORCH_DTYPE, device=TORCH_DEVICE)

		# Pre-allocate memory for histories
		self.position_history = []
		self.velocity_history = []

		self.label = label
		self.label_confidence = confidence

		# Confidence metrics
		self.static_confidence = 0.0
		self.moving_confidence = 0.0
		self.velocity_stability = 0.0

		# Initialize Kalman filter for position and velocity tracking
		self._init_kalman_filter(self.centroid)
		self.kf_initialized = False

		self.position_history.append((self.centroid, timestamp))

	def _init_kalman_filter(self, initial_position):
		"""
		Initialize Kalman filter for position and velocity tracking using PyTorch.
		"""

		# State transition matrix (physics model)
		self.F = torch.eye(6, dtype=TORCH_DTYPE, device=TORCH_DEVICE).contiguous()
		self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = 0.1

		# Measurement function
		self.H = torch.zeros((3, 6), dtype=TORCH_DTYPE, device=TORCH_DEVICE).contiguous()
		self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

		# Initial state
		self.x = torch.zeros(6, dtype=TORCH_DTYPE, device=TORCH_DEVICE).contiguous()
		self.x[:3] = initial_position

		# Covariance matrices - tuned for better tracking performance
		self.P = torch.eye(6, dtype=TORCH_DTYPE, device=TORCH_DEVICE).contiguous() * 10.0	# Initial state uncertainty
		self.R = torch.eye(3, dtype=TORCH_DTYPE, device=TORCH_DEVICE).contiguous() * 0.1	# Measurement uncertainty

		# Process noise - how much we expect the model to deviate from reality
		self.Q = torch.eye(6, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
		self.Q[0:3, 0:3] *= 0.1		# position noise
		self.Q[3:6, 3:6] *= 0.05	# velocity noise

	def _should_update_label(self, new_confidence):
		"""
		Determine if the label should be updated based on confidence.
		"""

		return new_confidence > self.label_confidence

	def __str__(self):
		"""Return a string representation of the object."""
		return (f"Object | ID:{self.id} | Label: {self.label} (conf: {self.label_confidence:.2f}) | Static: {self.is_static()} | Speed: {self.speed:.2f} m/s")

	def predict(self):
		"""
		Kalman filter prediction step using PyTorch.
		"""

		# x = F·x
		self.x = torch.matmul(self.F, self.x)

		# P = F·P·F' + Q
		self.P = torch.matmul(torch.matmul(self.F, self.P), self.F.transpose(0, 1)) + self.Q

		return self.x

	def kalman_update(self, z):
		"""
		Kalman filter update step using PyTorch.
		"""

		# y = z - H·x
		y = z - torch.matmul(self.H, self.x)

		# S = H·P·H' + R
		S = torch.matmul(torch.matmul(self.H, self.P), self.H.transpose(0, 1)) + self.R

		# K = P·H'·S^-1
		K = torch.matmul(torch.matmul(self.P, self.H.transpose(0, 1)), torch.inverse(S))

		# x = x + K·y
		self.x = self.x + torch.matmul(K, y)

		# P = (I - K·H)·P
		I = torch.eye(self.x.shape[0], dtype=TORCH_DTYPE, device=TORCH_DEVICE)
		self.P = torch.matmul((I - torch.matmul(K, self.H)), self.P)

		return self.x

	def update(self, centroid, points, features, timestamp, label, confidence=1.0, speed_threshold=0.05):
		"""
		Update the object with new information.
		"""

		if self._should_update_label(confidence):

			self.label = label
			self.label_confidence = confidence

		# Store old values
		old_centroid = self.centroid
		old_timestamp = self.timestamp

		# Update basic attributes
		self.points = points	# Keep as Open3D object
		self.centroid = centroid
		self.features = features

		# Calculate time difference
		dt = timestamp - old_timestamp

		# Update Kalman filter
		if self.kf_initialized:

			# Update time step in state transition matrix
			self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt

			# Predict step
			self.predict()

			# Update step with new measurement
			self.kalman_update(centroid)

		else:

			# Initialize Kalman filter with first measurement
			self.x[:3] = centroid
			self.kf_initialized = True

		# Update timestamp after processing
		self.timestamp = timestamp

		# Add to position history
		self.position_history.append((centroid, timestamp))

		if len(self.position_history) > self.max_history_length:

			self.position_history.pop(0)

		# Calculate and update speed based on Kalman filter velocity estimate
		kalman_velocity = self.x[3:6]
		kalman_speed = torch.norm(kalman_velocity).item()

		# Only use Kalman speed if we have enough history
		if len(self.position_history) >= 3:

			self._update_velocity_statistics(old_centroid, centroid, dt)
			self._calculate_motion_confidence()

			# If we're confident this is a static object, force speed to zero
			if self.is_static():

				kalman_speed = 0.0
				self.x[3:6] = torch.zeros(3, dtype=TORCH_DTYPE, device=TORCH_DEVICE)

		# Set speed from Kalman filter
		self.speed = kalman_speed

		if self.speed < speed_threshold:

			self.speed = 0.0
			self.x[3:6] = torch.zeros(3, dtype=TORCH_DTYPE, device=TORCH_DEVICE)

		else:

			self.direction = kalman_velocity / kalman_speed

	def _update_velocity_statistics(self, old_pos, new_pos, dt):
		"""
		Update velocity statistics based on recent movement.
		"""

		# Calculate raw velocity
		displacement = torch.norm(new_pos - old_pos).item()
		raw_velocity = displacement / dt

		# Add to velocity history
		self.velocity_history.append(raw_velocity)

		if len(self.velocity_history) > self.max_history_length:

			self.velocity_history.pop(0)

		# Calculate velocity stability (coefficient of variation)
		if len(self.velocity_history) >= 3:

			# Only use recent history for calculations
			recent_velocities = torch.tensor(self.velocity_history[-5:], device=TORCH_DEVICE)
			mean_velocity = torch.mean(recent_velocities).item()

			if mean_velocity > 0.1:

				self.velocity_stability = 1.0 - min(1.0, torch.std(recent_velocities).item() / max(0.1, mean_velocity))

			else:

				self.velocity_stability = 1.0

	def _calculate_motion_confidence(self):
		"""Calculate confidence levels for static vs. moving classification."""

		if len(self.position_history) < 3:

			return

		# Extract positions from history
		positions = torch.stack([pos for pos, _ in self.position_history])

		# Calculate positional variance
		pos_variance = torch.var(positions, dim=0)
		max_pos_variance = torch.max(pos_variance).item()

		# Calculate velocity metrics
		velocity_tensor = torch.tensor(self.velocity_history, device=TORCH_DEVICE)
		velocity_mean = torch.mean(velocity_tensor).item()

		# Static confidence increases with low variance and low mean velocity
		static_score = 1.0 - min(1.0, max_pos_variance / 0.05) * min(1.0, velocity_mean / 0.5)

		# Moving confidence increases with consistent non-zero velocity and directional consistency
		moving_score = min(1.0, velocity_mean / 0.5) * self.velocity_stability

		# Update confidence levels with temporal smoothing
		self.static_confidence = 0.6 * self.static_confidence + 0.4 * static_score
		self.moving_confidence = 0.6 * self.moving_confidence + 0.4 * moving_score

		# Normalize confidence values
		total = self.static_confidence + self.moving_confidence

		if total > 0:

			norm_factor = 1.0 / total
			self.static_confidence *= norm_factor
			self.moving_confidence *= norm_factor

	def is_static(self, confidence_threshold=0.5):
		"""
		Determine if the object is static based on movement history.
		"""

		# Quick check for objects with very limited history
		if len(self.position_history) < 3:

			return self.speed < 0.5

		# For objects with enough history, use more complex criteria
		if self.static_confidence > confidence_threshold:

			return True

		# Check if static confidence exceeds moving confidence with low recent velocity
		if self.static_confidence > self.moving_confidence:

			recent_velocities = torch.tensor(self.velocity_history[-3:], device=TORCH_DEVICE)
			return torch.mean(recent_velocities).item() < 0.2

		return False

	def get_filtered_position(self):
		"""Get the Kalman-filtered position."""

		return self.x[:3]

	def compute_bounding_box(self):
		"""
		Compute the bounding box of the object.
		"""

		return self.points.get_axis_aligned_bounding_box()

	def predict_position(self, time_delta):
		"""
		Predict the object's position after a given time interval.
		"""

		if not self.kf_initialized:

			return self.centroid

		# Use tensor operations to predict future position
		x_pred = torch.zeros(6, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
		x_pred[:3] = self.x[:3] + time_delta * self.x[3:6]
		x_pred[3:6] = self.x[3:6]

		return x_pred[:3]

	def get_speed(self):
		"""Get the current speed in m/s."""

		return self.speed

	def get_direction(self):
		"""Get the current movement direction (normalized vector)."""

		return self.direction

	def get_label(self):
		"""Get the current estimated label."""

		return self.label

	def get_label_confidence(self):
		"""Get the confidence of the current label."""

		return self.label_confidence