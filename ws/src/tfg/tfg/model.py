import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from tfg.constants import LABEL_MAP


def read_calib_file(calib_path):
	"""Read KITTI calibration file."""

	calib = {}

	with open(calib_path, 'r') as f:

		for line in f.readlines():

			line.rstrip()

			if not line.startswith('#') and len(line) > 1:

				key, value = line.split(':', 1)
				calib[key] = np.array([float(x) for x in value.split()])

	calib['P0'] = calib['P0'].reshape(3, 4)
	calib['P1'] = calib['P1'].reshape(3, 4)
	calib['P2'] = calib['P2'].reshape(3, 4)
	calib['P3'] = calib['P3'].reshape(3, 4)
	calib['R0_rect'] = calib['R0_rect'].reshape(3, 3)
	calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)

	return calib

def get_3d_box_points(center, dimensions, rotation_y):
	"""Get 3D box corners in camera coordinates."""

	l, h, w = dimensions

	x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
	y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
	z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

	R = np.array([
		[np.cos(rotation_y), 0, np.sin(rotation_y)],
		[0, 1, 0],
		[-np.sin(rotation_y), 0, np.cos(rotation_y)]
	])

	corners = np.vstack([x_corners, y_corners, z_corners])
	corners = np.dot(R, corners)
	corners = corners + np.array(center).reshape(3, 1)

	return corners.T

def transform_points_to_velo(points, calib):
	"""Transform points from camera to velodyne coordinates."""

	R_rect = np.eye(4)
	R_rect[:3, :3] = calib['R0_rect']

	Tr_velo_to_cam = np.eye(4)
	Tr_velo_to_cam[:3, :4] = calib['Tr_velo_to_cam']

	T_cam_to_velo = np.linalg.inv(Tr_velo_to_cam @ R_rect)

	points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
	points_velo = (T_cam_to_velo @ points_hom.T).T

	return points_velo[:, :3]

def extract_objects_from_scene(point_cloud, labels, calib, margin=0.3):
	"""Extract individual objects from the scene using 3D bounding boxes."""
	objects = []
	object_labels = []

	for label in labels:

		label_parts = label.strip().split()

		if label_parts[0] in LABEL_MAP:

			obj_type = label_parts[0]
			dimensions = [float(label_parts[8]), float(label_parts[9]), float(label_parts[10])]
			location = [float(label_parts[11]), float(label_parts[12]), float(label_parts[13])]
			rotation_y = float(label_parts[14])

			box_corners = get_3d_box_points(location, dimensions, rotation_y)

			box_corners_velo = transform_points_to_velo(box_corners, calib)

			min_bound = np.min(box_corners_velo, axis=0) - margin
			max_bound = np.max(box_corners_velo, axis=0) + margin

			mask = np.all((point_cloud[:, :3] >= min_bound) & (point_cloud[:, :3] <= max_bound), axis=1)
			object_points = point_cloud[mask]

			if len(object_points) >= 50:

				objects.append(object_points[:, :3])
				object_labels.append(LABEL_MAP[obj_type])

	return objects, object_labels

def normalize_point_cloud_tensor(points, num_points=128):

	centroid = torch.mean(points, dim=0, keepdim=True)
	points = points - centroid

	m = torch.max(torch.norm(points, dim=1))

	if m > 0:

		points = points / m

	n = points.shape[0]
	if n > num_points:

		idx = torch.randperm(n)[:num_points]
		points = points[idx]

	elif n < num_points:

		idx = torch.randint(n, (num_points - n,))
		points = torch.cat([points, points[idx]], dim=0)

	return points

class KittiDataset(Dataset):

	def __init__(self, dataset_path, num_points=1024):

		self.dataset_path = dataset_path
		self.num_points = num_points
		self.point_clouds = []
		self.labels = []
		self.load_kitti_data()

	def load_kitti_data(self):

		velodyne_path = os.path.join(self.dataset_path, 'velodyne')
		label_path = os.path.join(self.dataset_path, 'label_2')
		calib_path = os.path.join(self.dataset_path, 'calib')

		for idx in range(len(os.listdir(velodyne_path))):

			bin_file = os.path.join(velodyne_path, f'{idx:06d}.bin')
			point_cloud = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)

			calib_file = os.path.join(calib_path, f'{idx:06d}.txt')
			calib = read_calib_file(calib_file)

			label_file = os.path.join(label_path, f'{idx:06d}.txt')

			with open(label_file, 'r') as f:

				labels = f.readlines()

			objects, object_labels = extract_objects_from_scene(point_cloud, labels, calib)

			for obj_points, obj_label in zip(objects, object_labels):

				normalized_points = normalize_point_cloud_tensor(obj_points, self.num_points)
				self.point_clouds.append(normalized_points)
				self.labels.append(obj_label)

	def __len__(self):

		return len(self.labels)

	def __getitem__(self, idx):

		return torch.tensor(self.point_clouds[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class PointNet(nn.Module):

	def __init__(self, num_classes):

		super(PointNet, self).__init__()

		self.input_transform = Transform3d()

		self.feature_transform = Transform3d(k=64)

		self.conv1 = nn.Conv1d(3, 64, 1)
		self.conv2 = nn.Conv1d(64, 128, 1)
		self.conv3 = nn.Conv1d(128, 256, 1)
		self.conv4 = nn.Conv1d(256, 512, 1)

		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(256)
		self.bn4 = nn.BatchNorm1d(512)

		self.fc1 = nn.Linear(512, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, num_classes)

		self.bn5 = nn.BatchNorm1d(512)
		self.bn6 = nn.BatchNorm1d(256)

		self.dropout = nn.Dropout(p=0.3)

	def forward(self, x):

		batch_size = x.size(0)
		num_points = x.size(1)

		x = x.transpose(2, 1)
		trans = self.input_transform(x)
		x = x.transpose(2, 1)
		x = torch.bmm(x, trans)
		x = x.transpose(2, 1)

		x = F.relu(self.bn1(self.conv1(x)))

		trans_feat = self.feature_transform(x)
		x = x.transpose(2, 1)
		x = torch.bmm(x, trans_feat)
		x = x.transpose(2, 1)

		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))

		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, 512)

		x = F.relu(self.bn5(self.fc1(x)))
		x = self.dropout(x)
		x = F.relu(self.bn6(self.fc2(x)))
		x = self.dropout(x)
		x = self.fc3(x)

		return F.log_softmax(x, dim=1)

class Transform3d(nn.Module):

	def __init__(self, k=3):

		super(Transform3d, self).__init__()

		self.k = k

		self.conv1 = nn.Conv1d(k, 64, 1)
		self.conv2 = nn.Conv1d(64, 128, 1)
		self.conv3 = nn.Conv1d(128, 1024, 1)

		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(1024)

		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, k*k)

		self.bn4 = nn.BatchNorm1d(512)
		self.bn5 = nn.BatchNorm1d(256)

	def forward(self, x):

		batch_size = x.size(0)

		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))

		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, 1024)

		x = F.relu(self.bn4(self.fc1(x)))
		x = F.relu(self.bn5(self.fc2(x)))
		x = self.fc3(x)

		iden = torch.eye(self.k).view(1, self.k*self.k).repeat(batch_size, 1)

		if x.is_cuda:

			iden = iden.cuda()

		x = x + iden
		x = x.view(-1, self.k, self.k)

		return x

class PointNetTrainer:

	def __init__(self, num_classes, dataset_path=None, batch_size=32, epochs=50, lr=0.001):

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.num_classes = num_classes
		self.batch_size = batch_size
		self.epochs = epochs
		self.lr = lr

		if dataset_path is not None:

			self.dataset = KittiDataset(dataset_path)
			self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

		self.model = PointNet(num_classes).to(self.device)
		self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
		self.criterion = nn.CrossEntropyLoss()

	def train(self):
		"""Training loop with progress tracking."""

		self.model.train()
		best_loss = float('inf')
		best_accuracy = 0.0

		for epoch in range(self.epochs):

			epoch_loss = 0.0
			correct = 0
			total = 0

			batch_count = len(self.dataloader)

			for batch_idx, (points, labels) in enumerate(self.dataloader, 1):

				points = points.to(self.device)
				labels = labels.to(self.device)

				self.optimizer.zero_grad()
				outputs = self.model(points)
				loss = self.criterion(outputs, labels)

				loss.backward()
				self.optimizer.step()

				epoch_loss += loss.item()
				_, predicted = outputs.max(1)
				total += labels.size(0)
				correct += predicted.eq(labels).sum().item()

				if batch_idx % 10 == 0 or batch_idx == batch_count:

					progress = (batch_idx / batch_count) * 100

			avg_loss = epoch_loss / batch_count
			accuracy = 100. * correct / total

			if accuracy > best_accuracy:

				best_accuracy = accuracy
				self.save_model('best_model.pth')

			if avg_loss < best_loss:

				best_loss = avg_loss

	def save_model(self, path='pointnet.pth'):
		"""Save model checkpoint."""

		torch.save({
			'epoch': self.epochs,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'best_accuracy': best_accuracy if 'best_accuracy' in locals() else None, # type: ignore
		}, path)

	def load_model(self, path='pointnet.pth'):
		"""Load model checkpoint."""

		checkpoint = torch.load(path, map_location=self.device)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def segment_scene(points, voxel_size=0.5, min_points=50):
	"""Segment scene into potential objects using DBSCAN clustering."""

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)

	downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)

	labels = np.array(downpcd.cluster_dbscan(eps=0.7, min_points=10))

	clusters = []

	for label in range(labels.max() + 1):

		cluster_points = np.asarray(downpcd.points)[labels == label]

		if len(cluster_points) >= min_points:

			clusters.append(cluster_points)

	return clusters

def main():

	trainer = PointNetTrainer(
		num_classes=len(LABEL_MAP),
		dataset_path='/media/pablo/Disco programas/datasets/Kitti/training',
		batch_size=32,
		epochs=50
	)

	trainer.train()
	trainer.save_model('multi_object_model.pth')

if __name__ == "__main__":

	main()