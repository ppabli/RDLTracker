import open3d as o3d
import torch
import numpy as np

O3D_DEVICE = o3d.core.Device("CPU:0")
TORCH_DEVICE = torch.device("cpu")

O3D_DTYPE = o3d.core.float32
TORCH_DTYPE = torch.float32

NP_DTYPE = np.float32

LABEL_MAP = {
	"Car": 0,
	"Pedestrian": 1,
	"Cyclist": 2,
}

INVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

RED = '\033[91m'
GREEN = '\033[92m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'