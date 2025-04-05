
import torch
import numpy as np
import mediapipe as mp


# Check for GPU or MPS availability
device = torch.device("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else
                     "cpu")
print(f"Using device: {device}")

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Constants
MAX_CAMERAS = 5  # Maximum number of cameras supported
MIN_CALIBRATION_SAMPLES = 10  # Minimum samples needed for calibration
MAX_CALIBRATION_SAMPLES = 500  # Maximum samples to collect for calibration
NUM_PARTICLES = 1000  # Number of particles for particle filter
PARTICLE_NOISE_POSITION = 0.01  # Noise level for particle position updates
PARTICLE_NOISE_ROTATION = 0.05  # Noise level for particle rotation updates
CONFIG_PATH = "./src/config/camera_config.yaml"  # Path to camera configuration file

# Hand keypoint indices from MediaPipe
WRIST_IDX       = 0
THUMB_CMC_IDX   = 1
THUMB_MCP_IDX   = 2
THUMB_IP_IDX    = 3
THUMB_TIP_IDX   = 4
INDEX_MCP_IDX   = 5
INDEX_PIP_IDX   = 6
INDEX_DIP_IDX   = 7
INDEX_TIP_IDX   = 8
MIDDLE_MCP_IDX  = 9
MIDDLE_PIP_IDX  = 10
MIDDLE_DIP_IDX  = 11
MIDDLE_TIP_IDX  = 12
RING_MCP_IDX    = 13
RING_PIP_IDX    = 14
RING_DIP_IDX    = 15
RING_TIP_IDX    = 16
PINKY_MCP_IDX   = 17
PINKY_PIP_IDX   = 18
PINKY_DIP_IDX   = 19
PINKY_TIP_IDX   = 20

# Default camera intrinsics (will be overridden by config)
DEFAULT_INTRINSICS = {
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 640.0,
    "cy": 360.0,
    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
}

# Camera type constants
CAMERA_TYPE_STANDARD = "standard"
CAMERA_TYPE_OAKD = "oakd"

# Default intrinsics for OAK-D cameras
OAKD_DEFAULT_INTRINSICS = {
    "fx": 860.0,  # Typical approximate value for OAK-D
    "fy": 860.0,
    "cx": 640.0,
    "cy": 360.0,
    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
}
