import argparse
import os
import time
import threading
import yaml
import queue
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you prefer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

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
MAX_CAMERAS = 3  # Maximum number of cameras supported
MIN_CALIBRATION_SAMPLES = 100  # Minimum samples needed for calibration
MAX_CALIBRATION_SAMPLES = 500  # Maximum samples to collect for calibration
NUM_PARTICLES = 1000  # Number of particles for particle filter
PARTICLE_NOISE_POSITION = 0.01  # Noise level for particle position updates
PARTICLE_NOISE_ROTATION = 0.05  # Noise level for particle rotation updates
CONFIG_PATH = "camera_config.yaml"  # Path to camera configuration file

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


##################################################################
#  Camera Configuration
##################################################################

class CameraConfig:
    """Camera configuration class"""
    
    @staticmethod
    def load_config(config_path=CONFIG_PATH):
        """Load camera configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        else:
            print(f"Config file {config_path} not found, using default settings")
            # Create a default config
            config = {
                "cameras": []
            }
            for i in range(MAX_CAMERAS):
                config["cameras"].append({
                    "id": i,
                    "type": CAMERA_TYPE_STANDARD,
                    "primary": i == 0,
                    "intrinsics": DEFAULT_INTRINSICS.copy()
                })
            return config
            
    @staticmethod
    def save_config(config, config_path=CONFIG_PATH):
        """Save camera configuration to YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
    @staticmethod
    def get_camera_intrinsics(config, camera_idx):
        """Get intrinsics for a specific camera"""
        for camera in config["cameras"]:
            if camera["id"] == camera_idx:
                return camera["intrinsics"]
        return DEFAULT_INTRINSICS.copy()
        
    @staticmethod
    def get_primary_camera_idx(config):
        """Get the index of the primary camera"""
        for camera in config["cameras"]:
            if camera.get("primary", False):
                return camera["id"]
        return 0  # Default to first camera if none marked as primary
    
    @staticmethod
    def get_camera_type(config, camera_idx):
        """Get the type of a specific camera"""
        for camera in config["cameras"]:
            if camera["id"] == camera_idx:
                return camera.get("type", CAMERA_TYPE_STANDARD)
        return CAMERA_TYPE_STANDARD


##################################################################
#  Pytorch Hand Model
##################################################################

class PytorchHandModel(nn.Module):
    """
    A simple differentiable hand model for calibration and
    pose-estimation tasks. It uses a canonical hand template
    plus a rigid transform (rotation + translation).
    """
    def __init__(self):
        super(PytorchHandModel, self).__init__()
        # 21 landmarks in 3D, no gradient required by default
        self.canonical_hand = nn.Parameter(
            torch.zeros(21, 3),
            requires_grad=False
        )
        self.initialize_canonical_hand()
        
    def initialize_canonical_hand(self):
        """Initialize the canonical hand model to a simple template."""
        hand = torch.zeros(21, 3)
        
        # Wrist at origin
        hand[WRIST_IDX] = torch.tensor([0.0, 0.0, 0.0])
        
        # Thumb
        hand[THUMB_CMC_IDX] = torch.tensor([-0.03, 0.01, 0.0])
        hand[THUMB_MCP_IDX] = torch.tensor([-0.04, 0.03, 0.01])
        hand[THUMB_IP_IDX]  = torch.tensor([-0.03, 0.05, 0.015])
        hand[THUMB_TIP_IDX] = torch.tensor([-0.02, 0.06, 0.02])
        
        # Index finger
        hand[INDEX_MCP_IDX] = torch.tensor([-0.01, 0.03, 0.0])
        hand[INDEX_PIP_IDX] = torch.tensor([-0.01, 0.05, 0.0])
        hand[INDEX_DIP_IDX] = torch.tensor([-0.01, 0.06, 0.0])
        hand[INDEX_TIP_IDX] = torch.tensor([-0.01, 0.07, 0.0])
        
        # Middle finger
        hand[MIDDLE_MCP_IDX] = torch.tensor([0.0, 0.03, 0.0])
        hand[MIDDLE_PIP_IDX] = torch.tensor([0.0, 0.05, 0.0])
        hand[MIDDLE_DIP_IDX] = torch.tensor([0.0, 0.06, 0.0])
        hand[MIDDLE_TIP_IDX] = torch.tensor([0.0, 0.07, 0.0])
        
        # Ring finger
        hand[RING_MCP_IDX] = torch.tensor([0.01, 0.03, 0.0])
        hand[RING_PIP_IDX] = torch.tensor([0.01, 0.05, 0.0])
        hand[RING_DIP_IDX] = torch.tensor([0.01, 0.06, 0.0])
        hand[RING_TIP_IDX] = torch.tensor([0.01, 0.07, 0.0])
        
        # Pinky finger
        hand[PINKY_MCP_IDX] = torch.tensor([0.02, 0.02, 0.0])
        hand[PINKY_PIP_IDX] = torch.tensor([0.02, 0.04, 0.0])
        hand[PINKY_DIP_IDX] = torch.tensor([0.02, 0.05, 0.0])
        hand[PINKY_TIP_IDX] = torch.tensor([0.02, 0.06, 0.0])
        
        self.canonical_hand.data = hand

    def forward(self, pose_params, scale=1.0):
        """
        Transform the canonical hand model using pose parameters.
        
        Args:
            pose_params: tensor of shape [batch_size, 6] containing
                         [tx, ty, tz, rx, ry, rz]
            scale: a global scale factor for the hand (default=1.0)
        
        Returns:
            Transformed hand landmarks [batch_size, 21, 3].
        """
        batch_size = pose_params.shape[0]
        
        # Extract translation and rotation parameters
        translation = pose_params[:, :3].unsqueeze(1)  # [B, 1, 3]
        euler_angles = pose_params[:, 3:6]             # [B, 3]
        
        # Convert Euler angles to rotation matrices
        rot_matrices = self.euler_to_rotmat(euler_angles)  # [B, 3, 3]
        
        # Scale the canonical hand
        scaled_hand = self.canonical_hand * scale  # [21, 3]
        
        # Expand canonical hand to batch dimension
        hand_batch = scaled_hand.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 21, 3]
        
        # Apply rotation
        rotated_hand = torch.bmm(hand_batch, rot_matrices.transpose(1, 2))  # [B, 21, 3]
        
        # Apply translation
        transformed_hand = rotated_hand + translation  # [B, 21, 3]
        
        return transformed_hand
    
    def euler_to_rotmat(self, euler_angles):
        """
        Convert Euler angles to rotation matrices (XYZ convention).
        
        Args:
            euler_angles: [batch_size, 3] => [rx, ry, rz]
        
        Returns:
            [batch_size, 3, 3] rotation matrices
        """
        batch_size = euler_angles.shape[0]
        rx, ry, rz = torch.unbind(euler_angles, dim=1)

        zeros = torch.zeros_like(rx)
        ones = torch.ones_like(rx)
        
        # X-axis rotation
        rot_x = torch.stack([
            torch.stack([ones, zeros, zeros], dim=1),
            torch.stack([zeros, torch.cos(rx), -torch.sin(rx)], dim=1),
            torch.stack([zeros, torch.sin(rx),  torch.cos(rx)], dim=1)
        ], dim=1)  # [B, 3, 3]
        
        # Y-axis rotation
        rot_y = torch.stack([
            torch.stack([ torch.cos(ry), zeros, torch.sin(ry)], dim=1),
            torch.stack([ zeros,         ones,  zeros],        dim=1),
            torch.stack([-torch.sin(ry), zeros, torch.cos(ry)], dim=1)
        ], dim=1)  # [B, 3, 3]
        
        # Z-axis rotation
        rot_z = torch.stack([
            torch.stack([ torch.cos(rz), -torch.sin(rz), zeros], dim=1),
            torch.stack([ torch.sin(rz),  torch.cos(rz), zeros], dim=1),
            torch.stack([ zeros,          zeros,         ones],  dim=1)
        ], dim=1)  # [B, 3, 3]
        
        # Combine rotations: Rz * Ry * Rx
        rot_mat = torch.bmm(rot_z, torch.bmm(rot_y, rot_x))
        return rot_mat


##################################################################
#  Camera Trackers
##################################################################

class CameraTracker:
    """Abstract base class for camera tracking."""
    
    def __init__(self, camera_id, intrinsics=None, show=False):
        self.camera_id = camera_id
        self.intrinsics = intrinsics or DEFAULT_INTRINSICS.copy()
        self.show = show
        self.running = False
        self.frame = None             # Current frame
        self.processed_frame = None   # Frame with visualizations
        self.hand_landmarks = None    # Latest hand landmarks
        self.detection_confidence = 0.0
        self.lock = threading.Lock()  # Thread safety for landmarks/confidence
        
        # MediaPipe hand detector
        self.hands = mp_hands.Hands(
            model_complexity=1, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=1
        )
        
        # Camera info placeholder
        self.camera_info = {
            "id": self.camera_id,
            "width": 0,
            "height": 0,
            "fps": 0
        }
    
    def start(self):
        """Start the camera tracking thread."""
        raise NotImplementedError
    
    def stop(self):
        """Stop the camera tracking thread."""
        raise NotImplementedError
    
    def get_hand_landmarks(self):
        """Thread-safe getter for the latest hand landmarks."""
        with self.lock:
            if self.hand_landmarks is not None:
                return self.hand_landmarks.copy(), self.detection_confidence
            return None, 0.0
    
    def process_frame(self, image):
        """Process a frame for hand detection and store the results."""
        if image is None:
            return image
        
        # Convert to RGB for MediaPipe
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        # Restore BGR for display
        image.flags.writeable = True
        
        # Store the raw frame
        self.frame = image.copy()
        self.add_camera_info_overlay(image)
        
        if results.multi_hand_landmarks:
            # Use the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_handedness = results.multi_handedness[0]
            confidence = hand_handedness.classification[0].score
            
            # Extract 3D keypoints
            landmark_array = np.array(
                [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark],
                dtype=np.float32
            )
            
            with self.lock:
                self.hand_landmarks = landmark_array
                self.detection_confidence = confidence
            
            # Draw the landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
            
            # Add confidence text
            cv2.putText(
                image, 
                f"Confidence: {confidence:.2f}",
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 255, 0), 
                2
            )
        else:
            with self.lock:
                self.hand_landmarks = None
                self.detection_confidence = 0.0
            
            cv2.putText(
                image, 
                "No Hand Detected",
                (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 255), 
                2
            )
        
        if self.show:
            # Flip for a mirrored display
            self.processed_frame = cv2.flip(image, 1)
        else:
            self.processed_frame = None
        
        return self.processed_frame
    
    def add_camera_info_overlay(self, image):
        """Add camera ID/resolution overlay to the image."""
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (400, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        camera_text = f"Camera {self.camera_id}"
        if "name" in self.camera_info:
            camera_text += f" - {self.camera_info['name']}"
        if "type" in self.camera_info:
            camera_text += f" ({self.camera_info['type']})"
        
        resolution_text = f"{self.camera_info['width']}x{self.camera_info['height']} @ {self.camera_info['fps']:.1f}fps"
        
        cv2.putText(
            image,
            camera_text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 220, 255),
            2
        )
        
        cv2.putText(
            image,
            resolution_text,
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1
        )

class SingleCameraTracker(CameraTracker):
    """Handles tracking for a single standard webcam."""
    
    def __init__(self, camera_id, intrinsics=None, show=False):
        super().__init__(camera_id, intrinsics, show)
        
        self.cap = cv2.VideoCapture(camera_id)
        self.camera_info = self.get_camera_info()
        
        # Increase buffer size for smoother video
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        # Attempt to set a default resolution if needed
        if self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) < 1280:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.thread = threading.Thread(target=self.run_tracker)
    
    def get_camera_info(self):
        """Get camera information for display (if available)."""
        info = {
            "id": self.camera_id,
            "type": CAMERA_TYPE_STANDARD,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
        }
        try:
            info["backend"] = self.cap.getBackendName()
            if hasattr(cv2, 'CAP_PROP_SERIAL'):
                info["serial"] = self.cap.get(cv2.CAP_PROP_SERIAL)
            if hasattr(cv2, 'CAP_PROP_DEVICE_NAME'):
                info["name"] = self.cap.get(cv2.CAP_PROP_DEVICE_NAME)
        except Exception as e:
            print(f"Could not get detailed camera info: {e}")
        return info
    
    def start(self):
        self.running = True
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()
    
    def run_tracker(self):
        while self.running and self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print(f"Failed to read from camera {self.camera_id}")
                time.sleep(0.1)
                continue
            self.process_frame(image)


##################################################################
#  OAK-D Camera Support
##################################################################

try:
    import depthai as dai
    class OAKDCameraManager:
        """Helper class to manage OAK-D cameras."""

        def __init__(self):
            self.active_devices = []
            self.device_info_list = []
        
        def get_available_devices(self):
            """Get list of available OAK-D devices."""
            device_infos = dai.Device.getAllAvailableDevices()
            self.device_info_list = device_infos
            
            available_devices = []
            for i, device_info in enumerate(device_infos):
                available_devices.append({
                    "id": f"oakd_{i}",
                    "mxid": device_info.getMxId(),
                    "name": device_info.name,
                    "type": CAMERA_TYPE_OAKD
                })
            print(f"Found {len(available_devices)} OAK-D devices")
            return available_devices
        
        def create_pipeline(self, resolution=(1280, 720), fps=30):
            """Create a pipeline for RGB camera stream from OAK-D."""
            pipeline = dai.Pipeline()
            # Define source and output
            cam_rgb = pipeline.create(dai.node.ColorCamera)
            xout_rgb = pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            
            # Properties
            cam_rgb.setPreviewSize(resolution[0], resolution[1])
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam_rgb.setFps(fps)
            
            # Link
            cam_rgb.preview.link(xout_rgb.input)
            return pipeline
        
        def get_camera_intrinsics(self, device, resolution=(1280, 720)):
            """Get camera intrinsics from OAK-D device calibration."""
            try:
                calibration = device.readCalibration()
                intrinsic_matrix = calibration.getCameraIntrinsics(
                    dai.CameraBoardSocket.RGB,
                    dai.Size2f(*resolution)
                )
                intrinsics = {
                    "fx": intrinsic_matrix[0][0],
                    "fy": intrinsic_matrix[1][1],
                    "cx": intrinsic_matrix[0][2],
                    "cy": intrinsic_matrix[1][2],
                    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
                }
                return intrinsics
            except Exception as e:
                print(f"Error getting OAK-D camera intrinsics: {e}")
                return OAKD_DEFAULT_INTRINSICS.copy()
    
    class OAKDCameraTracker(CameraTracker):
        """Handles tracking for an OAK-D camera."""
        
        def __init__(self, device_info, index, intrinsics=None,
                     show=False, resolution=(1280, 720), fps=30):
            camera_id = f"oakd_{index}"
            super().__init__(camera_id, intrinsics, show)
            self.device_info = device_info
            self.resolution = resolution
            self.fps = fps
            self.oakd_manager = OAKDCameraManager()
            self.device = None
            self.queue = None
            
            self.camera_info = {
                "id": self.camera_id,
                "type": CAMERA_TYPE_OAKD,
                "name": device_info.name if device_info else "OAK-D",
                "mxid": device_info.getMxId() if device_info else "Unknown",
                "width": resolution[0],
                "height": resolution[1],
                "fps": fps
            }
            self.thread = threading.Thread(target=self.run_tracker)
        
        def start(self):
            # Create pipeline
            pipeline = self.oakd_manager.create_pipeline(self.resolution, self.fps)
            try:
                self.device = dai.Device(pipeline, self.device_info)
                self.queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                
                # Try to get camera intrinsics
                device_intrinsics = self.oakd_manager.get_camera_intrinsics(
                    self.device, self.resolution
                )
                if device_intrinsics:
                    self.intrinsics = device_intrinsics
                    print(f"Using OAK-D intrinsics for {self.camera_id}")
                
                self.running = True
                self.thread.start()
            except Exception as e:
                print(f"Failed to start OAK-D camera {self.camera_id}: {e}")
        
        def stop(self):
            self.running = False
            if self.thread.is_alive():
                self.thread.join()
            if self.device:
                self.device.close()
                self.device = None
        
        def run_tracker(self):
            while self.running:
                if self.queue is None:
                    time.sleep(0.1)
                    continue
                try:
                    in_rgb = self.queue.get()
                    frame = in_rgb.getCvFrame()
                    self.process_frame(frame)
                except Exception as e:
                    print(f"Error processing OAK-D frame: {e}")
                    time.sleep(0.1)

except ImportError:
    # If depthai is not installed, define dummies or skip OAK-D functionality
    print("DepthAI not installed, OAK-D support is disabled.")

    class OAKDCameraTracker(CameraTracker):
        """Dummy OAK-D tracker if depthai is not available."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError("OAK-D camera not supported because 'depthai' is not installed.")


##################################################################
#  PyTorch-Based Camera Calibrator
##################################################################

class PyTorchCameraCalibrator:
    """Camera calibration using PyTorch for differentiable optimization."""
    
    def __init__(self, num_cameras, primary_camera_idx=0):
        self.num_cameras = num_cameras
        self.primary_camera_idx = primary_camera_idx
        self.calibration_samples = [[] for _ in range(num_cameras)]
        
        # Initialize camera extrinsics as PyTorch parameters
        # [tx, ty, tz, rx, ry, rz] for each camera
        self.camera_extrinsics = nn.ParameterList([
            nn.Parameter(torch.zeros(6, device=device))  # [tx, ty, tz, rx, ry, rz]
            for _ in range(num_cameras)
        ])
        
        # Primary camera is fixed at identity (no optimization)
        self.camera_extrinsics[primary_camera_idx].requires_grad = False
        
        # Hand model for optimization
        self.hand_model = PytorchHandModel().to(device)
        
        self.is_calibrated = False
        self.calibration_progress = 0.0
    
    def add_sample(self, camera_idx, landmarks, confidence):
        """Add a calibration sample from a camera."""
        if confidence > 0.7 and not self.is_calibrated:  # only use high confidence
            self.calibration_samples[camera_idx].append(landmarks)
            return True
        return False
    
    def get_transform_matrix(self, camera_idx):
        """Get the 4x4 transformation matrix for a camera."""
        if not self.is_calibrated:
            # Primary camera is identity, others are None until calibration
            if camera_idx == self.primary_camera_idx:
                return np.eye(4)
            else:
                return None
        
        # Convert PyTorch parameters to NumPy transformation matrix
        params = self.camera_extrinsics[camera_idx].detach().cpu().numpy()
        tx, ty, tz, rx, ry, rz = params
        rotation = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()
        
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [tx, ty, tz]
        return transform
    
    def check_calibration_progress(self):
        """Check if we have enough samples to perform calibration."""
        # Find the minimum number of samples across non-primary cameras
        min_samples = float('inf')
        for i in range(self.num_cameras):
            if i != self.primary_camera_idx:
                num_samples = len(self.calibration_samples[i])
                if num_samples < min_samples:
                    min_samples = num_samples
        
        # If no samples in non-primary cameras, min_samples = 0
        if min_samples == float('inf'):
            min_samples = 0
        
        # Update progress
        self.calibration_progress = min(1.0, min_samples / MIN_CALIBRATION_SAMPLES)
        
        # If we have enough samples and haven't calibrated yet, do it
        if min_samples >= MIN_CALIBRATION_SAMPLES and not self.is_calibrated:
            self.perform_calibration()
            return True
        return False

    def perform_calibration(self):
        """Execute the calibration process with PyTorch optimization."""
        print("Starting camera calibration with PyTorch...")
        
        # Gather parameters except the primary camera
        params_to_optimize = [p for i, p in enumerate(self.camera_extrinsics)
                              if i != self.primary_camera_idx]
        optimizer = optim.Adam(params_to_optimize, lr=0.01)
        
        # Example batch_size
        batch_size = 50
        num_iterations = 1000
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            total_loss = 0.0
            
            for camera_idx in range(self.num_cameras):
                if camera_idx == self.primary_camera_idx:
                    continue
                if not self.calibration_samples[camera_idx]:
                    continue
                
                # Some random subset
                indices = np.random.choice(len(self.calibration_samples[camera_idx]),
                                           batch_size, replace=False)
                
                # Make sure these go to the GPU (device)
                camera_batch = torch.tensor(
                    [self.calibration_samples[camera_idx][i] for i in indices],
                    dtype=torch.float32,
                    device=device
                )
                primary_batch = torch.tensor(
                    [self.calibration_samples[self.primary_camera_idx][i] for i in indices],
                    dtype=torch.float32,
                    device=device
                )
                
                # Retrieve the extrinsics on GPU
                camera_params  = self.camera_extrinsics[camera_idx].unsqueeze(0)      # shape [1,6], on GPU
                primary_params = self.camera_extrinsics[self.primary_camera_idx].unsqueeze(0)  # [1,6], on GPU
                
                # Example: invert transforms (both on GPU)
                inv_cam  = self.invert_transform_params(camera_params)   # on GPU
                inv_prim = self.invert_transform_params(primary_params)  # on GPU
                
                # Compute loss across batch
                key_indices = [WRIST_IDX, THUMB_TIP_IDX, INDEX_TIP_IDX,
                               MIDDLE_TIP_IDX, RING_TIP_IDX, PINKY_TIP_IDX]
                
                frame_loss = 0.0
                for b in range(batch_size):
                    c_lm = camera_batch[b].unsqueeze(0)   # [1, 21, 3]
                    p_lm = primary_batch[b].unsqueeze(0)  # [1, 21, 3]
                    
                    c_lm_key = c_lm[:, key_indices]
                    p_lm_key = p_lm[:, key_indices]
                    
                    # Transform to "world" space
                    c_world = self.transform_points(c_lm_key, inv_cam)
                    p_world = self.transform_points(p_lm_key, inv_prim)
                    
                    frame_loss += F.mse_loss(c_world, p_world)
                
                total_loss += frame_loss
            
            total_loss.backward()
            optimizer.step()
            
            if (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration+1}/{num_iterations}, Loss={total_loss.item():.6f}")
        
        self.is_calibrated = True
        print("Camera calibration completed!")
    
    def transform_points(self, points, transform_params):
        """
        Transform points using extrinsic parameters.
        points: [B, N, 3]
        transform_params: [B, 6] => [tx, ty, tz, rx, ry, rz]
        """
        batch_size, num_points, _ = points.shape
        translation = transform_params[:, :3].unsqueeze(1)  # [B, 1, 3]
        euler_angles = transform_params[:, 3:6]             # [B, 3]
        
        rotation_matrices = self.hand_model.euler_to_rotmat(euler_angles)  # [B, 3, 3]
        
        # Rotate
        rotated = torch.bmm(points, rotation_matrices.transpose(1, 2))  # [B, N, 3]
        # Translate
        transformed = rotated + translation
        return transformed
    
    def invert_transform_params(self, params):
        """
        Invert transformation [tx, ty, tz, rx, ry, rz].
        Returns the inverted transform in the same 6D format.
        """
        # We'll do a simple approach: T^-1 = R^-1 * (-T), R^-1 = R^T for small angles
        tx, ty, tz, rx, ry, rz = torch.unbind(params, dim=1)
        # Convert to rotation matrix
        euler = torch.stack([rx, ry, rz], dim=1)  # [B, 3]
        rot_mat = self.hand_model.euler_to_rotmat(euler)   # [B, 3, 3]
        
        # Position
        pos = torch.stack([tx, ty, tz], dim=1).unsqueeze(2)  # [B, 3, 1]
        
        # Invert rotation by transposing
        rot_mat_inv = rot_mat.transpose(1, 2)
        
        # Invert position
        inv_pos = -torch.bmm(rot_mat_inv, pos).squeeze(2)  # [B, 3]
        
        # Invert euler angles: negative angles in reverse order
        # A simpler (though not strictly correct for all Euler combos) approach:
        inv_euler = -euler.flip(dims=[1])
        
        return torch.cat([inv_pos, inv_euler], dim=1)


##################################################################
#  GPU Particle Filter
##################################################################

class GPUParticleFilter:
    """GPU-accelerated particle filter for hand pose estimation."""
    def __init__(self, num_particles=NUM_PARTICLES):
        self.num_particles = num_particles
        self.particles = None
        self.weights = torch.ones(num_particles, device=device) / num_particles
        self.initialized = False
        self.hand_model = PytorchHandModel().to(device)
        print(f"Initialized GPU Particle Filter with {num_particles} particles on {device}")
    
    def initialize(self, initial_pose):
        # Extract position & orientation from a 4x4 matrix
        position = torch.tensor(initial_pose[:3, 3], dtype=torch.float32, device=device)
        rotation = Rotation.from_matrix(initial_pose[:3, :3])
        euler = torch.tensor(rotation.as_euler('xyz'), dtype=torch.float32, device=device)
        
        # Create random noise
        position_noise = torch.normal(0.0, PARTICLE_NOISE_POSITION, size=(self.num_particles, 3), device=device)
        rotation_noise = torch.normal(0.0, PARTICLE_NOISE_ROTATION, size=(self.num_particles, 3), device=device)
        
        positions = position.unsqueeze(0).expand(self.num_particles, -1) + position_noise
        rotations = euler.unsqueeze(0).expand(self.num_particles, -1) + rotation_noise
        
        self.particles = torch.cat([positions, rotations], dim=1)  # [N, 6]
        self.weights = torch.ones(self.num_particles, device=device) / self.num_particles
        self.initialized = True
    
    def predict(self):
        """Apply motion model with noise."""
        if not self.initialized:
            return
        pos_noise = torch.normal(0.0, PARTICLE_NOISE_POSITION, size=(self.num_particles, 3), device=device)
        rot_noise = torch.normal(0.0, PARTICLE_NOISE_ROTATION, size=(self.num_particles, 3), device=device)
        self.particles[:, :3] += pos_noise
        self.particles[:, 3:] += rot_noise
    
    def update(self, observations, camera_transforms):
        """
        Update with new observations from multiple cameras.
        observations: dict camera_idx -> (landmarks, confidence)
        camera_transforms: list of 4x4 transformations for each camera
        """
        if not self.initialized:
            return
        
        log_weights = torch.zeros(self.num_particles, device=device)
        
        # For each camera
        for camera_idx, (landmarks, confidence) in observations.items():
            if landmarks is None or confidence < 0.6:
                continue
            transform = camera_transforms[camera_idx]
            if transform is None:
                continue
            
            # Convert to torch
            obs = torch.tensor(landmarks, dtype=torch.float32, device=device).unsqueeze(0)
            key_indices = [WRIST_IDX, THUMB_TIP_IDX, INDEX_TIP_IDX, MIDDLE_TIP_IDX, RING_TIP_IDX, PINKY_TIP_IDX]
            obs_key = obs[:, key_indices]  # [1, 6, 3]
            
            # Convert transform -> 6D
            rot = Rotation.from_matrix(transform[:3, :3])
            euler = rot.as_euler('xyz')
            cam_params = torch.tensor(
                np.concatenate([transform[:3, 3], euler]),
                dtype=torch.float32,
                device=device
            ).unsqueeze(0)  # [1, 6]
            

            # Evaluate in batches
            batch_size = 1000
            for start_idx in range(0, self.num_particles, batch_size):
                end_idx = min(start_idx + batch_size, self.num_particles)
                batch_particles = self.particles[start_idx:end_idx]  # [bs, 6]
                
                # Predict hand keypoints in world space
                predicted_hand = self.hand_model.forward(batch_particles)  # [bs, 21, 3]
                predicted_key = predicted_hand[:, key_indices]  # [bs, 6, 3]

                # Transform predicted to camera space (world -> camera)
                # We'll do an easy approach: just invert cam_params if needed,
                # but let's define direct transform for world->cam as cam_params.
                cam_params_expanded = cam_params.expand(batch_particles.shape[0], -1)
                predicted_cam = self.transform_points(predicted_key, cam_params_expanded)  # [bs, 6, 3]
                
                # Compare with obs_key
                sq_dist = torch.sum((predicted_cam - obs_key) ** 2, dim=2)  # [bs, 6]
                mean_err = torch.mean(sq_dist, dim=1)                       # [bs]
                
                log_weights[start_idx:end_idx] += -mean_err * 100.0 * confidence
        
        # Convert log weights to normalized weights
        max_log_weight = torch.max(log_weights)
        weights = torch.exp(log_weights - max_log_weight)
        w_sum = torch.sum(weights)
        if w_sum > 0:
            self.weights = weights / w_sum
        else:
            self.weights = torch.ones_like(self.weights) / self.num_particles
        
        # Resample if needed
        n_eff = 1.0 / torch.sum(self.weights ** 2)
        if n_eff < self.num_particles / 2:
            self.resample()
    
    def transform_points(self, points, transform_params):
        """Transform points using extrinsic parameters [tx, ty, tz, rx, ry, rz]."""
        batch_size, num_points, _ = points.shape
        translation = transform_params[:, :3].unsqueeze(1)  # [B,1,3]
        euler_angles = transform_params[:, 3:6]             # [B,3]
        
        rotation_matrices = self.hand_model.euler_to_rotmat(euler_angles)  # [B,3,3]
        
        rotated = torch.bmm(points, rotation_matrices.transpose(1, 2))  # [B,N,3]
        transformed = rotated + translation
        return transformed
    
    def resample(self):
        """Multinomial resampling."""
        indices = torch.multinomial(self.weights, self.num_particles, replacement=True)
        self.particles = self.particles[indices]
        self.weights = torch.ones_like(self.weights) / self.num_particles
    
    def get_estimated_pose(self):
        """Return the pose as a 4x4 matrix from the best or weighted mean of particles."""
        if not self.initialized:
            return np.eye(4)
        # Weighted mean of position
        pos = torch.sum(self.particles[:, :3] * self.weights.unsqueeze(1), dim=0).cpu().numpy()
        # Take rotation from best particle
        best_idx = torch.argmax(self.weights).item()
        euler = self.particles[best_idx, 3:6].cpu().numpy()
        rot_mat = Rotation.from_euler('xyz', euler).as_matrix()
        
        pose = np.eye(4)
        pose[:3, :3] = rot_mat
        pose[:3, 3] = pos
        return pose


##################################################################
#  Enhanced Hand Pose Fuser
##################################################################

class EnhancedHandPoseFuser:
    """Fuses hand poses from multiple cameras using a GPU particle filter."""
    
    def __init__(self, num_cameras, primary_camera_idx=0):
        self.num_cameras = num_cameras
        self.primary_camera_idx = primary_camera_idx
        self.particle_filter = GPUParticleFilter(NUM_PARTICLES)
        self.initialized = False
        self.hand_pose_history = deque(maxlen=10)
        self.hand_model = PytorchHandModel().to(device)
    
    def update(self, camera_observations, camera_transforms):
        """
        Update the fused hand pose with new observations.
        camera_observations: dict of camera_idx -> (landmarks, confidence)
        camera_transforms: list of 4x4 transforms for each camera
        """
        # Filter valid observations
        valid_obs = {idx: obs for idx, obs in camera_observations.items()
                     if obs[0] is not None and obs[1] > 0.5}
        
        if not valid_obs:
            return None
        
        # Initialize PF if needed and we have an observation from the primary camera
        if not self.initialized and self.primary_camera_idx in valid_obs:
            landmarks, _ = valid_obs[self.primary_camera_idx]
            # Use wrist as initial position
            initial_pose = np.eye(4)
            initial_pose[:3, 3] = landmarks[WRIST_IDX]
            self.particle_filter.initialize(initial_pose)
            self.initialized = True
        
        if not self.initialized:
            return None
        
        # Particle filter steps
        self.particle_filter.predict()
        self.particle_filter.update(valid_obs, camera_transforms)
        
        fused_pose = self.particle_filter.get_estimated_pose()
        self.hand_pose_history.append(fused_pose)
        return self.get_smoothed_pose()
    
    def get_smoothed_pose(self):
        """Simple smoothing by weighted average of recent poses."""
        if not self.hand_pose_history:
            return None
        
        # Weighted average over the last N poses
        weights = np.linspace(0.5, 1.0, len(self.hand_pose_history))
        weights /= weights.sum()
        
        positions = np.array([pose[:3, 3] for pose in self.hand_pose_history])
        avg_pos = np.average(positions, axis=0, weights=weights)
        
        # Average rotation via quaternions
        quats = []
        for pose in self.hand_pose_history:
            r = Rotation.from_matrix(pose[:3, :3])
            quats.append(r.as_quat())
        quats = np.array(quats)
        
        # Fix signs for averaging
        ref = quats[0]
        for i in range(1, len(quats)):
            if np.dot(ref, quats[i]) < 0:
                quats[i] = -quats[i]
        
        avg_quat = np.average(quats, axis=0, weights=weights)
        avg_quat /= np.linalg.norm(avg_quat)
        
        rot_mat = Rotation.from_quat(avg_quat).as_matrix()
        
        smoothed = np.eye(4)
        smoothed[:3, :3] = rot_mat
        smoothed[:3, 3] = avg_pos
        return smoothed
    
    def get_hand_landmarks_from_pose(self, pose):
        """Generate the 21 hand landmarks from a given 4x4 pose."""
        rot = Rotation.from_matrix(pose[:3, :3])
        euler = rot.as_euler('xyz')
        
        params = torch.tensor(
            np.concatenate([pose[:3, 3], euler]),
            dtype=torch.float32, device=device
        ).unsqueeze(0)
        
        with torch.no_grad():
            landmarks = self.hand_model.forward(params)  # [1,21,3]
        return landmarks[0].cpu().numpy()


##################################################################
#  3D Visualizer
##################################################################

class HandVisualizer3D:
    """3D Matplotlib visualization for the tracked hand pose."""
    
    def __init__(self):
        self.fig = plt.figure(figsize=(8,8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-0.2, 0.2)
        self.ax.set_ylim(-0.2, 0.2)
        self.ax.set_zlim(-0.2, 0.2)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("3D Hand Pose")
        
        # Scatter for hand points
        self.hand_scatter = self.ax.scatter([], [], [], marker='o', s=50)
        self.wrist_scatter = self.ax.scatter([], [], [], marker='o', s=100, color='r')
        
        # Connections
        self.lines = []
        self.connections = [
            # Thumb
            (WRIST_IDX, THUMB_CMC_IDX),
            (THUMB_CMC_IDX, THUMB_MCP_IDX),
            (THUMB_MCP_IDX, THUMB_IP_IDX),
            (THUMB_IP_IDX, THUMB_TIP_IDX),
            
            # Index
            (WRIST_IDX, INDEX_MCP_IDX),
            (INDEX_MCP_IDX, INDEX_PIP_IDX),
            (INDEX_PIP_IDX, INDEX_DIP_IDX),
            (INDEX_DIP_IDX, INDEX_TIP_IDX),
            
            # Middle
            (WRIST_IDX, MIDDLE_MCP_IDX),
            (MIDDLE_MCP_IDX, MIDDLE_PIP_IDX),
            (MIDDLE_PIP_IDX, MIDDLE_DIP_IDX),
            (MIDDLE_DIP_IDX, MIDDLE_TIP_IDX),
            
            # Ring
            (WRIST_IDX, RING_MCP_IDX),
            (RING_MCP_IDX, RING_PIP_IDX),
            (RING_PIP_IDX, RING_DIP_IDX),
            (RING_DIP_IDX, RING_TIP_IDX),
            
            # Pinky
            (WRIST_IDX, PINKY_MCP_IDX),
            (PINKY_MCP_IDX, PINKY_PIP_IDX),
            (PINKY_PIP_IDX, PINKY_DIP_IDX),
            (PINKY_DIP_IDX, PINKY_TIP_IDX),
            
            # Palm
            (INDEX_MCP_IDX, MIDDLE_MCP_IDX),
            (MIDDLE_MCP_IDX, RING_MCP_IDX),
            (RING_MCP_IDX, PINKY_MCP_IDX)
        ]
        for _ in self.connections:
            line, = self.ax.plot([], [], [], 'gray')
            self.lines.append(line)
        
        # Axis frame lines
        self.axes_artists = [
            self.ax.plot([], [], [], 'r-', linewidth=2)[0],  # x
            self.ax.plot([], [], [], 'g-', linewidth=2)[0],  # y
            self.ax.plot([], [], [], 'b-', linewidth=2)[0]   # z
        ]
        
        plt.ion()
        plt.show()
    
    def update(self, landmarks, pose):
        if landmarks is None:
            return
        
        xs = landmarks[:, 0]
        ys = landmarks[:, 1]
        zs = landmarks[:, 2]
        
        # Hand scatter: all except wrist
        self.hand_scatter._offsets3d = (xs[1:], ys[1:], zs[1:])
        # Wrist scatter: index 0
        self.wrist_scatter._offsets3d = ([xs[0]], [ys[0]], [zs[0]])
        
        # Update lines
        for i, (start, end) in enumerate(self.connections):
            self.lines[i].set_data([xs[start], xs[end]], [ys[start], ys[end]])
            self.lines[i].set_3d_properties([zs[start], zs[end]])
        
        # Pose axes
        origin = pose[:3, 3]
        axis_len = 0.1
        x_axis = origin + axis_len * pose[:3, 0]
        y_axis = origin + axis_len * pose[:3, 1]
        z_axis = origin + axis_len * pose[:3, 2]
        
        # X-axis
        self.axes_artists[0].set_data([origin[0], x_axis[0]],
                                      [origin[1], x_axis[1]])
        self.axes_artists[0].set_3d_properties([origin[2], x_axis[2]])
        
        # Y-axis
        self.axes_artists[1].set_data([origin[0], y_axis[0]],
                                      [origin[1], y_axis[1]])
        self.axes_artists[1].set_3d_properties([origin[2], y_axis[2]])
        
        # Z-axis
        self.axes_artists[2].set_data([origin[0], z_axis[0]],
                                      [origin[1], z_axis[1]])
        self.axes_artists[2].set_3d_properties([origin[2], z_axis[2]])
        
        # Rescale axes to fit
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        self.ax.set_xlim(min_x - 0.1, max_x + 0.1)
        self.ax.set_ylim(min_y - 0.1, max_y + 0.1)
        self.ax.set_zlim(min_z - 0.1, max_z + 0.1)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def close(self):
        plt.close(self.fig)


##################################################################
#  Main Enhanced Multi-Camera Hand Tracker
##################################################################

class EnhancedMultiCameraHandTracker:
    """Main class for multi-camera hand tracking (with optional OAK-D)."""
    
    def __init__(self, config_path=CONFIG_PATH, show_visualizations=True):
        self.config = CameraConfig.load_config(config_path)
        self.primary_camera_idx = CameraConfig.get_primary_camera_idx(self.config)
        self.camera_ids = [cam["id"] for cam in self.config["cameras"]]
        self.num_cameras = len(self.camera_ids)
        self.show_visualizations = show_visualizations
        
        print(f"Primary camera index: {self.primary_camera_idx}")
        print(f"Using cameras: {self.camera_ids}")
        
        # Initialize camera trackers
        self.camera_trackers = []
        for cam_cfg in self.config["cameras"]:
            cam_id = cam_cfg["id"]
            cam_type = cam_cfg.get("type", CAMERA_TYPE_STANDARD)
            intrinsics = cam_cfg.get("intrinsics", DEFAULT_INTRINSICS.copy())
            
            if cam_type == CAMERA_TYPE_OAKD:
                # If you have actual device_info, pass it here. 
                # This example passes None or a dummy index.
                # Adjust logic as needed to pick the correct OAK-D device.
                tracker = OAKDCameraTracker(
                    device_info=None,    # In real usage, you’d find matching device_info
                    index=cam_id,
                    intrinsics=intrinsics,
                    show=show_visualizations
                )
            else:
                tracker = SingleCameraTracker(
                    camera_id=cam_id,
                    intrinsics=intrinsics,
                    show=show_visualizations
                )
            self.camera_trackers.append(tracker)
        
        # Calibrator
        self.calibrator = PyTorchCameraCalibrator(self.num_cameras, self.primary_camera_idx)
        
        # Pose fuser
        self.pose_fuser = EnhancedHandPoseFuser(self.num_cameras, self.primary_camera_idx)
        
        # State & control
        self.state = "INITIALIZING"  # or CALIBRATING, TRACKING
        self.running = False
        self.calibration_start_time = None
        
        # Visualizer
        self.visualizer = HandVisualizer3D() if show_visualizations else None
    
    def start(self):
        # Start camera trackers
        for t in self.camera_trackers:
            t.start()
        
        self.running = True
        self.state = "INITIALIZING"
        print("Initializing - press SPACE to start calibration.")
        
        try:
            while self.running:
                self.update()
                self.display()
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    self.running = False
                elif key == 32:  # SPACE
                    if self.state == "INITIALIZING":
                        self.start_calibration()
                    elif self.state == "CALIBRATING" and self.calibrator.is_calibrated:
                        self.state = "TRACKING"
                        print("Calibration complete - now tracking")
        finally:
            self.stop()
    
    def start_calibration(self):
        self.state = "CALIBRATING"
        self.calibration_start_time = time.time()
        print("Starting calibration - move your hand around the views.")
    
    def update(self):
        observations = {}
        for idx, tracker in enumerate(self.camera_trackers):
            lm, conf = tracker.get_hand_landmarks()
            observations[idx] = (lm, conf)
        
        if self.state == "CALIBRATING":
            # Add samples
            for idx, (lm, conf) in observations.items():
                if lm is not None:
                    self.calibrator.add_sample(idx, lm, conf)
            # Check progress
            self.calibrator.check_calibration_progress()
            
            # Possibly auto-switch after some time if done
            if self.calibrator.is_calibrated:
                if (time.time() - self.calibration_start_time) > 5:
                    self.state = "TRACKING"
                    print("Calibration complete - now tracking.")
        
        elif self.state == "TRACKING":
            if self.calibrator.is_calibrated:
                camera_transforms = [
                    self.calibrator.get_transform_matrix(i) for i in range(self.num_cameras)
                ]
                fused_pose = self.pose_fuser.update(observations, camera_transforms)
                if fused_pose is not None and self.visualizer is not None:
                    # Get landmarks in fused world space
                    landmarks_3d = self.pose_fuser.get_hand_landmarks_from_pose(fused_pose)
                    self.visualizer.update(landmarks_3d, fused_pose)
    
    def display(self):
        if not self.show_visualizations:
            return
        
        # Gather processed frames
        frames = []
        for t in self.camera_trackers:
            if t.processed_frame is not None:
                frames.append(t.processed_frame)
        
        if not frames:
            return
        
        # Make sure all frames are same size
        h, w = frames[0].shape[:2]
        for i in range(1, len(frames)):
            frames[i] = cv2.resize(frames[i], (w, h))
        
        if len(frames) == 1:
            grid = frames[0]
        elif len(frames) == 2:
            grid = np.hstack(frames)
        else:
            # up to 4 in a 2x2
            top_row = np.hstack(frames[:2])
            if len(frames) > 2:
                bottom_row = frames[2]
                if len(frames) > 3:
                    bottom_row = np.hstack([bottom_row, frames[3]])
                else:
                    # add black placeholder
                    bottom_row = np.hstack([bottom_row, np.zeros_like(frames[0])])
                grid = np.vstack([top_row, bottom_row])
            else:
                grid = top_row
        
        status_text = f"State: {self.state}"
        if self.state == "CALIBRATING":
            progress = int(self.calibrator.calibration_progress * 100)
            status_text += f" ({progress}%)"
        cv2.putText(
            grid,
            status_text,
            (10, grid.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.imshow("Multi-Camera Hand Tracker", grid)
    
    def stop(self):
        for t in self.camera_trackers:
            t.stop()
        cv2.destroyAllWindows()
        if self.visualizer:
            self.visualizer.close()
        print("Shutting down.")


##################################################################
#  Utility to create a default config
##################################################################

def create_default_config():
    """Create a default camera config file for up to MAX_CAMERAS standard webcams."""
    config = {"cameras": []}
    
    # Probe system for available cameras
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    
    camera_ids = available[:MAX_CAMERAS]
    for i, cid in enumerate(camera_ids):
        config["cameras"].append({
            "id": cid,
            "primary": (i == 0),
            "type": CAMERA_TYPE_STANDARD,
            "intrinsics": DEFAULT_INTRINSICS.copy()
        })
    
    CameraConfig.save_config(config)
    print(f"Created default configuration with cameras: {camera_ids}")
    return config


##################################################################
#  Main entry point
##################################################################

def main():
    """Run the multi-camera hand tracking system."""
    print("=== Enhanced Multi-Camera Hand Tracking with OAK-D Support ===")
    parser = argparse.ArgumentParser(description='Multi-Camera Hand Tracking System')
    parser.add_argument('--reset-config', action='store_true', help='Reset configuration file')
    parser.add_argument('--no-viz', action='store_true', help='Disable 3D visualization')
    parser.add_argument('--config', type=str, default=CONFIG_PATH, help='Path to config file')
    args = parser.parse_args()
    
    # If config is missing or reset requested, create a new one
    if args.reset_config or not os.path.exists(args.config):
        print("Creating default configuration...")
        create_default_config()
    
    # Create the tracker
    tracker = EnhancedMultiCameraHandTracker(
        config_path=args.config,
        show_visualizations=not args.no_viz
    )
    
    # Check camera count
    if len(tracker.camera_trackers) == 0:
        print("No cameras found. Please check your hardware.")
        return
    
    print("\nInstructions:")
    print("1. Position a single hand visible to at least the primary camera.")
    print("2. Press SPACE to start calibration.")
    print("3. Move your hand slowly to cover the field of view.")
    print("4. Once calibrated, the system will track your hand in 3D.")
    print("5. Press ESC to exit.\n")
    
    tracker.start()


if __name__ == "__main__":
    # Optional: If you want to use CUDA with OpenCV
    try:
        cv2.setUseOptimized(True)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print(f"OpenCV sees {cv2.cuda.getCudaEnabledDeviceCount()} CUDA device(s).")
    except:
        print("OpenCV CUDA support not available.")
    
    # Check DepthAI version if installed
    try:
        import depthai
        print(f"DepthAI version: {depthai.__version__}")
    except ImportError:
        pass
    except Exception as e:
        print(f"DepthAI error: {e}")
    
    main()
