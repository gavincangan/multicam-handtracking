##################################################################
#  PyTorch-Based Camera Calibrator
##################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from scipy.spatial.transform import Rotation
import numpy as np
from .constants import *
from .hand_tracker import PytorchHandModel


class PyTorchCameraCalibrator:
    """Camera calibration using PyTorch for differentiable optimization."""
    
    def __init__(self, num_cameras, primary_camera_idx=0, visualizer=None):
        self.num_cameras = num_cameras
        self.primary_camera_idx = primary_camera_idx
        self.calibration_samples = [[] for _ in range(num_cameras)]
        self.visualizer = visualizer  # Reference to the HandVisualizer3D instance
        self.collection_started = False  # Flag to track if collection has started
        
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
            # Start visualization when first sample is collected
            if not self.collection_started and self.visualizer is not None:
                self.visualizer.start_visualization()
                self.collection_started = True
                
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
        # Find the minimum number of samples across all cameras
        min_samples = float('inf')
        for i in range(self.num_cameras):
            num_samples = len(self.calibration_samples[i])
            if num_samples < min_samples:
                min_samples = num_samples
        
        # If no samples in any cameras, min_samples = 0
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
        
        # Example batch_size - ensure it's not larger than our smallest sample set
        min_samples = min(len(samples) for samples in self.calibration_samples)
        batch_size = min(50, min_samples)
        num_iterations = 300
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            total_loss = 0.0
            
            # Generate indices that are valid for all cameras
            valid_indices = np.random.choice(min_samples, batch_size, replace=False)
            
            for camera_idx in range(self.num_cameras):
                if camera_idx == self.primary_camera_idx:
                    continue
                
                # Make sure these go to the GPU (device)
                # First create a single numpy array, then convert to tensor for better performance
                camera_batch = torch.tensor(
                    np.stack([self.calibration_samples[camera_idx][i] for i in valid_indices]),
                    dtype=torch.float32,
                    device=device
                )
                primary_batch = torch.tensor(
                    np.stack([self.calibration_samples[self.primary_camera_idx][i] for i in valid_indices]),
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
        
        # Stop visualization when calibration is complete
        if self.visualizer is not None:
            self.visualizer.stop_visualization()
            
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
