##################################################################
#  Enhanced Hand Pose Fuser
##################################################################

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from collections import deque

from .hand_tracker import PytorchHandModel
from .gpu_particle_filter import GPUParticleFilter
from .constants import *
from .camera_calibrator import PyTorchCameraCalibrator

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