##################################################################
#  GPU Particle Filter
##################################################################

import torch
import numpy as np
from scipy.spatial.transform import Rotation

from .hand_tracker import PytorchHandModel
from .constants import *

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