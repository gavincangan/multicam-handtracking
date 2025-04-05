
##################################################################
#  Pytorch Hand Model
##################################################################

import torch
import torch.nn as nn

from .constants import *

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
