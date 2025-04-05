
##################################################################
#  3D Visualizer
##################################################################

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation

from .constants import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation

from .constants import *


class HandVisualizer3D:
    """3D Matplotlib visualization for the tracked hand pose and camera positions."""
    
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-0.5, 0.5)  # Broader limits to accommodate cameras
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_zlim(-0.5, 0.5)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("3D Hand Pose with Camera Positions")
        
        # Scatter for hand points
        self.hand_scatter = self.ax.scatter([], [], [], marker='o', s=50)
        self.wrist_scatter = self.ax.scatter([], [], [], marker='o', s=100, color='r')
        
        # Camera visualization
        self.camera_scatters = []
        self.camera_axes = []  # List to store camera coordinate axes
        
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
        
        # Axis frame lines for hand pose
        self.axes_artists = [
            self.ax.plot([], [], [], 'r-', linewidth=2)[0],  # x
            self.ax.plot([], [], [], 'g-', linewidth=2)[0],  # y
            self.ax.plot([], [], [], 'b-', linewidth=2)[0]   # z
        ]
        
        plt.ion()
        plt.show()
    
    def update(self, landmarks, pose, camera_transforms=None, primary_camera_idx=0):
        """
        Update the visualization with hand landmarks, pose, and camera positions.
        
        Parameters:
        landmarks: 3D landmarks of hand keypoints
        pose: Transformation matrix of hand pose
        camera_transforms: List of camera transformation matrices relative to world
        primary_camera_idx: Index of the primary camera (which will be at origin)
        """
        if landmarks is None:
            return
        
        # Update hand landmarks and pose
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
        
        # Update camera visualizations if provided
        if camera_transforms is not None:
            # Clear previous camera visualizations
            for scatter in self.camera_scatters:
                scatter.remove()
            for axes_set in self.camera_axes:
                for axis in axes_set:
                    axis.remove()
                    
            self.camera_scatters = []
            self.camera_axes = []
            
            # Get the inverse of the primary camera transform to make it the origin
            primary_cam_inv = np.linalg.inv(camera_transforms[primary_camera_idx])
            
            # Add each camera in the primary camera's reference frame
            for i, cam_transform in enumerate(camera_transforms):
                # Transform camera position to primary camera frame
                rel_transform = primary_cam_inv @ cam_transform
                cam_pos = rel_transform[:3, 3]
                
                # Add camera position marker
                color = 'blue' if i == primary_camera_idx else 'green'
                size = 150 if i == primary_camera_idx else 100
                scatter = self.ax.scatter(
                    cam_pos[0], cam_pos[1], cam_pos[2], 
                    color=color, marker='s', s=size, 
                    label=f"Camera {i}")
                self.camera_scatters.append(scatter)
                
                # Add coordinate axes for each camera
                axis_len = 0.15
                
                # Calculate axis endpoints
                x_end = cam_pos + axis_len * rel_transform[:3, 0]
                y_end = cam_pos + axis_len * rel_transform[:3, 1]
                z_end = cam_pos + axis_len * rel_transform[:3, 2]
                
                # Create axes lines
                x_axis = self.ax.plot([cam_pos[0], x_end[0]], 
                                      [cam_pos[1], x_end[1]], 
                                      [cam_pos[2], x_end[2]], 'r-', linewidth=2)[0]
                
                y_axis = self.ax.plot([cam_pos[0], y_end[0]], 
                                      [cam_pos[1], y_end[1]], 
                                      [cam_pos[2], z_end[2]], 'g-', linewidth=2)[0]
                
                z_axis = self.ax.plot([cam_pos[0], z_end[0]], 
                                      [cam_pos[1], z_end[1]], 
                                      [cam_pos[2], z_end[2]], 'b-', linewidth=2)[0]
                
                self.camera_axes.append([x_axis, y_axis, z_axis])
            
            # Add a legend
            self.ax.legend()
        
        # Fixed/expanded axes limits to accommodate cameras and hand
        min_x, max_x = min(min(xs), -0.5), max(max(xs), 0.5)
        min_y, max_y = min(min(ys), -0.5), max(max(ys), 0.5)
        min_z, max_z = min(min(zs), -0.5), max(max(zs), 0.5)
        
        # Expand by 20% in each direction for better visibility
        range_x = max_x - min_x
        range_y = max_y - min_y
        range_z = max_z - min_z
        
        self.ax.set_xlim(min_x - 0.2 * range_x, max_x + 0.2 * range_x)
        self.ax.set_ylim(min_y - 0.2 * range_y, max_y + 0.2 * range_y)
        self.ax.set_zlim(min_z - 0.2 * range_z, max_z + 0.2 * range_z)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def close(self):
        plt.close(self.fig)