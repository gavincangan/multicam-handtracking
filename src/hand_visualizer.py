
##################################################################
#  3D Visualizer
##################################################################

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation

from .constants import *


class HandVisualizer3D:
    """3D Matplotlib visualization for the tracked hand pose."""
    
    def __init__(self):
        self.is_active = False  # Flag to control visualization
        self.is_active = False  # Flag to control visualization
        self.fig = plt.figure(figsize=(10,8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(-0.5, 0.5)
        self.ax.set_ylim(-0.5, 0.5)
        self.ax.set_zlim(-0.5, 0.5)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("3D Hand Pose and Camera Positions")
        # Scatter for hand points
        self.hand_scatter = self.ax.scatter([], [], [], marker='o', s=50)
        self.wrist_scatter = self.ax.scatter([], [], [], marker='o', s=100, color='r')
        
        # Camera visualizations
        self.camera_scatter = None
        self.camera_labels = []
        self.camera_axes = []
        
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
        
        # Define palm connections (for later coloring)
        self.palm_connections = [
            (INDEX_MCP_IDX, MIDDLE_MCP_IDX),
            (MIDDLE_MCP_IDX, RING_MCP_IDX),
            (RING_MCP_IDX, PINKY_MCP_IDX)
        ]
        
        for i, connection in enumerate(self.connections):
            # Use orange color for palm connections, gray for others
            color = 'orange' if connection in self.palm_connections else 'gray'
            line, = self.ax.plot([], [], [], color, linewidth=2 if color == 'orange' else 1)
            self.lines.append(line)
        
        # Axis frame lines
        self.axes_artists = [
            self.ax.plot([], [], [], 'r-', linewidth=2)[0],  # x
            self.ax.plot([], [], [], 'g-', linewidth=2)[0],  # y
            self.ax.plot([], [], [], 'b-', linewidth=2)[0]   # z
        ]
        
        plt.ion()
        plt.show()
    
    def transform_points(self, points, pose):
        """
        Transform 3D points using the given pose matrix.
        
        Args:
            points: Nx3 array of points
            pose: 4x4 transformation matrix
            
        Returns:
            Transformed points as Nx3 array
        """
        # Make homogeneous coordinates (Nx4)
        homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Apply transformation (matrix multiplication)
        transformed_homogeneous = np.dot(homogeneous_points, pose.T)
        
        # Convert back to 3D points
        transformed_points = transformed_homogeneous[:, :3]
        
        return transformed_points
    
    def extract_camera_info(self, transform_matrix):
        """
        Extract camera position and orientation from a transformation matrix.
        
        Args:
            transform_matrix: 4x4 transformation matrix
            
        Returns:
            position: 3D position of the camera
            rotation: 3x3 rotation matrix representing camera orientation
        """
        # Camera position is the translation component of the matrix
        position = transform_matrix[:3, 3]
        
        # Camera orientation is the rotation component of the matrix
        rotation = transform_matrix[:3, :3]
        
        return position, rotation
    
    def update_camera_visualization(self, camera_transforms, primary_idx=0):
        """
        Update the camera visualization based on camera transforms.
        
        Args:
            camera_transforms: List of 4x4 transformation matrices
            primary_idx: Index of the primary camera
        """
        if not camera_transforms or not self.is_active:
            return
        
        # Clear previous camera visualizations
        if self.camera_scatter is not None:
            self.camera_scatter.remove()
            
        for label in self.camera_labels:
            label.remove()
        self.camera_labels = []
        
        for axes in self.camera_axes:
            for axis in axes:
                axis.remove()
        self.camera_axes = []
        
        # Extract camera positions and colors
        positions = []
        colors = []
        
        for i, transform in enumerate(camera_transforms):
            pos, rot = self.extract_camera_info(transform)
            positions.append(pos)
            
            # Primary camera is red, others are blue
            color = 'red' if i == primary_idx else 'blue'
            colors.append(color)
            
            # Add camera label
            label = self.ax.text(pos[0], pos[1], pos[2], f"Camera {i}", color=color, fontsize=10)
            self.camera_labels.append(label)
            
            # Add camera axes (smaller than hand axes)
            axis_len = 0.15
            axes_artists = []
            
            # X-axis (red)
            x_axis = pos + axis_len * rot[:, 0]
            x_line = self.ax.plot([pos[0], x_axis[0]],
                                 [pos[1], x_axis[1]],
                                 [pos[2], x_axis[2]], 'r-', linewidth=1.5)[0]
            axes_artists.append(x_line)
            
            # Y-axis (green)
            y_axis = pos + axis_len * rot[:, 1]
            y_line = self.ax.plot([pos[0], y_axis[0]],
                                 [pos[1], y_axis[1]],
                                 [pos[2], y_axis[2]], 'g-', linewidth=1.5)[0]
            axes_artists.append(y_line)
            
            # Z-axis (blue)
            z_axis = pos + axis_len * rot[:, 2]
            z_line = self.ax.plot([pos[0], z_axis[0]],
                                 [pos[1], z_axis[1]],
                                 [pos[2], z_axis[2]], 'b-', linewidth=1.5)[0]
            axes_artists.append(z_line)
            
            self.camera_axes.append(axes_artists)
        
        # Create scatter plot for camera positions
        positions = np.array(positions)
        self.camera_scatter = self.ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2], 
            color=colors, marker='^', s=100, label="Cameras"
        )
    
    def update_axis_limits(self, points_list):
        """
        Update axis limits to include all points with a margin.
        
        Args:
            points_list: List of Nx3 arrays containing points to include
        """
        if not points_list:
            return
            
        # Combine all points
        all_points = np.vstack(points_list)
        
        # Get min and max for each axis
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)
        
        # Calculate the range and center
        ranges = max_vals - min_vals
        centers = (min_vals + max_vals) / 2
        
        # Use the maximum range for all axes to maintain proportions
        max_range = np.max(ranges) * 1.2  # Add 20% margin
        
        # Set limits
        self.ax.set_xlim(centers[0] - max_range/2, centers[0] + max_range/2)
        self.ax.set_ylim(centers[1] - max_range/2, centers[1] + max_range/2)
        self.ax.set_zlim(centers[2] - max_range/2, centers[2] + max_range/2)
    
    def update(self, landmarks, pose, camera_transforms=None, primary_camera_idx=0):
        """
        Update the visualization with hand landmarks and camera positions.
        
        Args:
            landmarks: Nx3 array of hand landmark points
            pose: 4x4 transformation matrix for the hand
            camera_transforms: List of 4x4 transformation matrices for cameras
            primary_camera_idx: Index of the primary camera
        """
        # Only update visualization when active
        if landmarks is None or not self.is_active:
            return
        
        # Transform landmarks using the pose matrix
        transformed_landmarks = self.transform_points(landmarks, pose)
        
        xs = transformed_landmarks[:, 0]
        ys = transformed_landmarks[:, 1]
        zs = transformed_landmarks[:, 2]
        
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
        axis_len = 0.25
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
        
        self.axes_artists[2].set_data([origin[0], z_axis[0]],
                                      [origin[1], z_axis[1]])
        self.axes_artists[2].set_3d_properties([origin[2], z_axis[2]])
        
        # Update camera visualization if provided
        if camera_transforms is not None:
            self.update_camera_visualization(camera_transforms, primary_camera_idx)
            
        # Update axis limits to include both hand and cameras
        points_to_include = [transformed_landmarks]
        
        # Add camera positions if available
        if camera_transforms is not None:
            camera_positions = np.array([t[:3, 3] for t in camera_transforms])
            points_to_include.append(camera_positions)
            
        # Update axis limits based on all points
        self.update_axis_limits(points_to_include)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.draw_idle()
    
    def close(self):
        plt.close(self.fig)

    def start_visualization(self):
        """Start the visualization process."""
        self.is_active = True
        print("3D visualization started")

    def stop_visualization(self):
        """Stop the visualization process."""
        self.is_active = False
        print("3D visualization stopped")
