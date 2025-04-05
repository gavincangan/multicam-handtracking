
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