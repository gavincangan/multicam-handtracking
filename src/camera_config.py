##################################################################
#  Camera Configuration
##################################################################

import yaml
import os
import cv2

from .constants import *


def create_default_config():
    """Create a default camera config file for all available cameras, with built-in as primary."""
    config = {"cameras": []}
    
    # Probe system for available cameras
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    
    # If no cameras found, return empty config
    if not available:
        print("No cameras detected")
        CameraConfig.save_config(config)
        return config
        
    # Built-in camera is typically the first one (index 0)
    built_in_camera_idx = 0
    # First add the built-in camera as primary
    config["cameras"].append({
        "id": available[built_in_camera_idx],
        "primary": True,
        "type": CAMERA_TYPE_STANDARD,
        "name": "Built-in Camera",
        "intrinsics": DEFAULT_INTRINSICS.copy()
    })
    
    # Then add all other cameras as supporting
    other_cameras = [idx for idx in available if idx != available[built_in_camera_idx]]
    camera_ids = other_cameras[:MAX_CAMERAS-1]  # Limit to MAX_CAMERAS-1 (since built-in is already added)
    
    for i, cid in enumerate(camera_ids):
        config["cameras"].append({
            "id": cid,
            "primary": False,
            "type": CAMERA_TYPE_STANDARD,
            "name": f"External Camera {i+1}",
            "intrinsics": DEFAULT_INTRINSICS.copy()
        })
    
    CameraConfig.save_config(config)
    all_camera_ids = [camera["id"] for camera in config["cameras"]]
    print(f"Created default configuration with built-in camera as primary and {len(all_camera_ids)} total cameras")
    print(f"Camera IDs: {all_camera_ids}")
    return config


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
            # Create and return a default config by detecting cameras
            return create_default_config()
            
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
