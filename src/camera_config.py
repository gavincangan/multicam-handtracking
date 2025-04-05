##################################################################
#  Camera Configuration
##################################################################

import yaml
from constants import *
import os
import cv2

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
