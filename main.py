import argparse
import os
import cv2

from src.constants import *
from src.multi_camera_tracker import EnhancedMultiCameraHandTracker
from src.camera_config import CameraConfig, create_default_config


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
