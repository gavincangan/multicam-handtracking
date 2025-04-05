##################################################################
# Enhanced Multi-Camera Hand Tracker
##################################################################

import cv2
import numpy as np
import time

from camera_config import CameraConfig
from base_camera_tracker import BaseCameraTracker
from oakd_handler import OAKDCameraTracker
from hand_pose_fuser import EnhancedHandPoseFuser
from hand_visualizer import HandVisualizer3D
from camera_calibrator import PyTorchCameraCalibrator

from constants import *

class EnhancedMultiCameraHandTracker:
    """Main class for multi-camera hand tracking (with optional OAK-D)."""
    
    def __init__(self, config_path=CONFIG_PATH, show_visualizations=True):
        self.config = CameraConfig.load_config(config_path)
        self.primary_camera_idx = CameraConfig.get_primary_camera_idx(self.config)
        self.camera_ids = [cam["id"] for cam in self.config["cameras"]]
        self.num_cameras = len(self.camera_ids)
        self.show_visualizations = show_visualizations
        
        print(f"Primary camera index: {self.primary_camera_idx}")
        print(f"Using cameras: {self.camera_ids}")
        
        # Initialize camera trackers
        self.camera_trackers = []
        for cam_cfg in self.config["cameras"]:
            cam_id = cam_cfg["id"]
            cam_type = cam_cfg.get("type", CAMERA_TYPE_STANDARD)
            intrinsics = cam_cfg.get("intrinsics", DEFAULT_INTRINSICS.copy())
            
            if cam_type == CAMERA_TYPE_OAKD:
                # If you have actual device_info, pass it here. 
                # This example passes None or a dummy index.
                # Adjust logic as needed to pick the correct OAK-D device.
                tracker = OAKDCameraTracker(
                    device_info=None,    # In real usage, you’d find matching device_info
                    index=cam_id,
                    intrinsics=intrinsics,
                    show=show_visualizations
                )
            else:
                tracker = SingleCameraTracker(
                    camera_id=cam_id,
                    intrinsics=intrinsics,
                    show=show_visualizations
                )
            self.camera_trackers.append(tracker)
        
        # Calibrator
        self.calibrator = PyTorchCameraCalibrator(self.num_cameras, self.primary_camera_idx)
        
        # Pose fuser
        self.pose_fuser = EnhancedHandPoseFuser(self.num_cameras, self.primary_camera_idx)
        
        # State & control
        self.state = "INITIALIZING"  # or CALIBRATING, TRACKING
        self.running = False
        self.calibration_start_time = None
        
        # Visualizer
        self.visualizer = HandVisualizer3D() if show_visualizations else None
    
    def start(self):
        # Start camera trackers
        for t in self.camera_trackers:
            t.start()
        
        self.running = True
        self.state = "INITIALIZING"
        print("Initializing - press SPACE to start calibration.")
        
        try:
            while self.running:
                self.update()
                self.display()
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    self.running = False
                elif key == 32:  # SPACE
                    if self.state == "INITIALIZING":
                        self.start_calibration()
                    elif self.state == "CALIBRATING" and self.calibrator.is_calibrated:
                        self.state = "TRACKING"
                        print("Calibration complete - now tracking")
        finally:
            self.stop()
    
    def start_calibration(self):
        self.state = "CALIBRATING"
        self.calibration_start_time = time.time()
        print("Starting calibration - move your hand around the views.")
    
    def update(self):
        observations = {}
        for idx, tracker in enumerate(self.camera_trackers):
            lm, conf = tracker.get_hand_landmarks()
            observations[idx] = (lm, conf)
        
        if self.state == "CALIBRATING":
            # Add samples
            for idx, (lm, conf) in observations.items():
                if lm is not None:
                    self.calibrator.add_sample(idx, lm, conf)
            # Check progress
            self.calibrator.check_calibration_progress()
            
            # Possibly auto-switch after some time if done
            if self.calibrator.is_calibrated:
                if (time.time() - self.calibration_start_time) > 5:
                    self.state = "TRACKING"
                    print("Calibration complete - now tracking.")
        
        elif self.state == "TRACKING":
            if self.calibrator.is_calibrated:
                camera_transforms = [
                    self.calibrator.get_transform_matrix(i) for i in range(self.num_cameras)
                ]
                fused_pose = self.pose_fuser.update(observations, camera_transforms)
                if fused_pose is not None and self.visualizer is not None:
                    # Get landmarks in fused world space
                    landmarks_3d = self.pose_fuser.get_hand_landmarks_from_pose(fused_pose)
                    self.visualizer.update(landmarks_3d, fused_pose)
    
    def display(self):
        if not self.show_visualizations:
            return
        
        # Gather processed frames
        frames = []
        for t in self.camera_trackers:
            if t.processed_frame is not None:
                frames.append(t.processed_frame)
        
        if not frames:
            return
        
        # Make sure all frames are same size
        h, w = frames[0].shape[:2]
        for i in range(1, len(frames)):
            frames[i] = cv2.resize(frames[i], (w, h))
        
        if len(frames) == 1:
            grid = frames[0]
        elif len(frames) == 2:
            grid = np.hstack(frames)
        else:
            # up to 4 in a 2x2
            top_row = np.hstack(frames[:2])
            if len(frames) > 2:
                bottom_row = frames[2]
                if len(frames) > 3:
                    bottom_row = np.hstack([bottom_row, frames[3]])
                else:
                    # add black placeholder
                    bottom_row = np.hstack([bottom_row, np.zeros_like(frames[0])])
                grid = np.vstack([top_row, bottom_row])
            else:
                grid = top_row
        
        status_text = f"State: {self.state}"
        if self.state == "CALIBRATING":
            progress = int(self.calibrator.calibration_progress * 100)
            status_text += f" ({progress}%)"
        cv2.putText(
            grid,
            status_text,
            (10, grid.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.imshow("Multi-Camera Hand Tracker", grid)
    
    def stop(self):
        for t in self.camera_trackers:
            t.stop()
        cv2.destroyAllWindows()
        if self.visualizer:
            self.visualizer.close()
        print("Shutting down.")