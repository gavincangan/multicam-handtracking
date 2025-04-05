##################################################################
#  OAK-D Camera Support
##################################################################

import time
import threading

from .constants import *
from .base_camera_tracker import BaseCameraTracker

try:
    import depthai as dai
    class OAKDCameraManager:
        """Helper class to manage OAK-D cameras."""

        def __init__(self):
            self.active_devices = []
            self.device_info_list = []
        
        def get_available_devices(self):
            """Get list of available OAK-D devices."""
            device_infos = dai.Device.getAllAvailableDevices()
            self.device_info_list = device_infos
            
            available_devices = []
            for i, device_info in enumerate(device_infos):
                available_devices.append({
                    "id": f"oakd_{i}",
                    "mxid": device_info.getMxId(),
                    "name": device_info.name,
                    "type": CAMERA_TYPE_OAKD
                })
            print(f"Found {len(available_devices)} OAK-D devices")
            return available_devices
        
        def create_pipeline(self, resolution=(1280, 720), fps=30):
            """Create a pipeline for RGB camera stream from OAK-D."""
            pipeline = dai.Pipeline()
            # Define source and output
            cam_rgb = pipeline.create(dai.node.ColorCamera)
            xout_rgb = pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            
            # Properties
            cam_rgb.setPreviewSize(resolution[0], resolution[1])
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setInterleaved(False)
            cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
            cam_rgb.setFps(fps)
            
            # Link
            cam_rgb.preview.link(xout_rgb.input)
            return pipeline
        
        def get_camera_intrinsics(self, device, resolution=(1280, 720)):
            """Get camera intrinsics from OAK-D device calibration."""
            try:
                calibration = device.readCalibration()
                intrinsic_matrix = calibration.getCameraIntrinsics(
                    dai.CameraBoardSocket.RGB,
                    dai.Size2f(*resolution)
                )
                intrinsics = {
                    "fx": intrinsic_matrix[0][0],
                    "fy": intrinsic_matrix[1][1],
                    "cx": intrinsic_matrix[0][2],
                    "cy": intrinsic_matrix[1][2],
                    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]
                }
                return intrinsics
            except Exception as e:
                print(f"Error getting OAK-D camera intrinsics: {e}")
                return OAKD_DEFAULT_INTRINSICS.copy()
    
    class OAKDCameraTracker(BaseCameraTracker):
        """Handles tracking for an OAK-D camera."""
        
        def __init__(self, device_info, index, intrinsics=None,
                     show=False, resolution=(1280, 720), fps=30):
            camera_id = f"oakd_{index}"
            super().__init__(camera_id, intrinsics, show)
            self.device_info = device_info
            self.resolution = resolution
            self.fps = fps
            self.oakd_manager = OAKDCameraManager()
            self.device = None
            self.queue = None
            
            self.camera_info = {
                "id": self.camera_id,
                "type": CAMERA_TYPE_OAKD,
                "name": device_info.name if device_info else "OAK-D",
                "mxid": device_info.getMxId() if device_info else "Unknown",
                "width": resolution[0],
                "height": resolution[1],
                "fps": fps
            }
            self.thread = threading.Thread(target=self.run_tracker)
        
        def start(self):
            # Create pipeline
            pipeline = self.oakd_manager.create_pipeline(self.resolution, self.fps)
            try:
                self.device = dai.Device(pipeline, self.device_info)
                self.queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                
                # Try to get camera intrinsics
                device_intrinsics = self.oakd_manager.get_camera_intrinsics(
                    self.device, self.resolution
                )
                if device_intrinsics:
                    self.intrinsics = device_intrinsics
                    print(f"Using OAK-D intrinsics for {self.camera_id}")
                
                self.running = True
                self.thread.start()
            except Exception as e:
                print(f"Failed to start OAK-D camera {self.camera_id}: {e}")
        
        def stop(self):
            self.running = False
            if self.thread.is_alive():
                self.thread.join()
            if self.device:
                self.device.close()
                self.device = None
        
        def run_tracker(self):
            while self.running:
                if self.queue is None:
                    time.sleep(0.1)
                    continue
                try:
                    in_rgb = self.queue.get()
                    frame = in_rgb.getCvFrame()
                    self.process_frame(frame)
                except Exception as e:
                    print(f"Error processing OAK-D frame: {e}")
                    time.sleep(0.1)

except ImportError:
    # If depthai is not installed, define dummies or skip OAK-D functionality
    print("DepthAI not installed, OAK-D support is disabled.")

    class OAKDCameraTracker(BaseCameraTracker):
        """Dummy OAK-D tracker if depthai is not available."""
        def __init__(self, *args, **kwargs):
            raise RuntimeError("OAK-D camera not supported because 'depthai' is not installed.")