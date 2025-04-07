# Straightforward Auto-calibration and Hand Tracking for a Multi-camera Setup

## Overview

A real-time 3D hand tracking system that utilizes multiple cameras to provide accurate hand pose estimation in three-dimensional space. The system features automatic camera calibration, support for both standard USB cameras and OAK-D devices, and real-time 3D visualization.

### Key Features
- Multi-camera hand tracking with automatic calibration
- Support for both standard USB cameras and OAK-D devices
- Real-time 3D visualization of hand poses
- GPU-accelerated tracking (CUDA support)
- Particle filter-based pose estimation
- Easy-to-use configuration system

## System Requirements

### Software Dependencies
- Python 3.x
- OpenCV (with optional CUDA support)
- MediaPipe
- PyTorch
- NumPy
- Matplotlib
- DepthAI (optional, for OAK-D camera support)
- SciPy

### Hardware Requirements
- One or more USB cameras (up to 3 supported)
- Optional: OAK-D camera(s)
- Optional: NVIDIA GPU for CUDA acceleration

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone git@github.com:gavincangan/multicam-handtracking.git
   cd multicam-handtracking
   ```

2. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe torch numpy matplotlib scipy
   ```

3. For OAK-D camera support (optional):
   ```bash
   pip install depthai
   ```

### Camera Configuration

The system uses a `camera_config.yaml` file to manage camera settings. Example configuration:

```yaml
cameras:
  - id: 0
    type: "standard"
    primary: true
    intrinsics:
      fx: 1000.0
      fy: 1000.0
      cx: 960.0
      cy: 540.0
      distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
  - id: 1
    type: "oakd"
    primary: false
    intrinsics:
      fx: 860.0
      fy: 860.0
      cx: 640.0
      cy: 360.0
      distortion: [0.0, 0.0, 0.0, 0.0, 0.0]
```

## Usage

### Command-line Arguments
- `--reset-config`: Reset the camera configuration to default values
- `--no-viz`: Disable 3D visualization
- `--config`: Specify a custom config file path (default: ./src/config/camera_config.yaml)

### Basic Workflow

1. **Camera Detection**:
   - Run `list_cameras.py` to detect available cameras:
     ```bash
     python list_cameras.py
     ```

2. **Start the System**:
   ```bash
   python main.py
   ```

3. **Calibration Process**:
   - Position a single hand visible to at least the primary camera
   - Press SPACE to start calibration
   - Move your hand slowly to cover the field of view
   - The system will automatically track your hand in 3D once calibrated

### Controls
- **SPACE**: Start calibration / Confirm calibration completion
- **ESC**: Exit the program

## Project Structure

```
.
├── main.py                 # Main entry point and program initialization
├── list_cameras.py         # Utility for detecting and listing available cameras
├── camera_config.yaml      # Camera configuration file
└── src/
    ├── base_camera_tracker.py    # Base class for camera tracking
    ├── hand_tracker.py           # PyTorch hand model implementation
    ├── hand_pose_fuser.py        # Multi-view pose fusion logic
    ├── hand_visualizer.py        # 3D visualization using Matplotlib
    ├── multi_camera_tracker.py   # Main multi-camera tracking implementation
    ├── camera_calibrator.py      # Auto-calibration system
    ├── oakd_handler.py           # OAK-D camera support
    ├── single_camera_tracker.py  # Single camera tracking implementation
    ├── gpu_particle_filter.py    # GPU-accelerated particle filter
    ├── constants.py              # System constants and configurations
    └── camera_config.py          # Configuration management
```

## Technical Details

### Hand Tracking Model
The system uses a combination of MediaPipe for 2D hand detection and a custom PyTorch-based hand model for 3D pose estimation. The hand model consists of 21 keypoints representing the hand's skeletal structure.

### Calibration Methodology
The auto-calibration system uses a PyTorch-based optimization approach to determine the relative poses of all cameras. The system requires a minimum of 100 samples and can use up to 500 samples for improved accuracy.

### 3D Reconstruction
- Multi-view fusion using particle filter optimization
- Real-time 3D pose estimation from multiple 2D observations
- Smooth tracking through GPU-accelerated particle filtering

### Performance Considerations
- CUDA acceleration support for improved performance on NVIDIA GPUs
- Efficient particle filter implementation for real-time tracking
- Configurable model complexity for different performance requirements
- Optional visualization can be disabled for maximum performance

### Camera Support
- Support for up to 3 simultaneous cameras
- Compatible with standard USB cameras and OAK-D devices
- Auto-detection of camera capabilities and properties
- Flexible configuration system for camera parameters
