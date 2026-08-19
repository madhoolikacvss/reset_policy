#!/usr/bin/env python3
"""
Minimal CubeTracker test - using same camera config as tracker.
"""

import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from cameras import CameraPair, CameraConfig
from april_tag.track_apriltag import AprilTagTracker
from cube_tracker import CubeTracker


# Configuration
CAMERA_INTRINSICS = str(
    Path(__file__).resolve().parent.parent.parent
    / "configs/camera_intrinsics.yaml"
)

TAG_SIZE = 0.04
WORLD_TAG_ID = 0
CUBE_TAG_ID = 1


def main():
    # Use the SAME camera config as your apriltag_tracker
    # Try different device IDs based on your system
    camera_configs_to_try = [
        CameraConfig(external_device="/dev/video0", wrist_device="/dev/video10"),
        CameraConfig(external_device="/dev/video2", wrist_device="/dev/video10"),
        CameraConfig(external_device="/dev/video4", wrist_device="/dev/video10"),
        CameraConfig(external_device="/dev/video6", wrist_device="/dev/video10"),
        CameraConfig(external_device="/dev/video8", wrist_device="/dev/video10"),
    ]
    
    for config in camera_configs_to_try:
        print(f"Trying external camera: {config.external_device}")
        try:
            with CameraPair(config) as cameras:
                # Try to read one frame to test
                try:
                    external, _ = cameras.read(timeout=2.0)
                    print(f"✅ Camera working at {config.external_device}!")
                    # If we get here, camera works. Now run the tracker.
                    run_tracker(cameras)
                    break
                except Exception as e:
                    print(f"❌ Camera {config.external_device} failed: {e}")
                    continue
        except Exception as e:
            print(f"❌ Failed to open {config.external_device}: {e}")
            continue
    else:
        print("❌ No working camera found!")


def run_tracker(cameras):
    """Run the tracker with working camera."""
    tag_tracker = AprilTagTracker(
        yaml_file=CAMERA_INTRINSICS,
        tag_size=TAG_SIZE,
        world_tag_id=WORLD_TAG_ID,
        cube_tag_id=CUBE_TAG_ID,
    )
    cube_tracker = CubeTracker(camera=cameras, tracker=tag_tracker)
    
    print("\n" + "=" * 50)
    print(f"{'Frame':>6} {'x (m)':>10} {'y (m)':>10} {'Status':>12}")
    print("-" * 50)
    
    frame = 0
    try:
        while True:
            frame += 1
            state = cube_tracker.get_state()
            
            if state.detected:
                print(f"{frame:6d} {state.x:10.4f} {state.y:10.4f} {'DETECTED':>12}")
            else:
                print(f"{frame:6d} {'--':>10} {'--':>10} {'NOT FOUND':>12}")
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()