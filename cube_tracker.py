"""
- Read external camera
- Run AprilTag tracker
- Convert AprilTag board coordinates
- Return cube state in fixed board coordinates
"""


from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional


from camera.cameras import CameraPair
from april_tag.track_apriltag import AprilTagTracker

@dataclass
class CubeState:

    # Cube position in board coordinate frame (cm)
    x: float
    y: float

    # Cube orientation
    yaw: float
    detected: bool
    timestamp: float

    # Raw AprilTag pose
    raw_pose: tuple | None = None

class CubeTracker:


    def __init__(self,camera: CameraPair,tracker: AprilTagTracker,):
        self.camera = camera
        self.tracker = tracker



    def _read_pose(self) -> Optional[tuple]:

        """
        Capture one frame and estimate cube pose.

        Returns:
            x,y,z,roll,pitch,yaw
            in meters/radians
        """

        external, _ = self.camera.read()
        pose, _ = self.tracker.process(external)
        return pose



    def get_state(self) -> CubeState:
        pose = self._read_pose()
        if pose is None:

            return CubeState(

                x=0.0,
                y=0.0,
                yaw=0.0,
                detected=False,
                timestamp=time.time(),
                raw_pose=None,
            )

        # Convert meters -> cm
        x, y, z, roll, pitch, yaw = pose


        return CubeState(

            x=x,
            y=y,
            yaw=yaw,
            detected=True,
            timestamp=time.time(),
            raw_pose=pose,
        )



    def get_position(self):

        """
        Convenience function.

        Returns:
            (x,y) in cm
        """

        state = self.get_state()
        if not state.detected:
            return None
        return (state.x,state.y,)