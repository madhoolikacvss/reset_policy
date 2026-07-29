from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from pupil_apriltags import Detector


class AprilTagTracker:
    """Track a moving AprilTag relative to a fixed world AprilTag."""

    def __init__(
        self,
        yaml_file: str,
        tag_size: float,
        world_tag_id: int = 1,
        cube_tag_id: int = 0,
    ) -> None:

        self.world_tag_id = world_tag_id
        self.cube_tag_id = cube_tag_id
        self.tag_size = tag_size

        # Load camera intrinsics
        fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)

        self.K = fs.getNode("camera_matrix").mat()
        self.dist = fs.getNode("distortion_coefficients").mat()

        fs.release()

        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

        self.detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
        )

    def _to_transform(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Convert rotation + translation to 4x4 homogeneous transform."""
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        return T

    def _pose_to_xyzrpy(self, T: np.ndarray) -> tuple[float, float, float, float, float, float]:

        x, y, z = T[:3, 3]

        rot = Rotation.from_matrix(T[:3, :3])

        roll, pitch, yaw = rot.as_euler("xyz", degrees=True)

        return x, y, z, roll, pitch, yaw

    def process(self, rgb: np.ndarray):

        # Convert RGB -> BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Undistort
        bgr = cv2.undistort(bgr, self.K, self.dist)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        tags = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(self.fx, self.fy, self.cx, self.cy),
            tag_size=self.tag_size,
        )

        detections = {}

        for tag in tags:

            T = self._to_transform(tag.pose_R, tag.pose_t)

            detections[tag.tag_id] = {
                "transform": T,
                "corners": tag.corners,
                "center": tag.center,
            }

            # Draw tag outline
            corners = tag.corners.astype(int)

            for i in range(4):
                cv2.line(
                    bgr,
                    tuple(corners[i]),
                    tuple(corners[(i + 1) % 4]),
                    (0, 255, 0),
                    2,
                )

            center = tuple(tag.center.astype(int))

            cv2.circle(bgr, center, 5, (0, 0, 255), -1)

            cv2.putText(
                bgr,
                f"ID {tag.tag_id}",
                (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        # Need both tags
        if (
            self.world_tag_id not in detections
            or self.cube_tag_id not in detections
        ):
            return None, bgr

        T_camera_world = detections[self.world_tag_id]["transform"]
        T_camera_cube = detections[self.cube_tag_id]["transform"]

        # Cube in world coordinates
        T_world_cube = np.linalg.inv(T_camera_world) @ T_camera_cube

        pose = self._pose_to_xyzrpy(T_world_cube)

        x, y, z, roll, pitch, yaw = pose

        cv2.putText(
            bgr,
            f"x={x:.3f} y={y:.3f} yaw={yaw:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        return pose, bgr