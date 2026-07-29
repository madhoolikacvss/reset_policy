import cv2

from cameras import CameraPair
from track_apriltag import AprilTagTracker


# SETTINGS
TAG_SIZE = 0.05      # 50 mm

tracker = AprilTagTracker(
    yaml_file="camera_intrinsics.yaml",
    tag_size=TAG_SIZE,
    world_tag_id=0,
    cube_tag_id=1,
)

with CameraPair() as cameras:

    while True:

        external, _ = cameras.read()

        pose, vis = tracker.process(external)

        if pose is not None:

            x, y, z, roll, pitch, yaw = pose

            print(
                f"x={x:.3f} m  y={y:.3f} m  z={z:.3f} m  yaw={yaw:.1f} deg"
            )

        cv2.imshow("Cube Tracker", vis)

        key = cv2.waitKey(1)

        if key == 27:
            break

cv2.destroyAllWindows()