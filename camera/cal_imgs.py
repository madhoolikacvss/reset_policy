# capture_calibration_images.py

from pathlib import Path
import cv2
import time

from reset_policy.camera.cameras import CameraPair

SAVE_DIR = Path("calibration_images")
SAVE_DIR.mkdir(exist_ok=True)

NUM_IMAGES = 30

print()
print("Press SPACE to save an image.")
print("Press ESC to quit.")

count = 0

with CameraPair() as cameras:

    while count < NUM_IMAGES:

        external, _ = cameras.read()

        bgr = cv2.cvtColor(external, cv2.COLOR_RGB2BGR)

        display = bgr.copy()

        cv2.putText(
            display,
            f"{count}/{NUM_IMAGES}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2,
        )

        cv2.imshow("Calibration", display)

        key = cv2.waitKey(1)

        if key == 27:
            break

        if key == ord(" "):

            filename = SAVE_DIR / f"image_{count:02d}.png"

            cv2.imwrite(str(filename), bgr)

            print(f"Saved {filename}")

            count += 1

            time.sleep(0.4)

cv2.destroyAllWindows()