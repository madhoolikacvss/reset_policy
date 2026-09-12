from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path

import cv2
import numpy as np


@dataclasses.dataclass(frozen=True)
class CameraConfig:
    external_serial: str = "144223023540"
    # wrist_device: str = "/dev/video10"
    width: int = 1280
    height: int = 720
    fps: int = 15


class CameraError(RuntimeError):
    pass

import subprocess
import glob


def find_camera_by_serial(serial: str, prefer_color: bool = True) -> str | None:
    """
    Find /dev/videoX for a camera with the given USB serial using v4l2-ctl.
    
    Args:
        serial: USB serial number string (e.g. "144223023540")
        prefer_color: If True, prefer devices that support YUYV/MJPG (color stream).
    
    Returns:
        Device path (e.g. "/dev/video0") or None if not found.
    """
    candidates = []
    
    for device in sorted(glob.glob("/dev/video*")):
        try:
            # Get serial number
            info = subprocess.run(
                ["v4l2-ctl", "-d", device, "--info"],
                capture_output=True, text=True, timeout=2.0,
            )
            dev_serial = None
            for line in info.stdout.splitlines():
                if line.strip().startswith("Serial"):
                    dev_serial = line.split(":", 1)[1].strip()
                    break
            
            if dev_serial != serial:
                continue
            
            # Check pixel format
            is_color = False
            if prefer_color:
                fmt = subprocess.run(
                    ["v4l2-ctl", "-d", device, "--get-fmt-video"],
                    capture_output=True, text=True, timeout=2.0,
                )
                if "YUYV" in fmt.stdout or "MJPG" in fmt.stdout:
                    is_color = True
            
            candidates.append((device, is_color))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    if not candidates:
        return None
    
    # Prefer color devices
    if prefer_color:
        for device, is_color in candidates:
            if is_color:
                return device
    
    # Fall back to first candidate
    return candidates[0][0]

class CameraPair:
    """Continuously grabs frames on background threads so ``read()`` is instant.

    A dedicated thread per camera keeps pulling frames (which flushes the V4L2
    buffer, so the stored frame is always the freshest one). ``read()`` just
    returns the latest cached RGB frames without blocking on capture I/O, which
    keeps it off the real-time control loop.
    """

    def __init__(self, config: CameraConfig = CameraConfig()) -> None:
        self._config = config

        #get path of external camera
        external_path = find_camera_by_serial(config.external_serial)
        if external_path is None:
            raise CameraError("cant find external camera with given serial number")
        self._external = self._open(external_path)
        # self._wrist = self._open(config.wrist_device)
        self._lock = threading.Lock()
        self._frames: dict[str, np.ndarray | None] = {
            "external": None,
            # "wrist": None,
        }
        self._running = True
        self._threads = [
            threading.Thread(
                target=self._grab_loop,
                args=(self._external, "external"),
                daemon=True,
            ),
            # threading.Thread(
            #     target=self._grab_loop,
            #     args=(self._wrist, "wrist"),
            #     daemon=True,
            # ),
        ]
        for thread in self._threads:
            thread.start()

    def _open(self, device: str) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        capture.set(cv2.CAP_PROP_FPS, self._config.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"Could not open camera {device}")
        return capture

    def _grab_loop(self, capture: cv2.VideoCapture, key: str) -> None:
        while self._running:
            ok, bgr = capture.read()
            if not ok:
                time.sleep(0.001)
                continue
            # Reject black/garbage frames (e.g. during sensor warmup).
            if float(bgr.std()) < 5.0:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._frames[key] = rgb

    def read(self, timeout: float = 5.0) -> np.ndarray:
        """
        Return the latest external RGB frame.

        Waits until a frame is available or the timeout expires.
        """
        deadline = time.monotonic() + timeout
        while True:
            time.sleep(0.2)
            with self._lock:
                external = self._frames["external"]
                # wrist = self._frames["wrist"]
            if external is not None:
                # Original dual-camera return:
                # if external is not None and wrist is not None:
                #     wrist = np.ascontiguousarray(np.rot90(wrist, 2))
                #     return external, wrist
                return external, None
            if time.monotonic() > deadline:
                raise CameraError("Timed out waiting for camera frames")

    def close(self) -> None:
        self._running = False
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._external.release()
        # self._wrist.release()

    def __enter__(self) -> CameraPair:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise CameraError(f"Could not save image to {path}")
