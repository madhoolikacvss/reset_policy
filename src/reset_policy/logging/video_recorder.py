"""
Video recorder for training episodes.

Records video using a secondary camera (identified by serial number).
Called from train.py at the start/end of each episode.
"""

from __future__ import annotations

import time
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from reset_policy.perception.cameras import (
    CameraConfig,
    CameraPair,
    find_camera_by_serial,
    CameraError,
)


# ============================================================
# Video recorder configuration (kept here, not in config.py)
# ============================================================

RECORD_VIDEO = True
VIDEO_CAMERA_SERIAL = "310643060553"
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 5

# Where to save videos (relative to project logs folder)
VIDEO_SUBDIR = "/home/madhoolika/workspace/reset_policy/src/logs/videos"  # -> logs/videos/episode_XXXX.mp4


# ============================================================
# Episode video recorder
# ============================================================

class EpisodeVideoRecorder:
    """Records one episode's video to logs/videos/episode_XXXX.mp4."""

    def __init__(
        self,
        serial: str = VIDEO_CAMERA_SERIAL,
        save_dir: Path | str | None = None,
        width: int = VIDEO_WIDTH,
        height: int = VIDEO_HEIGHT,
        fps: int = VIDEO_FPS,
    ):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps

        # Default save dir: logs/videos
        if save_dir is None:
            logs_dir = Path(__file__).resolve().parent.parent.parent / "logs"
            save_dir = logs_dir / VIDEO_SUBDIR
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Resolve camera device
        self.config = CameraConfig(
            external_serial=serial,            
            width=width,
            height=height,
            fps=fps,
        )
        self.camera = CameraPair(config=self.config)

        # Video writer state
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = None
        self.frame_interval = 1.0 / fps
        self.last_write_time = 0.0

    def start_episode(self, episode_num: int):
        """Begin a new video file for the given episode."""
        self.close_writer()

        output_path = self.save_dir / f"episode_{episode_num:04d}.mp4"
        self.writer = cv2.VideoWriter(
            str(output_path),
            self.fourcc,
            self.fps,
            (self.width, self.height),
        )
        if not self.writer.isOpened():
            raise CameraError(f"Failed to open VideoWriter for {output_path}")

        print(f"[VIDEO] Recording episode {episode_num} -> {output_path}")

    def capture_frame(self):
        """Capture and write one frame (rate-limited to target fps)."""
        if self.writer is None:
            return

        now = time.monotonic()
        if now - self.last_write_time < self.frame_interval:
            return  # skip to maintain target fps

        try:
            rgb_frame,_ = self.camera.read()
        except Exception as e:
            print(f"[VIDEO] Frame read failed: {e}")
            return

        if rgb_frame is None:
            return

        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr_frame)
        self.last_write_time = now

    def end_episode(self):
        """Finalize the current episode's video."""
        self.close_writer()

    def close_writer(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def close(self):
        """Release all resources."""
        self.close_writer()
        try:
            self.camera.close()
        except Exception:
            pass