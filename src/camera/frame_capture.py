"""
Threaded video capture stream with auto-reconnection and FPS profiling.
"""

import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from src.utils.logging_config import setup_logger

logger = setup_logger("frame_capture")


class CameraStream:
    """
    Asynchronous, non-blocking camera frame capture worker thread.
    Prevents OpenCV frame capture from blocking the main perception and WebSocket loops.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 30,
        auto_reconnect: bool = True,
        reconnect_delay_sec: float = 1.0,
    ):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay_sec = reconnect_delay_sec

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.frame_timestamp: float = 0.0
        self.frame_counter: int = 0
        self.running: bool = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        # Diagnostic Stats
        self.actual_fps: float = 0.0
        self._last_fps_calc_time = time.time()
        self._fps_frame_count = 0

    def start(self) -> "CameraStream":
        """Initializes the capture device and starts the background worker thread."""
        self._open_device()
        self.running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True, name="CameraWorker")
        self.thread.start()
        logger.info(f"Camera capture stream started on device index {self.device_index} ({self.width}x{self.height}).")
        return self

    def _open_device(self) -> bool:
        """Attempts to open the OpenCV VideoCapture device."""
        if self.cap is not None:
            self.cap.release()

        logger.info(f"Connecting to camera device {self.device_index}...")
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            logger.warning(f"Failed to open video capture device {self.device_index}.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

        # Attempt to enable hardware autofocus if supported by camera
        try:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        except Exception:
            pass

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera opened: requested {self.width}x{self.height}, actual {actual_w}x{actual_h}")
        return True

    def _capture_worker(self):
        """Worker loop continuously querying latest frames."""
        frame_delay = 1.0 / max(self.target_fps, 1)

        while self.running:
            start_time = time.time()

            if self.cap is None or not self.cap.isOpened():
                if self.auto_reconnect and self.running:
                    time.sleep(self.reconnect_delay_sec)
                    self._open_device()
                else:
                    time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            now = time.time()

            if not ret or frame is None:
                logger.warning("Frame read failed or camera disconnected.")
                if self.auto_reconnect and self.running:
                    time.sleep(self.reconnect_delay_sec)
                    self._open_device()
                continue

            # Mirror frame horizontally for natural, intuitive front-facing interaction
            frame = cv2.flip(frame, 1)

            # Thread-safe frame swap
            with self.lock:
                self.frame = frame
                self.frame_timestamp = now
                self.frame_counter += 1

            # Update FPS metric
            self._fps_frame_count += 1
            elapsed = now - self._last_fps_calc_time
            if elapsed >= 1.0:
                self.actual_fps = self._fps_frame_count / elapsed
                self._fps_frame_count = 0
                self._last_fps_calc_time = now

            # Throttle to target FPS
            work_duration = time.time() - start_time
            sleep_time = max(0.0, frame_delay - work_duration)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Reads the most recent frame in a non-blocking thread-safe manner.
        Returns: (success_flag, frame_bgr, timestamp)
        """
        with self.lock:
            if self.frame is None:
                return False, None, 0.0
            return True, self.frame.copy(), self.frame_timestamp

    def stop(self):
        """Stops the worker thread and releases the hardware camera device."""
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("Camera capture stream stopped.")

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
