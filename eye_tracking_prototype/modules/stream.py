import os
import cv2
import threading
import time
from collections import deque
from .performance import FrameRate, timing_summary


def open_camera(device_idx, width, height, fps, backend="auto", fourcc=None, buffer_size=None):
    """Request a mode; property readback is not proof of delivered FPS."""
    backends = {"auto": cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY,
                "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
    cap = cv2.VideoCapture(device_idx, backends[backend])
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera {device_idx} with backend {backend}")
    requests = []
    if fourcc:
        requests.append(("fourcc", cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)))
    requests.extend((("width", cv2.CAP_PROP_FRAME_WIDTH, width),
                     ("height", cv2.CAP_PROP_FRAME_HEIGHT, height),
                     ("fps", cv2.CAP_PROP_FPS, fps)))
    if buffer_size is not None:
        requests.append(("buffer_size", cv2.CAP_PROP_BUFFERSIZE, buffer_size))
    try:
        settings = {name: {"requested": value, "set_succeeded": bool(cap.set(prop, value))}
                    for name, prop, value in requests}
        if fourcc and int(cap.get(cv2.CAP_PROP_FOURCC)) != cv2.VideoWriter_fourcc(*fourcc):
            # DirectShow can renegotiate the subtype during width/height/FPS sets.
            settings['fourcc']['reapply_succeeded'] = bool(
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)))
        for name, prop, _ in requests:
            settings[name]["readback"] = float(cap.get(prop))
            settings[name]["matches_request"] = abs(settings[name]['readback'] - settings[name]['requested']) < 0.01
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps_reported": float(cap.get(cv2.CAP_PROP_FPS)),
                "fourcc": "".join(chr((code >> (8 * i)) & 255) for i in range(4)),
                "buffer_size_reported": float(cap.get(cv2.CAP_PROP_BUFFERSIZE)),
                "exposure_reported": float(cap.get(cv2.CAP_PROP_EXPOSURE)),
                "auto_exposure_reported": float(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
                "backend": cap.getBackendName(), "settings": settings,
                "warnings": [f"{name}: request not honored by backend" for name, item in settings.items()
                             if not item['matches_request']]}
        return cap, info
    except Exception:
        cap.release()
        raise

class WebcamStream:
    def __init__(self, device_idx, width, height, fps, flip=True, **camera_options):
        self.cap, self.camera_info = open_camera(device_idx, width, height, fps, **camera_options)
        self.flip = flip
        self.ret, self.frame = False, None
            
        self.stopped = False
        self.lock = threading.Lock()
        self.frame_ready = threading.Condition(self.lock)
        self.capture_count = self.frame_id = 0
        self.capture_timestamp = 0.0
        self.capture_started = None
        self.rate = FrameRate()
        self.read_times = deque(maxlen=36000)
        self.intervals = deque(maxlen=36000)
        self.read_failures = 0
        self.thread = None
        self.error = None

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Capture stream already started")
        self.thread = threading.Thread(target=self.update, name="EyeCameraCapture", daemon=True)
        self.thread.start()
        return self

    def update(self):
        try:
            while not self.stopped:
                started = time.perf_counter()
                ret, frame = self.cap.read()
                captured = time.perf_counter()
                if not ret:
                    with self.lock:
                        self.read_failures += 1
                    time.sleep(0.005)
                    continue
                if self.flip:
                    frame = cv2.flip(frame, 1)
                with self.frame_ready:
                    if self.capture_count:
                        self.intervals.append((captured - self.capture_timestamp) * 1000.0)
                    else:
                        self.capture_started = captured
                    self.ret, self.frame = ret, frame
                    self.capture_count += 1
                    self.frame_id += 1
                    self.capture_timestamp = captured
                    self.read_times.append((captured - started) * 1000.0)
                    self.rate.add(captured)
                    self.frame_ready.notify_all()
        except Exception as exc:
            self.error = str(exc)
        finally:
            # Release from the owning thread, never concurrently with cap.read().
            self.cap.release()
            with self.frame_ready:
                self.stopped = True
                self.frame_ready.notify_all()

    def read(self):
        with self.lock:
            frame = self.frame
        return (True, frame.copy()) if frame is not None else (False, None)

    def read_latest(self, after_frame_id=0, timeout=0.1):
        """Wait for and return a frame newer than ``after_frame_id``."""
        with self.frame_ready:
            if self.frame_id <= after_frame_id and not self.stopped:
                self.frame_ready.wait_for(
                    lambda: self.frame_id > after_frame_id or self.stopped,
                    timeout=timeout,
                )
            if self.stopped or not self.ret or self.frame is None or self.frame_id <= after_frame_id:
                return False, None, after_frame_id, 0.0
            frame, frame_id, captured = self.frame, self.frame_id, self.capture_timestamp
        # Capture replaces published arrays; it never mutates them afterwards.
        return True, frame.copy(), frame_id, captured

    def diagnostics(self):
        with self.lock:
            span = self.capture_timestamp - (self.capture_started or self.capture_timestamp)
            result = dict(self.camera_info)
            result.update(capture_fps_observed=(self.capture_count - 1) / span if span > 0 else 0.0,
                          capture_fps_window=self.rate.fps(time.perf_counter()),
                          capture_count=self.capture_count, read_failures=self.read_failures,
                          error=self.error)
            reads, intervals = list(self.read_times), list(self.intervals)
        result["capture_read_ms"] = timing_summary(reads)
        result["capture_interval_ms"] = timing_summary(intervals)
        return result

    def observed_fps(self):
        with self.lock:
            return self.rate.fps(time.perf_counter())

    def stop(self):
        with self.frame_ready:
            self.stopped = True
            self.frame_ready.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                raise RuntimeError("Camera read did not stop within 2 seconds")
        else:
            self.cap.release()
