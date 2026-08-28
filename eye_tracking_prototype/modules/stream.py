import os
import cv2
import threading
import time

class WebcamStream:
    def __init__(self, device_idx, width, height, fps, flip=True):
        self.cap = cv2.VideoCapture(device_idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.flip = flip
        
        self.ret, self.frame = self.cap.read()
        if self.ret and self.flip:
            self.frame = cv2.flip(self.frame, 1)
            
        self.stopped = False
        self.lock = threading.Lock()
        self.capture_count = 1 if self.ret else 0
        self.capture_started = time.perf_counter()

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            if self.flip:
                frame = cv2.flip(frame, 1)
            with self.lock:
                self.ret = ret
                self.frame = frame
                self.capture_count += 1

    def read(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def diagnostics(self):
        elapsed = max(time.perf_counter() - self.capture_started, 1e-6)
        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps_reported": float(self.cap.get(cv2.CAP_PROP_FPS)),
            "capture_fps_observed": self.capture_count / elapsed,
            "backend": self.cap.getBackendName() if self.cap.isOpened() else "closed",
        }

    def observed_fps(self):
        elapsed = max(time.perf_counter() - self.capture_started, 1e-6)
        return self.capture_count / elapsed

    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        self.cap.release()
