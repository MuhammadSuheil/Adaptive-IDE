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

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if self.flip:
                frame = cv2.flip(frame, 1)
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else (False, None)

    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        self.cap.release()