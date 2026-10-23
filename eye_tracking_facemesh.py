import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import datetime
import time
import math
import os
import threading
import queue
import urllib.request
#from modules.stream import WebcamStreamWithQueue, WFrame
from typing import NamedTuple

class IrisGazeTracker:
    def __init__(self, max_faces=1, save_dir="dataset_eyes", queue_size=1200):
        # 1. Auto-download model if missing
        self.model_path = 'models/face_landmarker.task'
        if not os.path.exists(self.model_path):
            print("Downloading Face Landmarker model (only happens once)...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete.")

        # 2. Setup Tasks API
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=max_faces
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # 3. Landmark Indices
        self.L_IRIS_CENTER = 468
        self.L_IRIS_EDGE = 469
        self.L_EYE_INNER = 362
        self.L_EYE_OUTER = 263
        self.L_EYE_TOP = 159
        self.L_EYE_BOTTOM = 145
        self.L_EAR_INDICES = [362, 385, 387, 263, 373, 380]
        
        self.R_IRIS_CENTER = 473
        self.R_IRIS_EDGE = 474
        self.R_EYE_INNER = 133
        self.R_EYE_OUTER = 33
        self.R_EYE_TOP = 386
        self.R_EYE_BOTTOM = 374
        self.R_EAR_INDICES = [33, 160, 158, 133, 153, 144]
        
        # 4. Disk Saving & Queue Setup
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.csv_path = os.path.join(self.save_dir, "iris_data.csv")
        
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w") as f:
                f.write("timestamp,fps,frame_id,left_radius,right_radius,left_radius_norm,right_radius_norm,left_gaze_h,right_gaze_h,left_gaze_v,right_gaze_v,left_ear,right_ear\n")
                
        self.disk_queue = queue.Queue(maxsize=queue_size)
        self.worker_thread = threading.Thread(target=self._disk_writer_worker, daemon=True)
        self.worker_thread.start()

    def _disk_writer_worker(self):
        """Background thread to handle saving without lagging the video."""
        while True:
            task = self.disk_queue.get()
            if task is None: 
                break
                
            if task["type"] == "image":
                cv2.imwrite(task["path"], task["data"])
            elif task["type"] == "csv":
                with open(self.csv_path, "a") as f:
                    f.write(task["data"])
                    
            self.disk_queue.task_done()

    def _calculate_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _calculate_gaze_ratio(self, iris_center, p1, p2):
        d1 = self._calculate_distance(iris_center, p1)
        total_d = self._calculate_distance(p1, p2)
        return d1 / total_d if total_d > 0 else 0.5

    def _calculate_ear(self, landmarks, img_w, img_h, indices):
        pts = [(int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)) for idx in indices]
        v1 = self._calculate_distance(pts[1], pts[5])
        v2 = self._calculate_distance(pts[2], pts[4])
        h = self._calculate_distance(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

    def _extract_iris_metrics(self, lm, img_w, img_h):
        lc = (int(lm[self.L_IRIS_CENTER].x * img_w), int(lm[self.L_IRIS_CENTER].y * img_h))
        le = (int(lm[self.L_IRIS_EDGE].x * img_w), int(lm[self.L_IRIS_EDGE].y * img_h))
        rc = (int(lm[self.R_IRIS_CENTER].x * img_w), int(lm[self.R_IRIS_CENTER].y * img_h))
        re = (int(lm[self.R_IRIS_EDGE].x * img_w), int(lm[self.R_IRIS_EDGE].y * img_h))
        
        return {
            "left_center": lc, "left_radius": self._calculate_distance(lc, le),
            "right_center": rc, "right_radius": self._calculate_distance(rc, re)
        }

    def _extract_gaze_metrics(self, lm, img_w, img_h, l_center, r_center):
        l_in = (int(lm[self.L_EYE_INNER].x * img_w), int(lm[self.L_EYE_INNER].y * img_h))
        l_out = (int(lm[self.L_EYE_OUTER].x * img_w), int(lm[self.L_EYE_OUTER].y * img_h))
        l_top = (int(lm[self.L_EYE_TOP].x * img_w), int(lm[self.L_EYE_TOP].y * img_h))
        l_bot = (int(lm[self.L_EYE_BOTTOM].x * img_w), int(lm[self.L_EYE_BOTTOM].y * img_h))
        
        r_in = (int(lm[self.R_EYE_INNER].x * img_w), int(lm[self.R_EYE_INNER].y * img_h))
        r_out = (int(lm[self.R_EYE_OUTER].x * img_w), int(lm[self.R_EYE_OUTER].y * img_h))
        r_top = (int(lm[self.R_EYE_TOP].x * img_w), int(lm[self.R_EYE_TOP].y * img_h))
        r_bot = (int(lm[self.R_EYE_BOTTOM].x * img_w), int(lm[self.R_EYE_BOTTOM].y * img_h))
        
        return {
            "left_gaze_h": self._calculate_gaze_ratio(l_center, l_in, l_out),
            "right_gaze_h": self._calculate_gaze_ratio(r_center, r_in, r_out),
            "left_gaze_v": self._calculate_gaze_ratio(l_center, l_top, l_bot),
            "right_gaze_v": self._calculate_gaze_ratio(r_center, r_top, r_bot)
        }

    def process_frameOld(self, image_bgr):
        img_h, img_w, _ = image_bgr.shape
        # print("process frame image size: ", img_w, "x", img_h)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect(mp_image)
        
        if not results.face_landmarks:
            return None
            
        lm = results.face_landmarks[0]
        
        iris_metrics = self._extract_iris_metrics(lm, img_w, img_h)
        gaze_metrics = self._extract_gaze_metrics(lm, img_w, img_h, iris_metrics["left_center"], iris_metrics["right_center"])
        ear_metrics = {
            "left_ear": self._calculate_ear(lm, img_w, img_h, self.L_EAR_INDICES),
            "right_ear": self._calculate_ear(lm, img_w, img_h, self.R_EAR_INDICES)
        }
        
        return {**iris_metrics, **gaze_metrics, **ear_metrics}
        
    def process_frame(self, image_bgr):
        img_h, img_w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = self.detector.detect(mp_image)
        
        if not results.face_landmarks:
            return None
            
        lm = results.face_landmarks[0]
        
        iris_metrics = self._extract_iris_metrics(lm, img_w, img_h)
        gaze_metrics = self._extract_gaze_metrics(lm, img_w, img_h, iris_metrics["left_center"], iris_metrics["right_center"])
        ear_metrics = {
            "left_ear": self._calculate_ear(lm, img_w, img_h, self.L_EAR_INDICES),
            "right_ear": self._calculate_ear(lm, img_w, img_h, self.R_EAR_INDICES)
        }
        
        # --- NEW: Normalization Math ---
        # 1. Get coordinates for the inner corners of both eyes
        l_inner = (int(lm[self.L_EYE_INNER].x * img_w), int(lm[self.L_EYE_INNER].y * img_h))
        r_inner = (int(lm[self.R_EYE_INNER].x * img_w), int(lm[self.R_EYE_INNER].y * img_h))
        
        # 2. Calculate the Intercanthal Distance (distance between the eyes)
        intercanthal_dist = self._calculate_distance(l_inner, r_inner)
        
        # 3. Normalize the radii against this distance
        norm_metrics = {
            "intercanthal_dist": intercanthal_dist,
            "left_radius_norm": iris_metrics["left_radius"] / intercanthal_dist if intercanthal_dist > 0 else 0,
            "right_radius_norm": iris_metrics["right_radius"] / intercanthal_dist if intercanthal_dist > 0 else 0
        }
        
        return {**iris_metrics, **gaze_metrics, **ear_metrics, **norm_metrics}

    def save_data(self, image_bgr, iris_data, frame_id, timestamp, fps, padding_multiplier=3.0):
        if not iris_data: return
            
        csv_row = (f"{timestamp},{fps:.0f},"
                   f"{frame_id},{iris_data['left_radius']:.2f},{iris_data['right_radius']:.2f},"
                   f"{iris_data['left_radius_norm']:.4f},{iris_data['right_radius_norm']:.4f},"
                   f"{iris_data['left_gaze_h']:.3f},{iris_data['right_gaze_h']:.3f},"
                   f"{iris_data['left_gaze_v']:.3f},{iris_data['right_gaze_v']:.3f},"
                   f"{iris_data['left_ear']:.3f},{iris_data['right_ear']:.3f}\n")
        
        if not self.disk_queue.full():
            self.disk_queue.put_nowait({"type": "csv", "data": csv_row})
            
            img_h, img_w, _ = image_bgr.shape
            lx, ly = iris_data["left_center"]
            rx, ry = iris_data["right_center"]
            l_pad = int(iris_data["left_radius"] * padding_multiplier)
            r_pad = int(iris_data["right_radius"] * padding_multiplier)
            
            l_y1, l_y2 = max(0, ly - l_pad), min(img_h, ly + l_pad)
            l_x1, l_x2 = max(0, lx - l_pad), min(img_w, lx + l_pad)
            r_y1, r_y2 = max(0, ry - r_pad), min(img_h, ry + r_pad)
            r_x1, r_x2 = max(0, rx - r_pad), min(img_w, rx + r_pad)
            
            l_crop = image_bgr[l_y1:l_y2, l_x1:l_x2]
            r_crop = image_bgr[r_y1:r_y2, r_x1:r_x2]
            
            if l_crop.size > 0:
                self.disk_queue.put_nowait({"type": "image", "path": os.path.join(self.save_dir, f"frame_{frame_id:04d}_left.jpg"), "data": l_crop})
            if r_crop.size > 0:
                self.disk_queue.put_nowait({"type": "image", "path": os.path.join(self.save_dir, f"frame_{frame_id:04d}_right.jpg"), "data": r_crop})

    def draw_overlay(self, image_bgr, iris_data):
        if not iris_data: return image_bgr
        
        lc, rc = iris_data["left_center"], iris_data["right_center"]
        lr, rr = int(iris_data["left_radius"]), int(iris_data["right_radius"])
        lg_h, lg_v = iris_data["left_gaze_h"], iris_data["left_gaze_v"]
        
        cv2.circle(image_bgr, lc, 2, (0, 255, 0), -1)
        cv2.circle(image_bgr, rc, 2, (0, 255, 0), -1)
        cv2.circle(image_bgr, lc, lr, (255, 0, 255), 1)
        cv2.circle(image_bgr, rc, rr, (255, 0, 255), 1)
        
        dir_h = "Center"
        if lg_h < 0.4: dir_h = "Right" 
        elif lg_h > 0.6: dir_h = "Left"
            
        dir_v = "Center"
        if lg_v < 0.4: dir_v = "Up"
        elif lg_v > 0.6: dir_v = "Down"
            
        cv2.putText(image_bgr, f"Gaze: {dir_v}-{dir_h} (H: {lg_h:.2f} V: {lg_v:.2f})", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image_bgr, f"EAR (Blink): L: {iris_data['left_ear']:.2f} R: {iris_data['right_ear']:.2f}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        return image_bgr

    def close(self):
        self.detector.close()
        self.disk_queue.put(None)
        self.worker_thread.join()

# data only class, using NamedTuple
# it is immutable! set values only from initialization
class WFrame(NamedTuple):
    frame: Any
    id: int
    timestamp: float
    fps: float

# each frame is saved in a Queue
# frame from cap.read() is only considered if hardware time already changed
class WebcamStreamWithQueue:
    def __init__(self, device_idx, width, height, fps, flip=True, **camera_options):
        # if DSHOW there is no hardware time, but quick loading
        # self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # if CAP_MSMF there is hardware time but loading time takes 1.5 minutes
        self.cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        # if CAP_ANY the program froze (nge-hang)
        # self.cap = cv2.VideoCapture(0,  cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.flip = flip
        self.stopped = False
        self.last_frame_time = 0
        self.frame_id = 0
        self.read_failures = 0
        self.frame_queue = queue.Queue(maxsize=1200)
        self.thread = None
        self.error = None
        time_baseline = time.time()
        perf_baseline = time.perf_counter()
        self.clock_offset = time_baseline - perf_baseline

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Capture stream already started")
        self.thread = threading.Thread(target=self.update, name="EyeCameraCapture", daemon=True)
        self.thread.start()
        return self

    def update(self):
        try:
            while not self.stopped:
                current_hardware_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                # started = time.perf_counter()
                ret, frame = self.cap.read()
                captured = time.perf_counter()
                timestamp = self.clock_offset + captured
                if not ret:
                    self.read_failures += 1
                    time.sleep(0.005)
                    continue
                if current_hardware_time == self.last_frame_time:
                    continue
                else:
                    time_diff = current_hardware_time - self.last_frame_time
                    # print("time diff:", time_diff)
                    if time_diff < 0:
                        print("Negative Hardware Time Diff, probably not supported, exiting..")
                        raise Exception("Negative Hardware Time Diff, probably not supported, exiting..")
                self.last_frame_time = current_hardware_time
                if self.flip:
                    frame = cv2.flip(frame, 1)
                wframe = WFrame(id=self.frame_id, frame=frame, fps=1000.0/time_diff, timestamp=timestamp)
                self.frame_queue.put(wframe)

                self.frame_id += 1
        except Exception as exc:
            self.error = str(exc)
            print("Error while updating: "+self.error)
        finally:
            # Release from the owning thread, never concurrently with cap.read().
            self.cap.release()
            self.stopped = True

    def read(self):
        wframe = self.frame_queue.get(block=True)
        # print("Read, qsize: ", self.frame_queue.qsize())
        return (True, wframe) if wframe is not None else (False, None)

    def read_latest(self, after_frame_id=0, timeout=0.1):
        return read()

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                raise RuntimeError("Camera read did not stop within 2 seconds")
        else:
            self.cap.release()

# ==========================================
# MAIN APPLICATION LOOP
# ==========================================
if __name__ == "__main__":
    # Generate a timestamp (e.g., "20261016_135300")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = f"dataset_eyes_{timestamp}"
    
    print(f"Saving session data to: {session_dir}")
    
    tracker = IrisGazeTracker(save_dir=session_dir)
    # stream = WebcamStreamWithQueue(0, 640, 480, 60, True).start()
    # stream = WebcamStreamWithQueue(0, 1280, 720, 60, True).start()
    stream = WebcamStreamWithQueue(0, 1920, 1080, 60, True).start()
    
    frame_id = 0
    DISPLAY_EVERY_N_FRAMES = 4  # e.g., 60 FPS capture / 15 FPS display

    print("Starting tracker. Press 'q' to exit.")
    start_time = time.perf_counter()
    
    while True:
        success, wframe = stream.read()
        if not success: break
        
        frame = wframe.frame
        clean_frame = frame.copy()

        # 1. Process data for every single frame
        iris_data = tracker.process_frame(frame)
        
        if iris_data:
            # 2. Save math to CSV and crops to Disk via background thread
            tracker.save_data(clean_frame, iris_data, frame_id, wframe.timestamp, wframe.fps)
            
            # 3. Only draw and show the frame occasionally to save CPU resources
            if frame_id % DISPLAY_EVERY_N_FRAMES == 0:
                frame = tracker.draw_overlay(frame, iris_data)
                display_frame = cv2.resize(frame, (640, 480))
                cv2.imshow('Real-Time Tracker', display_frame)
                
        frame_id += 1
        print("FPS: ", wframe.fps, "FrameID: ", wframe.id, "timestamp: ", wframe.timestamp)

        # 4. Handle exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_id >= 600:
            break

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    avg_fps = total_frames / elapsed_time

    print(f'Total Frames Read: {total_frames}')
    print(f'Elapsed Time: {elapsed_time:.4f} seconds')
    print(f'Average FPS: {avg_fps:.0f}')

    tracker.close()
    stream.stop()
    cv2.destroyAllWindows()