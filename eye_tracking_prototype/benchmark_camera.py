"""Camera diagnostics without calibration. Does not save images or change exposure."""
import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2
from modules.config import Config
from modules.stream import open_camera, WebcamStream
from modules.performance import timing_summary, TimerResolution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--warmup', type=float, default=3)
    parser.add_argument('--backend', choices=['auto', 'dshow', 'msmf', 'any'])
    parser.add_argument('--fourcc', help='default, MJPG, YUY2, or another four-character code')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--timer-ms', type=int, choices=[1], help='Test scoped Windows timer resolution request')
    parser.add_argument('--stage', choices=['capture', 'preprocess', 'inference'], default='capture')
    parser.add_argument('--threaded', action='store_true', help='Use the application latest-frame capture thread')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 0 < args.seconds < float('inf') or not 0 <= args.warmup < float('inf'):
        parser.error('seconds must be positive and warmup non-negative, both finite')
    cfg = Config(args.config)
    fourcc = cfg.camera_fourcc if args.fourcc is None else (None if args.fourcc == 'default' else args.fourcc)
    if fourcc is not None and len(fourcc) != 4:
        parser.error('fourcc must contain four characters or be default')
    if any(value is not None and value <= 0 for value in (args.width, args.height)):
        parser.error('width/height must be positive')
    landmarker = None
    cap = None
    stream = None
    timer = TimerResolution(args.timer_ms).start()
    try:
        if args.stage == 'inference':
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            model = cfg.data['mediapipe']
            landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=cfg.model_path),
                running_mode=vision.RunningMode.VIDEO, num_faces=model['num_faces'],
                min_face_detection_confidence=model['min_face_detection_confidence'],
                min_face_presence_confidence=model['min_face_presence_confidence'],
                min_tracking_confidence=model['min_tracking_confidence']))
        camera_args = (cfg.webcam_idx, args.width or cfg.webcam_w, args.height or cfg.webcam_h, cfg.webcam_fps)
        camera_options = dict(backend=args.backend or cfg.camera_backend, fourcc=fourcc,
                              buffer_size=cfg.camera_buffer_size)
        if args.threaded:
            stream = WebcamStream(*camera_args, flip=cfg.flip_horizontal, **camera_options).start()
            camera = stream.camera_info
        else:
            cap, camera = open_camera(*camera_args, **camera_options)
        print(json.dumps(camera, indent=2), flush=True)
        started = time.perf_counter()
        measure_at = started + args.warmup
        deadline = measure_at + args.seconds
        timestamps, reads, processing = [], [], []
        failures = faces = dropped = 0
        last_frame_id = 0
        measured_frame_id = None
        capture_ages = []
        last_model_ts = -1
        while time.perf_counter() < deadline:
            before = time.perf_counter()
            if stream is not None:
                ok, frame, frame_id, captured = stream.read_latest(last_frame_id, timeout=.2)
                if stream.stopped:
                    raise RuntimeError(stream.error or 'Capture stream stopped')
                if ok:
                    last_frame_id = frame_id
            else:
                ok, frame = cap.read()
                captured = time.perf_counter()
            received = time.perf_counter()
            if not ok:
                if received >= measure_at:
                    failures += 1
                time.sleep(0.005)
                continue
            if args.stage != 'capture':
                if cfg.flip_horizontal and stream is None:
                    frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (cfg.inference_w, cfg.inference_h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if landmarker is not None:
                    last_model_ts = max(last_model_ts + 1, int((captured - started) * 1000))
                    result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), last_model_ts)
                    if captured >= measure_at:
                        faces += bool(result.face_landmarks)
            finished = time.perf_counter()
            if captured >= measure_at:
                if stream is not None:
                    if measured_frame_id is not None:
                        dropped += frame_id - measured_frame_id - 1
                    measured_frame_id = frame_id
                timestamps.append(captured)
                reads.append((received - before) * 1000)
                capture_ages.append((received - captured) * 1000)
                processing.append((finished - received) * 1000)
        intervals = [(b - a) * 1000 for a, b in zip(timestamps, timestamps[1:])]
        span = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        fps = (len(timestamps) - 1) / span if span else 0
        report = {'stage': args.stage, 'architecture': 'latest-frame thread' if stream else 'synchronous read then processing',
                  'opencv': cv2.__version__, 'python': platform.python_version(), 'camera': camera,
                  'timer_ms': args.timer_ms, 'timer_request_active': timer.active,
                  'warmup_seconds': args.warmup, 'requested_seconds': args.seconds,
                  'frames': len(timestamps), 'effective_fps': fps, 'read_failures': failures,
                  'dropped_frames': dropped, 'dropped_ratio': dropped / max(len(timestamps) + dropped, 1),
                  'face_frames': faces if landmarker else None,
                  'capture_interval_ms': timing_summary(intervals),
                  'read_or_wait_ms': timing_summary(reads), 'processing_ms': timing_summary(processing),
                  'capture_age_ms': timing_summary(capture_ages),
                  'capture_thread': stream.diagnostics() if stream else None,
                  'note': 'Host delivery timing; not sensor exposure time or proof of distinct images.'}
        print(json.dumps(report, indent=2), flush=True)
        output = args.output or Path('tmp') / ('camera_benchmark_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as report_file:
            json.dump(report, report_file, indent=2)
        print(f'Report: {output.resolve()}')
    finally:
        try:
            if stream is not None:
                stream.stop()
            if cap is not None:
                cap.release()
        finally:
            try:
                if landmarker is not None:
                    landmarker.close()
            finally:
                timer.close()


if __name__ == '__main__':
    main()
