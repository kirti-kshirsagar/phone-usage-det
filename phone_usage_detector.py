import os
import sys
import math
import time
import argparse
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# YOLO (Ultralytics)
from ultralytics import YOLO

# MediaPipe hands + pose (landmarks)
import mediapipe as mp

# Audio handling
import ffmpeg


# ----------------------------
# Utility structures
# ----------------------------
@dataclass
class Detection:
    cls: str
    conf: float
    xyxy: Tuple[int, int, int, int]  # left, top, right, bottom


@dataclass
class Track:
    track_id: int
    cls: str
    conf: float
    xyxy: Tuple[int, int, int, int]
    last_seen_frame: int
    history: deque  # list of centers for speed / motion calc


# ----------------------------
# Geometry helpers
# ----------------------------
def box_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def point_in_box(pt, box):
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    return (max(0, int(x1)), max(0, int(y1)), min(int(x2), w - 1), min(int(y2), h - 1))


# ----------------------------
# Drawing
# ----------------------------
def draw_box_with_label(frame, box, label, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


# ----------------------------
# MediaPipe helpers
# ----------------------------
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

HAND_IDX = {
    "WRIST": 0,
    "THUMB_TIP": 4,
    "INDEX_FINGER_TIP": 8,
    "MIDDLE_FINGER_TIP": 12,
    "RING_FINGER_TIP": 16,
    "PINKY_TIP": 20,
}

POSE_IDX = {
    "NOSE": 0,
    "LEFT_EYE": 2,
    "RIGHT_EYE": 5,
    "LEFT_EAR": 7,
    "RIGHT_EAR": 8,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
}

def to_abs_landmarks(landmarks, w, h):
    pts = []
    for lm in landmarks:
        pts.append((int(lm.x * w), int(lm.y * h)))
    return pts


def make_head_box(pose_pts, w, h, scale=1.2):
    keys = [POSE_IDX["NOSE"], POSE_IDX["LEFT_EYE"], POSE_IDX["RIGHT_EYE"], POSE_IDX["LEFT_EAR"], POSE_IDX["RIGHT_EAR"]]
    use = [pose_pts[i] for i in keys if 0 <= i < len(pose_pts)]
    if not use:
        return None
    xs = [p[0] for p in use]
    ys = [p[1] for p in use]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w0, h0 = (x2 - x1), (y2 - y1)
    w1, h1 = w0 * scale, h0 * scale
    return clip_box((cx - w1 / 2, cy - h1 / 2, cx + w1 / 2, cy + h1 / 2), w, h)


def make_lap_box(pose_pts, w, h, expand=0.15):
    """Rough lap region between hips and knees."""
    try:
        lh, rh = pose_pts[POSE_IDX["LEFT_HIP"]], pose_pts[POSE_IDX["RIGHT_HIP"]]
        lk, rk = pose_pts[POSE_IDX["LEFT_KNEE"]], pose_pts[POSE_IDX["RIGHT_KNEE"]]
    except Exception:
        return None
    x1 = min(lh[0], rh[0], lk[0], rk[0])
    x2 = max(lh[0], rh[0], lk[0], rk[0])
    y1 = min(lh[1], rh[1], lk[1], rk[1])
    y2 = max(lh[1], rh[1], lk[1], rk[1])
    # Slight expansion
    dx = int((x2 - x1) * expand)
    dy = int((y2 - y1) * expand)
    return clip_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), w, h)


# ----------------------------
# Simple phone tracker (center-distance matching)
# ----------------------------
class SimpleTracker:
    def __init__(self, max_age=30, max_dist=80):
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.max_age = max_age
        self.max_dist = max_dist

    def update(self, detections: List[Detection], frame_idx: int) -> Dict[int, Track]:
        # Only track phones
        phone_dets = [d for d in detections if d.cls == "cell phone"]
        det_centers = [box_center(d.xyxy) for d in phone_dets]

        # Build list of active track centers
        track_ids = list(self.tracks.keys())
        track_centers = [box_center(self.tracks[tid].xyxy) for tid in track_ids]

        # Match greedily by nearest center
        assigned_det = set()
        assigned_track = set()
        for ti, tcenter in enumerate(track_centers):
            best = -1
            best_dist = 1e9
            for di, dcenter in enumerate(det_centers):
                if di in assigned_det:
                    continue
                dd = dist(tcenter, dcenter)
                if dd < best_dist and dd < self.max_dist:
                    best_dist = dd
                    best = di
            if best >= 0:
                # Update existing track
                tid = track_ids[ti]
                d = phone_dets[best]
                self.tracks[tid].xyxy = d.xyxy
                self.tracks[tid].conf = d.conf
                self.tracks[tid].last_seen_frame = frame_idx
                c = box_center(d.xyxy)
                self.tracks[tid].history.append(c)
                if len(self.tracks[tid].history) > 15:
                    self.tracks[tid].history.popleft()
                assigned_det.add(best)
                assigned_track.add(tid)

        # Create new tracks for unmatched detections
        for di, d in enumerate(phone_dets):
            if di in assigned_det:
                continue
            tid = self.next_id
            self.next_id += 1
            c = box_center(d.xyxy)
            self.tracks[tid] = Track(
                track_id=tid,
                cls=d.cls,
                conf=d.conf,
                xyxy=d.xyxy,
                last_seen_frame=frame_idx,
                history=deque([c], maxlen=15),
            )
            assigned_track.add(tid)

        # Remove stale tracks
        to_del = []
        for tid, tr in self.tracks.items():
            if frame_idx - tr.last_seen_frame > self.max_age:
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

        return {tid: self.tracks[tid] for tid in assigned_track}


def track_speed(track: Track) -> float:
    """Average per-frame displacement in pixels."""
    if len(track.history) < 3:
        return 0.0
    d = 0.0
    for i in range(1, len(track.history)):
        d += dist(track.history[i - 1], track.history[i])
    return d / (len(track.history) - 1)


# ----------------------------
# Active phone usage logic
# ----------------------------
def is_active_phone_use(
    phone_box,
    person_boxes,
    hand_points,
    head_box,
    lap_box,
    track_speed_px,
    frame_diag,
    proximity_px=80,
    min_speed_px=1.5,
):
    """
    Returns True if phone considered 'actively being used'.
    Stricter heuristics to avoid false positives:
      - Must be near hands AND moving (indicating interaction)
      - OR near face AND moving (indicating viewing/interaction)
      - OR in lap AND near hands AND moving (indicating active use)
      - Static phones (on table, idle) are ignored
    """
    x1, y1, x2, y2 = phone_box
    p_center = ((x1 + x2) / 2, (y1 + y2) / 2)

    # Must be within/overlapping a person box (avoid table phones)
    overlap_person = any(iou(phone_box, pb) > 0.05 for pb in person_boxes)
    if not overlap_person:
        return False  # Phone not near any person

    # Check proximity to hands
    near_hand = False
    for hp in hand_points:
        if dist(p_center, hp) <= proximity_px:
            near_hand = True
            break

    # Check proximity to head/face
    near_head = False
    if head_box is not None and iou(phone_box, head_box) > 0.05:
        near_head = True

    # Check if in lap area
    in_lap = False
    if lap_box is not None and point_in_box(p_center, lap_box):
        in_lap = True

    # Motion cue - must have significant motion to indicate active use
    moving = track_speed_px >= min_speed_px

    # Stricter decision logic:
    # Case 1: Near hands AND moving (active interaction)
    if near_hand and moving:
        return True

    # Case 2: Near face AND moving (active viewing/interaction)
    if near_head and moving:
        return True

    # Case 3: In lap AND near hands AND moving (active lap use)
    if in_lap and near_hand and moving:
        return True

    # Case 4: High motion near person (indicating active manipulation)
    if moving and track_speed_px >= min_speed_px * 2:  # Double the minimum speed
        return True

    return False


# ----------------------------
# Audio helpers
# ----------------------------
def extract_audio_to_temp(in_path) -> Optional[str]:
    try:
        tmp_audio = tempfile.NamedTemporaryFile(suffix=".aac", delete=False)
        tmp_audio.close()
        (
            ffmpeg
            .input(in_path)
            .output(tmp_audio.name, vn=None, acodec='copy')
            .overwrite_output()
            .run(quiet=True)
        )
        return tmp_audio.name
    except Exception:
        return None


def mux_audio(video_no_audio, audio_path, out_path):
    try:
        (
            ffmpeg
            .input(video_no_audio)
            .input(audio_path)
            .output(out_path, vcodec='copy', acodec='aac', strict='experimental', shortest=None)
            .overwrite_output()
            .run(quiet=True)
        )
        return True
    except Exception:
        return False


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    print("Starting phone usage detector...")
    parser = argparse.ArgumentParser("Phone Usage Detector")
    parser.add_argument("--input", required=True, help="Input video (MP4/AVI/MOV)")
    parser.add_argument("--output", required=True, help="Output annotated MP4")
    parser.add_argument("--summary", default=None, help="Write a short text summary file")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model (n/s/m/l)")
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO confidence threshold")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Inference device")
    parser.add_argument("--max_age", type=int, default=30, help="Track stale age (frames)")
    parser.add_argument("--max_dist", type=int, default=80, help="Track match max center distance")
    parser.add_argument("--proximity", type=int, default=60, help="Pixels to consider 'near hand'")
    parser.add_argument("--min_speed", type=float, default=1.5, help="Min px/frame to be 'moving'")
    parser.add_argument("--skip", type=int, default=0, help="Process every (skip+1)th frame for speed")
    args = parser.parse_args()

    print(f"Input video: {args.input}")
    print(f"Output video: {args.output}")
    
    assert os.path.exists(args.input), f"Input not found: {args.input}"
    print("Input file exists, creating output directories...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.summary:
        os.makedirs(os.path.dirname(args.summary), exist_ok=True)

    # Open video
    print("Opening video file...")
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print("Failed to open input video.", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video properties: {w}x{h} @ {fps}fps, {total_frames} frames")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Temporary video without audio (we'll re-mux)
    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_video_path = tmp_video.name
    tmp_video.close()
    writer = cv2.VideoWriter(tmp_video_path, fourcc, fps, (w, h))

    # Extract audio if present
    audio_path = extract_audio_to_temp(args.input)

    # Load YOLO
    print("Loading YOLO model...")
    model = YOLO(args.model)
    # Map class names
    names = model.model.names if hasattr(model.model, "names") else model.names
    # Defensive mapping
    name_to_id = {v: k for k, v in (names.items() if isinstance(names, dict) else enumerate(names))}
    print(f"Available classes: {list(name_to_id.keys())}")
    # Require these classes exist
    if "person" not in name_to_id or ("cell phone" not in name_to_id and "cellphone" not in name_to_id):
        print("Model must include 'person' and 'cell phone' classes (COCO).", file=sys.stderr)
        sys.exit(1)
    phone_key = "cell phone" if "cell phone" in name_to_id else "cellphone"
    print(f"Using phone class: {phone_key}")

    # Track state
    tracker = SimpleTracker(max_age=args.max_age, max_dist=args.max_dist)
    tracker.active_count = {}  # Initialize active count dictionary
    frame_idx = 0

    # Logs
    events = []  # list of {frame, time_s, track_id, conf, x1,y1,x2,y2}
    active_state: Dict[int, bool] = defaultdict(lambda: False)

    # MediaPipe contexts
    mp_hands_ctx = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3,
        model_complexity=0
    )
    mp_pose_ctx = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3
    )

    frame_diag = math.hypot(w, h)

    print("Starting video processing...")
    pbar = tqdm(total=total_frames, desc="Processing", unit="f")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally skip frames to speed up; we still write every frame (copy prev annotations)
        process_this = (args.skip == 0) or (frame_idx % (args.skip + 1) == 0)

        # Default overlays copied from last results (we’ll compute fresh if process_this)
        draw_overlays: List[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]] = []

        if process_this:
            # Run YOLO
            yolo_res = model.predict(
                source=frame,
                conf=args.conf,
                device=0 if args.device == "cuda" else "cpu",
                verbose=False
            )[0]

            dets: List[Detection] = []
            person_boxes = []
            phone_detections = []
            
            for *xyxy, conf, cls_id in yolo_res.boxes.data.cpu().numpy():
                cls_id = int(cls_id)
                cls_name = names[cls_id] if isinstance(names, dict) else names[cls_id]
                x1, y1, x2, y2 = clip_box(xyxy, w, h)
                if cls_name == "person":
                    person_boxes.append((x1, y1, x2, y2))
                if cls_name == phone_key:
                    phone_detections.append(Detection(cls="cell phone", conf=float(conf), xyxy=(x1, y1, x2, y2)))
            
            # Remove overlapping phone detections
            if len(phone_detections) > 1:
                # Sort by confidence (highest first)
                phone_detections.sort(key=lambda x: x.conf, reverse=True)
                
                # Keep only non-overlapping detections
                filtered_detections = []
                for det in phone_detections:
                    is_overlapping = False
                    for existing in filtered_detections:
                        # Check if boxes overlap significantly
                        overlap = iou(det.xyxy, existing.xyxy)
                        if overlap > 0.4:  # 40% overlap threshold
                            is_overlapping = True
                            break
                    if not is_overlapping:
                        filtered_detections.append(det)
                
                dets = filtered_detections
            else:
                dets = phone_detections

            # MediaPipe landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_points = []
            hands_res = mp_hands_ctx.process(rgb)
            if hands_res.multi_hand_landmarks:
                for hand_lms in hands_res.multi_hand_landmarks:
                    pts = to_abs_landmarks(hand_lms.landmark, w, h)
                    # Use a subset for proximity checks: wrist + finger tips
                    for k in ["WRIST", "THUMB_TIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_TIP", "RING_FINGER_TIP", "PINKY_TIP"]:
                        hand_points.append(pts[HAND_IDX[k]])

            pose_res = mp_pose_ctx.process(rgb)
            head_box, lap_box = None, None
            if pose_res.pose_landmarks:
                pose_pts = to_abs_landmarks(pose_res.pose_landmarks.landmark, w, h)
                head_box = make_head_box(pose_pts, w, h, scale=1.4)
                lap_box = make_lap_box(pose_pts, w, h, expand=0.2)

            # Update tracks
            active_tracks = tracker.update(dets, frame_idx)

            # Remove duplicate tracks that are too close
            if len(active_tracks) > 1:
                track_items = list(active_tracks.items())
                tracks_to_remove = []
                
                for i, (tid1, tr1) in enumerate(track_items):
                    for j, (tid2, tr2) in enumerate(track_items[i+1:], i+1):
                        if tid2 not in tracks_to_remove:
                            # Calculate distance between track centers
                            center1 = ((tr1.xyxy[0] + tr1.xyxy[2]) / 2, (tr1.xyxy[1] + tr1.xyxy[3]) / 2)
                            center2 = ((tr2.xyxy[0] + tr2.xyxy[2]) / 2, (tr2.xyxy[1] + tr2.xyxy[3]) / 2)
                            distance = dist(center1, center2)
                            
                            # If tracks are very close, keep the one with higher confidence
                            if distance < 60:  # 60 pixel threshold
                                if tr1.conf > tr2.conf:
                                    tracks_to_remove.append(tid2)
                                else:
                                    tracks_to_remove.append(tid1)
                
                # Remove duplicate tracks
                for tid in tracks_to_remove:
                    if tid in active_tracks:
                        del active_tracks[tid]

            # Decide activity
            for tid, tr in active_tracks.items():
                spd = track_speed(tr)
                active = is_active_phone_use(
                    tr.xyxy, person_boxes, hand_points, head_box, lap_box,
                    track_speed_px=spd, frame_diag=frame_diag,
                    proximity_px=args.proximity, min_speed_px=args.min_speed
                )

                # Temporal smoothing to reduce flicker and false positives:
                # require 3 consecutive positives to turn on, 2 negatives to turn off
                was_active = active_state[tid]
                
                # Initialize tracking history if not exists
                if tid not in active_state:
                    active_state[tid] = False
                if tid not in tracker.active_count:
                    tracker.active_count[tid] = 0
                
                # Update consecutive frame counters
                if active:
                    tracker.active_count[tid] = min(tracker.active_count[tid] + 1, 5)  # Cap at 5
                else:
                    tracker.active_count[tid] = max(tracker.active_count[tid] - 1, -3)  # Cap at -3

                # Decision logic with stricter thresholds
                if tracker.active_count[tid] >= 3 and not was_active:
                    # Turn on after 3 consecutive active frames
                    active_state[tid] = True
                elif tracker.active_count[tid] <= -2 and was_active:
                    # Turn off after 2 consecutive inactive frames
                    active_state[tid] = False

                if active_state[tid]:
                    label_color = (0, 255, 0)
                    display_label = f"phone {tr.conf:.2f}"
                    events.append({
                        "frame": frame_idx,
                        "time_s": frame_idx / fps,
                        "track_id": tid,
                        "conf": tr.conf,
                        "x1": tr.xyxy[0], "y1": tr.xyxy[1], "x2": tr.xyxy[2], "y2": tr.xyxy[3],
                        "speed_px": spd
                    })
                    draw_overlays.append((tr.xyxy, display_label, label_color))

        # Draw overlays
        for b, lbl, col in draw_overlays:
            draw_box_with_label(frame, b, lbl, col, 2)

        writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    # Mux audio back
    if audio_path:
        ok = mux_audio(tmp_video_path, audio_path, args.output)
        # cleanup temp video if mux succeeded
        if ok:
            try:
                os.remove(tmp_video_path)
            except Exception:
                pass
    else:
        # no audio, just move tmp to output
        os.replace(tmp_video_path, args.output)

    if audio_path:
        try:
            os.remove(audio_path)
        except Exception:
            pass

    # Simple summary
    if args.summary:
        total_active_s = 0.0
        if len(events) > 0:
            df = pd.DataFrame(events)
            total_active_s = (df["time_s"].max() - df["time_s"].min()) if len(df) > 1 else 0.0
        
        summary_lines = [
            f"Video: {os.path.basename(args.input)}",
            f"FPS: {fps:.3f} | Resolution: {w}x{h}",
            f"Total active phone time: {total_active_s:.2f} seconds"
        ]
        with open(args.summary, "w") as f:
            f.write("\n".join(summary_lines))

    print("Done.")


if __name__ == "__main__":
    main()
