"""
Object Tracker
===============
Wraps YOLOv8 with ByteTrack for persistent multi-object tracking.
Maintains a history of positions for trajectory analysis.
"""

import numpy as np
from collections import defaultdict
from ultralytics import YOLO

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    YOLO_MODEL, YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD,
    TRACKER_TYPE, CLASSES_OF_INTEREST, MAX_TRACK_HISTORY,
    COCO_CLASSES
)


class ObjectTracker:
    """
    YOLOv8 + ByteTrack object tracker.

    Provides:
    - Real-time object detection and tracking
    - Persistent track IDs across frames
    - Velocity calculation for each tracked object
    - Position history for trajectory/loitering analysis
    """

    def __init__(self, model_path: str = None):
        """Initialize the tracker with a YOLOv8 model."""
        model_name = model_path or YOLO_MODEL
        print(f"  📦 Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)
        print(f"  ✅ Model loaded successfully")

        # Track history: {track_id: [(cx, cy), ...]}
        self.track_history: dict[int, list[tuple[int, int]]] = defaultdict(list)

        # Frame count per track: {track_id: frame_count}
        self.track_frame_count: dict[int, int] = defaultdict(int)

        # Previous positions for velocity: {track_id: (cx, cy)}
        self.prev_positions: dict[int, tuple[int, int]] = {}

    def process_frame(self, frame: np.ndarray) -> tuple[list[dict], list[dict]]:
        """
        Run detection + tracking on a single frame.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            (detections, tracked_objects) — both as lists of dicts
        """
        # Run YOLOv8 tracking
        results = self.model.track(
            frame,
            persist=True,
            tracker=TRACKER_TYPE,
            conf=YOLO_CONFIDENCE,
            iou=YOLO_IOU_THRESHOLD,
            classes=CLASSES_OF_INTEREST,
            verbose=False
        )

        detections = []
        tracked_objects = []

        if results and len(results) > 0:
            result = results[0]

            # Check if we have valid boxes
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                # Track IDs might not be available on every frame
                has_ids = result.boxes.id is not None
                track_ids = result.boxes.id.cpu().numpy().astype(int) if has_ids else None

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    class_id = int(class_ids[i])
                    class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
                    conf = float(confidences[i])

                    # Raw detection
                    detection = {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(conf, 3),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "center": [int(cx), int(cy)]
                    }
                    detections.append(detection)

                    # Tracked object (with ID)
                    if has_ids and track_ids is not None:
                        tid = int(track_ids[i])

                        # Calculate velocity
                        velocity = 0.0
                        if tid in self.prev_positions:
                            prev_cx, prev_cy = self.prev_positions[tid]
                            velocity = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)

                        # Update history
                        self.track_history[tid].append((int(cx), int(cy)))
                        if len(self.track_history[tid]) > MAX_TRACK_HISTORY:
                            self.track_history[tid] = self.track_history[tid][-MAX_TRACK_HISTORY:]

                        # Update frame count
                        self.track_frame_count[tid] += 1

                        # Update previous position
                        self.prev_positions[tid] = (int(cx), int(cy))

                        tracked_obj = {
                            "track_id": tid,
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": round(conf, 3),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "center": [int(cx), int(cy)],
                            "velocity": round(float(velocity), 2),
                            "frames_tracked": self.track_frame_count[tid]
                        }
                        tracked_objects.append(tracked_obj)

        return detections, tracked_objects

    def get_track_history(self, track_id: int) -> list[tuple[int, int]]:
        """Get position history for a specific track ID."""
        return self.track_history.get(track_id, [])

    def get_all_histories(self) -> dict[int, list[tuple[int, int]]]:
        """Get position histories for all tracked objects."""
        return dict(self.track_history)

    def get_annotated_frame(self, results) -> np.ndarray:
        """Get the frame with YOLO annotations drawn."""
        if results and len(results) > 0:
            return results[0].plot()
        return None

    def cleanup_stale_tracks(self, active_ids: set[int]):
        """Remove history for tracks that are no longer active."""
        stale_ids = set(self.track_history.keys()) - active_ids
        for tid in stale_ids:
            # Keep stale tracks for a while in case they reappear
            if self.track_frame_count.get(tid, 0) > 0:
                self.track_frame_count[tid] = 0  # Mark as inactive
            else:
                # Remove after being inactive
                self.track_history.pop(tid, None)
                self.track_frame_count.pop(tid, None)
                self.prev_positions.pop(tid, None)
