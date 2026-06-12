"""
Anomaly Detection Engine
=========================
Rule-based anomaly detector that analyzes tracked objects
to identify suspicious or unusual events.

Supported anomaly types:
- Crowd Formation: Too many people in the frame
- Loitering: Object remains stationary for too long
- Speed Violation: Object moving unreasonably fast
- Zone Intrusion: Object enters a restricted area
- New Large Object: Sudden appearance of a large object
"""

import numpy as np
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CROWD_THRESHOLD, LOITER_FRAME_THRESHOLD, LOITER_MOVEMENT_TOLERANCE,
    SPEED_THRESHOLD, NEW_OBJECT_MIN_AREA, RESTRICTED_ZONES
)


class AnomalyDetector:
    """
    Rule-based anomaly detection engine.

    Analyzes tracked objects and their histories to detect
    unusual patterns that may indicate security concerns.
    """

    def __init__(self):
        """Initialize the anomaly detector."""
        self.known_objects: set[int] = set()  # Track IDs we've seen before
        self.anomaly_cooldown: dict[str, int] = {}  # Prevent alert spam
        self.cooldown_frames = 90  # ~3 seconds at 30fps

    def analyze(
        self,
        tracked_objects: list[dict],
        track_histories: dict[int, list[tuple[int, int]]],
        frame_id: int,
        timestamp: str
    ) -> list[dict]:
        """
        Run all anomaly detection checks.

        Args:
            tracked_objects: List of tracked object dicts
            track_histories: {track_id: [(cx, cy), ...]}
            frame_id: Current frame number
            timestamp: ISO timestamp string

        Returns:
            List of anomaly dicts
        """
        anomalies = []

        # Decrement cooldowns
        expired_keys = [k for k, v in self.anomaly_cooldown.items() if v <= frame_id]
        for k in expired_keys:
            del self.anomaly_cooldown[k]

        # ── Check 1: Crowd Formation ──
        crowd_anomaly = self._check_crowd(tracked_objects, frame_id, timestamp)
        if crowd_anomaly:
            anomalies.append(crowd_anomaly)

        # ── Check 2: Loitering ──
        loiter_anomalies = self._check_loitering(
            tracked_objects, track_histories, frame_id, timestamp
        )
        anomalies.extend(loiter_anomalies)

        # ── Check 3: Speed Violation ──
        speed_anomalies = self._check_speed(tracked_objects, frame_id, timestamp)
        anomalies.extend(speed_anomalies)

        # ── Check 4: Zone Intrusion ──
        zone_anomalies = self._check_zone_intrusion(
            tracked_objects, frame_id, timestamp
        )
        anomalies.extend(zone_anomalies)

        # ── Check 5: New Large Object ──
        new_obj_anomalies = self._check_new_large_object(
            tracked_objects, frame_id, timestamp
        )
        anomalies.extend(new_obj_anomalies)

        # Update known objects
        for obj in tracked_objects:
            self.known_objects.add(obj["track_id"])

        return anomalies

    def _is_on_cooldown(self, key: str, frame_id: int) -> bool:
        """Check if an anomaly type is on cooldown to prevent spam."""
        return key in self.anomaly_cooldown

    def _set_cooldown(self, key: str, frame_id: int):
        """Set a cooldown for a specific anomaly key."""
        self.anomaly_cooldown[key] = frame_id + self.cooldown_frames

    # ──────────────────────────────────────────
    #  ANOMALY CHECKS
    # ──────────────────────────────────────────

    def _check_crowd(
        self, tracked_objects: list[dict], frame_id: int, timestamp: str
    ) -> dict | None:
        """Detect crowd formation (too many people)."""
        cooldown_key = "crowd"
        if self._is_on_cooldown(cooldown_key, frame_id):
            return None

        people = [obj for obj in tracked_objects if obj["class_name"] == "person"]
        if len(people) >= CROWD_THRESHOLD:
            self._set_cooldown(cooldown_key, frame_id)

            # Calculate centroid of all people
            centers = [obj["center"] for obj in people]
            avg_x = int(np.mean([c[0] for c in centers]))
            avg_y = int(np.mean([c[1] for c in centers]))

            return {
                "anomaly_type": "crowd_formation",
                "severity": "HIGH" if len(people) >= CROWD_THRESHOLD * 2 else "MEDIUM",
                "description": f"Crowd of {len(people)} people detected (threshold: {CROWD_THRESHOLD})",
                "involved_objects": [
                    {"track_id": p["track_id"], "class": p["class_name"]}
                    for p in people
                ],
                "location": [avg_x, avg_y],
                "frame_id": frame_id,
                "timestamp": timestamp
            }
        return None

    def _check_loitering(
        self,
        tracked_objects: list[dict],
        track_histories: dict[int, list[tuple[int, int]]],
        frame_id: int,
        timestamp: str
    ) -> list[dict]:
        """Detect loitering (object stationary for too long)."""
        anomalies = []

        for obj in tracked_objects:
            tid = obj["track_id"]
            cooldown_key = f"loiter_{tid}"

            if self._is_on_cooldown(cooldown_key, frame_id):
                continue

            history = track_histories.get(tid, [])
            if len(history) < LOITER_FRAME_THRESHOLD:
                continue

            # Check movement over the loiter window
            recent = history[-LOITER_FRAME_THRESHOLD:]
            start_pos = np.array(recent[0])
            end_pos = np.array(recent[-1])
            total_displacement = np.linalg.norm(end_pos - start_pos)

            # Also check max spread (to catch oscillating objects)
            positions = np.array(recent)
            spread = np.max(positions, axis=0) - np.min(positions, axis=0)
            max_spread = np.max(spread)

            if total_displacement < LOITER_MOVEMENT_TOLERANCE and max_spread < LOITER_MOVEMENT_TOLERANCE * 2:
                self._set_cooldown(cooldown_key, frame_id)
                anomalies.append({
                    "anomaly_type": "loitering",
                    "severity": "MEDIUM",
                    "description": (
                        f"{obj['class_name'].title()} (ID:{tid}) loitering — "
                        f"stationary for {obj['frames_tracked']} frames "
                        f"(displacement: {total_displacement:.1f}px)"
                    ),
                    "involved_objects": [
                        {"track_id": tid, "class": obj["class_name"]}
                    ],
                    "location": obj["center"],
                    "frame_id": frame_id,
                    "timestamp": timestamp
                })

        return anomalies

    def _check_speed(
        self, tracked_objects: list[dict], frame_id: int, timestamp: str
    ) -> list[dict]:
        """Detect speed violations (abnormally fast movement)."""
        anomalies = []

        for obj in tracked_objects:
            tid = obj["track_id"]
            cooldown_key = f"speed_{tid}"

            if self._is_on_cooldown(cooldown_key, frame_id):
                continue

            if obj["velocity"] > SPEED_THRESHOLD:
                self._set_cooldown(cooldown_key, frame_id)
                anomalies.append({
                    "anomaly_type": "speed_violation",
                    "severity": "HIGH",
                    "description": (
                        f"{obj['class_name'].title()} (ID:{tid}) moving at "
                        f"{obj['velocity']:.1f} px/frame (limit: {SPEED_THRESHOLD})"
                    ),
                    "involved_objects": [
                        {"track_id": tid, "class": obj["class_name"]}
                    ],
                    "location": obj["center"],
                    "frame_id": frame_id,
                    "timestamp": timestamp
                })

        return anomalies

    def _check_zone_intrusion(
        self, tracked_objects: list[dict], frame_id: int, timestamp: str
    ) -> list[dict]:
        """Detect objects entering restricted zones."""
        anomalies = []

        if not RESTRICTED_ZONES:
            return anomalies

        for obj in tracked_objects:
            tid = obj["track_id"]
            cooldown_key = f"zone_{tid}"

            if self._is_on_cooldown(cooldown_key, frame_id):
                continue

            cx, cy = obj["center"]

            for zone_idx, zone_polygon in enumerate(RESTRICTED_ZONES):
                if self._point_in_polygon(cx, cy, zone_polygon):
                    self._set_cooldown(cooldown_key, frame_id)
                    anomalies.append({
                        "anomaly_type": "zone_intrusion",
                        "severity": "CRITICAL",
                        "description": (
                            f"{obj['class_name'].title()} (ID:{tid}) entered "
                            f"restricted zone #{zone_idx + 1}"
                        ),
                        "involved_objects": [
                            {"track_id": tid, "class": obj["class_name"]}
                        ],
                        "location": obj["center"],
                        "frame_id": frame_id,
                        "timestamp": timestamp
                    })
                    break  # One zone alert per object per cooldown

        return anomalies

    def _check_new_large_object(
        self, tracked_objects: list[dict], frame_id: int, timestamp: str
    ) -> list[dict]:
        """Detect sudden appearance of large objects (e.g., abandoned bag)."""
        anomalies = []

        for obj in tracked_objects:
            tid = obj["track_id"]

            # Only flag if this is a NEW track ID (not seen before)
            if tid in self.known_objects:
                continue

            cooldown_key = f"new_obj_{tid}"
            if self._is_on_cooldown(cooldown_key, frame_id):
                continue

            # Calculate bounding box area
            x1, y1, x2, y2 = obj["bbox"]
            area = (x2 - x1) * (y2 - y1)

            if area >= NEW_OBJECT_MIN_AREA:
                self._set_cooldown(cooldown_key, frame_id)
                anomalies.append({
                    "anomaly_type": "new_large_object",
                    "severity": "MEDIUM",
                    "description": (
                        f"New large {obj['class_name']} (ID:{tid}) appeared — "
                        f"area: {area}px² (threshold: {NEW_OBJECT_MIN_AREA}px²)"
                    ),
                    "involved_objects": [
                        {"track_id": tid, "class": obj["class_name"]}
                    ],
                    "location": obj["center"],
                    "frame_id": frame_id,
                    "timestamp": timestamp
                })

        return anomalies

    @staticmethod
    def _point_in_polygon(x: int, y: int, polygon: list[tuple[int, int]]) -> bool:
        """
        Ray-casting algorithm to check if a point is inside a polygon.

        Args:
            x, y: Point coordinates
            polygon: List of (x, y) tuples defining the polygon vertices

        Returns:
            True if point is inside the polygon
        """
        n = len(polygon)
        inside = False
        j = n - 1

        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside
