"""
Video Surveillance Agent — Configuration
=========================================
Centralized configuration loaded from environment variables (.env file).
All thresholds and settings are configurable here.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────
#  GROQ LLM CONFIGURATION
# ─────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ─────────────────────────────────────────────
#  YOLO MODEL CONFIGURATION
# ─────────────────────────────────────────────
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.35"))
YOLO_IOU_THRESHOLD = float(os.getenv("YOLO_IOU_THRESHOLD", "0.45"))

# Tracker: "bytetrack.yaml" or "botsort.yaml"
TRACKER_TYPE = os.getenv("TRACKER_TYPE", "bytetrack.yaml")

# ─────────────────────────────────────────────
#  VIDEO SOURCE
# ─────────────────────────────────────────────
_video_source = os.getenv("VIDEO_SOURCE", "0")
# Convert to int if it's a webcam index, otherwise keep as string (file path / RTSP)
try:
    VIDEO_SOURCE = int(_video_source)
except ValueError:
    VIDEO_SOURCE = _video_source

# ─────────────────────────────────────────────
#  ANOMALY DETECTION THRESHOLDS
# ─────────────────────────────────────────────

# Crowd detection: number of people in frame to trigger alert
CROWD_THRESHOLD = int(os.getenv("CROWD_THRESHOLD", "5"))

# Loitering: number of frames an object must remain nearly stationary
LOITER_FRAME_THRESHOLD = int(os.getenv("LOITER_FRAME_THRESHOLD", "100"))

# Loitering: pixel movement tolerance to count as "stationary"
LOITER_MOVEMENT_TOLERANCE = float(os.getenv("LOITER_MOVEMENT_TOLERANCE", "30.0"))

# Speed violation: max pixel displacement per frame
SPEED_THRESHOLD = float(os.getenv("SPEED_THRESHOLD", "200.0"))

# New large object: minimum bounding box area (pixels²) to flag
NEW_OBJECT_MIN_AREA = int(os.getenv("NEW_OBJECT_MIN_AREA", "50000"))

# ─────────────────────────────────────────────
#  RESTRICTED ZONES
# ─────────────────────────────────────────────
# Define as list of polygons. Each polygon is a list of (x, y) tuples.
# Example: [[(100, 100), (300, 100), (300, 400), (100, 400)]]
RESTRICTED_ZONES = []

# ─────────────────────────────────────────────
#  TRACKING HISTORY
# ─────────────────────────────────────────────
# Max number of positions to remember per tracked object
MAX_TRACK_HISTORY = int(os.getenv("MAX_TRACK_HISTORY", "300"))

# ─────────────────────────────────────────────
#  DISPLAY SETTINGS
# ─────────────────────────────────────────────
DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", "1280"))
DISPLAY_HEIGHT = int(os.getenv("DISPLAY_HEIGHT", "720"))
SHOW_TRAJECTORIES = os.getenv("SHOW_TRAJECTORIES", "true").lower() == "true"

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, "events.jsonl")

# ─────────────────────────────────────────────
#  CLASSES OF INTEREST
# ─────────────────────────────────────────────
# COCO class IDs to track (None = track all)
# 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
CLASSES_OF_INTEREST = [0, 1, 2, 3, 5, 7, 24, 25, 26, 28]
# 24=backpack, 25=umbrella, 26=handbag, 28=suitcase

# COCO class names lookup
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
    24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie",
    28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
    45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush"
}
