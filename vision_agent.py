"""
Vision Agent Node
==================
Runs YOLOv8 object tracking on the input video frame.
"""

from typing import Dict, Any
import cv2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state import SurveillanceState
from core.tracker import ObjectTracker

# Global tracker instance
_tracker = None

def get_tracker() -> ObjectTracker:
    """Get or initialize the global ObjectTracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = ObjectTracker()
    return _tracker

def vision_node(state: SurveillanceState) -> Dict[str, Any]:
    """
    LangGraph node that processes the video frame using YOLOv8 tracking.
    
    Args:
        state: Current surveillance state
        
    Returns:
        State update dictionary containing detections, tracked_objects, and person_count.
    """
    frame_bytes = state.get("frame_bytes")
    if not frame_bytes:
        return {
            "detections": [],
            "tracked_objects": [],
            "person_count": 0
        }
        
    # Decode frame from bytes (since state might be serialized)
    import numpy as np
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return {
            "detections": [],
            "tracked_objects": [],
            "person_count": 0
        }
        
    tracker = get_tracker()
    detections, tracked_objects = tracker.process_frame(frame)
    
    # Count people
    person_count = sum(1 for obj in detections if obj["class_name"] == "person")
    
    # Cleanup stale histories in tracker based on active IDs
    active_ids = {obj["track_id"] for obj in tracked_objects}
    tracker.cleanup_stale_tracks(active_ids)
    
    return {
        "detections": detections,
        "tracked_objects": tracked_objects,
        "person_count": person_count
    }
