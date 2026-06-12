"""
Anomaly Agent Node
===================
Analyzes tracked objects to detect any rule-based anomalies.
"""

from typing import Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state import SurveillanceState
from core.anomaly_detector import AnomalyDetector
from agents.vision_agent import get_tracker

# Global anomaly detector instance
_detector = None

def get_detector() -> AnomalyDetector:
    """Get or initialize the global AnomalyDetector instance."""
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector

def anomaly_node(state: SurveillanceState) -> Dict[str, Any]:
    """
    LangGraph node that runs rule-based anomaly checks on tracked objects.
    
    Args:
        state: Current surveillance state
        
    Returns:
        State update dictionary containing anomalies and the has_anomaly flag.
    """
    tracked_objects = state.get("tracked_objects", [])
    frame_id = state.get("frame_id", 0)
    timestamp = state.get("timestamp", "")
    
    tracker = get_tracker()
    detector = get_detector()
    
    # Get all position histories from the tracker
    track_histories = tracker.get_all_histories()
    
    # Analyze for anomalies
    anomalies = detector.analyze(
        tracked_objects=tracked_objects,
        track_histories=track_histories,
        frame_id=frame_id,
        timestamp=timestamp
    )
    
    has_anomaly = len(anomalies) > 0
    
    return {
        "anomalies": anomalies,
        "has_anomaly": has_anomaly
    }
