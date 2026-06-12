"""
Video Surveillance Agent — Main Entry Point
==========================================
Coordinates video input streams, executes the LangGraph agent pipeline,
and renders a premium, feature-rich OpenCV visualization output.
"""

import os
import cv2
import time
import argparse
from datetime import datetime
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    VIDEO_SOURCE, YOLO_MODEL, SHOW_TRAJECTORIES, RESTRICTED_ZONES,
    DISPLAY_WIDTH, DISPLAY_HEIGHT
)
from agents.graph import surveillance_graph
from agents.vision_agent import get_tracker
from agents.reasoning_agent import get_llm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI Surveillance Agent using YOLOv8 & LangGraph")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to video file, RTSP stream URL, or webcam index (default from config)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="Display the visual monitoring window"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DISPLAY_WIDTH,
        help="Width of display window"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DISPLAY_HEIGHT,
        help="Height of display window"
    )
    return parser.parse_args()

def draw_overlay(frame: np.ndarray, state: dict, fps: float):
    """
    Renders a premium visual heads-up display overlay on the video frame,
    showing system statistics, active alerts, and status indicators.
    """
    h, w, _ = frame.shape
    
    # ── 1. Top Status Banner ──
    # Create semi-transparent overlay banner at top
    banner_h = 75
    banner_overlay = frame.copy()
    cv2.rectangle(banner_overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    # Blend with original
    cv2.addWeighted(banner_overlay, 0.75, frame, 0.25, 0, frame)
    
    # Render Status Title
    cv2.putText(
        frame, "AI SURVEILLANCE MATRIX v1.0", (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
    )
    
    # Render Time
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        frame, now_str, (20, 55),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA
    )
    
    # Render Status Badges
    # 1. Pipeline Status
    cv2.rectangle(frame, (w - 530, 20), (w - 420, 45), (0, 120, 0), -1)
    cv2.putText(
        frame, "SYS: ACTIVE", (w - 520, 37),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
    )
    
    # 2. LLM Engine Status
    llm = get_llm()
    llm_status = "GROQ LLM" if llm is not None else "SIMULATED"
    llm_color = (0, 150, 255) if llm is not None else (120, 120, 120)
    cv2.rectangle(frame, (w - 400, 20), (w - 290, 45), llm_color, -1)
    cv2.putText(
        frame, llm_status, (w - 390, 37),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
    )
    
    # 3. FPS & Count metrics
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (w - 270, 37),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
    )
    person_cnt = state.get("person_count", 0)
    cv2.putText(
        frame, f"PEOPLE: {person_cnt}", (w - 180, 37),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
    )
    
    # ── 2. Render Detections and Tracks ──
    tracker = get_tracker()
    tracked_objects = state.get("tracked_objects", [])
    
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj["bbox"]
        tid = obj["track_id"]
        cname = obj["class_name"]
        
        # Color based on class: red for people, green for cars/others
        box_color = (0, 165, 255) if cname == "person" else (255, 255, 0)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        
        # Tag text: "Person #3"
        tag = f"{cname.title()} #{tid}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 10, y1), box_color, -1)
        cv2.putText(
            frame, tag, (x1 + 5, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA
        )
        
        # Render trajectory line if enabled
        if SHOW_TRAJECTORIES:
            history = tracker.get_track_history(tid)
            if len(history) > 1:
                # Draw lines connecting centers
                for idx in range(1, len(history)):
                    cv2.line(frame, history[idx-1], history[idx], box_color, 1, cv2.LINE_AA)

    # ── 3. Render Restricted Zones ──
    for zone_idx, zone in enumerate(RESTRICTED_ZONES):
        pts = np.array(zone, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        # Label zone
        cv2.putText(
            frame, f"RESTRICTED ZONE #{zone_idx + 1}", (zone[0][0], zone[0][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA
        )

    # ── 4. Render Active Alerts Card (Left Side) ──
    anomalies = state.get("anomalies", [])
    if anomalies:
        card_w = 350
        card_h = 100 + (len(anomalies) * 45)
        # Prevent going off frame
        card_h = min(card_h, h - banner_h - 20)
        
        card_overlay = frame.copy()
        # Draw dark red backing card for alert status
        cv2.rectangle(card_overlay, (15, banner_h + 15), (15 + card_w, banner_h + 15 + card_h), (0, 0, 60), -1)
        cv2.addWeighted(card_overlay, 0.75, frame, 0.25, 0, frame)
        
        # Outer neon red border
        cv2.rectangle(frame, (15, banner_h + 15), (15 + card_w, banner_h + 15 + card_h), (0, 0, 255), 1)
        
        cv2.putText(
            frame, "⚠️ SECURITY EXCEPTION DETECTED", (30, banner_h + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA
        )
        
        # Draw listing of recent anomalies
        for a_idx, anomaly in enumerate(anomalies[:5]):  # limit to top 5
            offset_y = banner_h + 75 + (a_idx * 45)
            # Dot indicator
            cv2.circle(frame, (35, offset_y - 5), 4, (0, 0, 255), -1)
            
            # Anomaly details
            anomaly_desc = anomaly["description"]
            if len(anomaly_desc) > 42:
                anomaly_desc = anomaly_desc[:39] + "..."
            
            cv2.putText(
                frame, anomaly_desc, (50, offset_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                frame, f"Severity: {anomaly['severity']} | {anomaly['anomaly_type'].upper()}",
                (50, offset_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA
            )

def main():
    """Main feed loop."""
    args = parse_args()
    
    # Resolve video source: CLI argument overrides env/config settings
    video_source = args.source
    if video_source is None:
        video_source = VIDEO_SOURCE
    else:
        # Check if index
        try:
            video_source = int(video_source)
        except ValueError:
            pass
            
    print(f"🎬 Initializing Video Source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source {video_source}.")
        print("Please verify connection or camera availability.")
        return
        
    print("🚀 Video Surveillance Pipeline Ready.")
    print("   Press 'q' in the window to terminate execution.")
    
    frame_id = 0
    fps_accum = 0.0
    fps_frames = 0
    
    # Warmup or track models
    tracker = get_tracker()
    
    try:
        while cap.isOpened():
            start_time = time.time()
            success, frame = cap.read()
            if not success:
                print("🎥 End of video source stream or cannot fetch frame.")
                break
                
            frame_id += 1
            
            # Resize frame for standard processing and rendering
            frame = cv2.resize(frame, (args.width, args.height))
            
            # Encode frame to JPEG bytes to pass in state (enables microservices & serializability)
            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            
            # Prepare state dict
            initial_state = {
                "frame_bytes": frame_bytes,
                "frame_id": frame_id,
                "timestamp": datetime.now().isoformat(),
                "detections": [],
                "tracked_objects": [],
                "person_count": 0,
                "anomalies": [],
                "has_anomaly": False,
                "llm_analysis": "",
                "messages": [],
                "alerts": [],
                "frame_shape": list(frame.shape),
                "fps": fps_accum / max(1, fps_frames)
            }
            
            # Invoke the LangGraph pipeline
            final_state = surveillance_graph.invoke(initial_state)
            
            # Calculate performance FPS
            elapsed = time.time() - start_time
            current_fps = 1.0 / elapsed
            fps_accum += current_fps
            fps_frames += 1
            avg_fps = fps_accum / fps_frames
            
            # Reset accumulator occasionally
            if fps_frames > 150:
                fps_accum = avg_fps
                fps_frames = 1
                
            # Draw premium visual HUD
            draw_overlay(frame, final_state, current_fps)
            
            # Render window
            if args.show:
                cv2.imshow("Surveillance Feed Matrix", frame)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("🛑 Quitting agent execution loop.")
                    break
                    
    except KeyboardInterrupt:
        print("🛑 Keyboard interrupt received. Exiting.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Surveillance pipeline stopped and windows closed.")

if __name__ == "__main__":
    main()
