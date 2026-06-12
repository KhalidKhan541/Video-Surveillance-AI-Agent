"""
Video Surveillance Agent — Streamlit Dashboard
==============================================
A premium, dark-themed dashboard interface providing:
1. Live camera monitor running the YOLOv8 + LangGraph pipeline.
2. Historical Incident log analyzer reading from `logs/events.jsonl`.
3. Metric breakdown charts for anomaly counts, threat levels, and hourly distribution.
"""

import os
import cv2
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LOG_FILE, YOLO_MODEL, CROWD_THRESHOLD, SPEED_THRESHOLD
from agents.graph import surveillance_graph
from agents.vision_agent import get_tracker
from agents.reasoning_agent import get_llm

# Page config: Theme is set to dark via custom CSS inside the app
st.set_page_config(
    page_title="AI Surveillance Control Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Glassmorphism & Neon tones)
st.markdown("""
<style>
    /* Dark theme customizations */
    .reportview-container {
        background: #0d0f12;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #00ffcc !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .stMetric {
        background: rgba(30, 41, 59, 0.45);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(0, 255, 204, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stMetric label {
        color: #94a3b8 !important;
    }
    .stMetric div[data-testid="stMetricValue"] {
        color: #00ffcc !important;
    }
    .stButton>button {
        background-color: #0f172a;
        color: #00ffcc;
        border: 1px solid #00ffcc;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00ffcc;
        color: #0f172a;
        box-shadow: 0 0 15px rgba(0, 255, 204, 0.4);
    }
    div[data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

def load_logs() -> list[dict]:
    """Load logged incident events from the JSON lines file."""
    if not os.path.exists(LOG_FILE):
        return []
        
    events = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
    except Exception as e:
        st.error(f"Error reading log file: {e}")
    return events

# Sidebar controls
st.sidebar.title("🛡️ Control Panel")
st.sidebar.markdown("---")

# Model indicator
llm = get_llm()
llm_name = "Groq Llama-3" if llm is not None else "Simulated Local Engine"
st.sidebar.subheader("System Status")
st.sidebar.info(f"🧠 LLM Engine: **{llm_name}**")
st.sidebar.success(f"👁️ Vision Model: **YOLOv8n**")

# Tab structure
tab_live, tab_history = st.tabs(["📺 Live Feed Monitor", "📊 Incident Analytics"])

# ── Tab 1: Live Feed Monitor ──
with tab_live:
    st.header("Real-Time Intelligent Camera Stream")
    
    col_video, col_stats = st.columns([3, 1])
    
    with col_stats:
        st.subheader("System Telemetry")
        
        # Placeholders for dynamic updates
        stat_fps = st.empty()
        stat_people = st.empty()
        stat_status = st.empty()
        
        st.markdown("---")
        st.subheader("Latest Detections")
        detection_display = st.empty()
        
        st.markdown("---")
        st.subheader("Interactive Settings")
        show_traj = st.checkbox("Render Trajectory Lines", value=True)
        conf_thresh = st.slider("Detection Confidence Threshold", 0.1, 1.0, 0.35, 0.05)

    with col_video:
        # Action controls for Streamlit capture
        start_btn = st.button("▶️ Start Surveillance Stream")
        stop_btn = st.button("⏹️ Stop Stream")
        
        video_placeholder = st.empty()
        
        if start_btn:
            # Set up capture
            cap = cv2.VideoCapture(0)  # Default webcam
            
            if not cap.isOpened():
                st.error("❌ Could not connect to webcam index 0. Make sure webcam is plugged in.")
            else:
                frame_id = 0
                fps_accum = 0.0
                fps_frames = 0
                tracker = get_tracker()
                
                # Active streaming loop
                while cap.isOpened() and not stop_btn:
                    start_time = time.time()
                    success, frame = cap.read()
                    if not success:
                        st.warning("Camera stream disconnected.")
                        break
                        
                    frame_id += 1
                    
                    # Process frame
                    frame = cv2.resize(frame, (854, 480))
                    
                    # Convert to bytes
                    _, buffer = cv2.imencode(".jpg", frame)
                    frame_bytes = buffer.tobytes()
                    
                    # Execute graph
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
                        "fps": 0.0
                    }
                    
                    final_state = surveillance_graph.invoke(initial_state)
                    
                    # Update stats
                    elapsed = time.time() - start_time
                    curr_fps = 1.0 / elapsed
                    
                    stat_fps.metric("Capture Speed", f"{curr_fps:.1f} FPS")
                    stat_people.metric("People in Frame", f"{final_state.get('person_count', 0)}")
                    
                    if final_state.get("has_anomaly"):
                        stat_status.markdown("<h3 style='color: #ef4444;'>🚨 ANOMALY DETECTED</h3>", unsafe_allow_html=True)
                    else:
                        stat_status.markdown("<h3 style='color: #22c55e;'>🟢 SECURE</h3>", unsafe_allow_html=True)
                        
                    # Render visual boxes on frame
                    annotated_frame = frame.copy()
                    for obj in final_state.get("tracked_objects", []):
                        x1, y1, x2, y2 = obj["bbox"]
                        tid = obj["track_id"]
                        cname = obj["class_name"]
                        box_color = (0, 165, 255) if cname == "person" else (255, 255, 0)
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(
                            annotated_frame, f"{cname.title()} #{tid}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA
                        )
                        
                        if show_traj:
                            history = tracker.get_track_history(tid)
                            for h_idx in range(1, len(history)):
                                cv2.line(annotated_frame, history[h_idx-1], history[h_idx], box_color, 1, cv2.LINE_AA)
                                
                    # Show image in Streamlit
                    # Streamlit requires RGB color space
                    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                    
                    # Show list of detections
                    det_list = [f"- {obj['class_name'].title()} (ID: {obj['track_id']})" for obj in final_state.get("tracked_objects", [])]
                    if det_list:
                        detection_display.markdown("\n".join(det_list))
                    else:
                        detection_display.info("No active objects tracked.")
                        
                    # Short sleep to prevent Streamlit overloading
                    time.sleep(0.01)
                    
                cap.release()

# ── Tab 2: Incident Analytics ──
with tab_history:
    st.header("Security Incident & Exception Log")
    
    logs = load_logs()
    
    if not logs:
        st.info("No exceptions logged yet. Run surveillance streams to populate log files.")
    else:
        # Convert to pandas dataframe
        df_list = []
        for l in logs:
            df_list.append({
                "Incident ID": l["alert_id"],
                "Time": datetime.fromisoformat(l["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                "Type": l["anomaly"]["anomaly_type"].replace("_", " ").title(),
                "Severity": l["anomaly"]["severity"],
                "Description": l["anomaly"]["description"]
            })
        df = pd.DataFrame(df_list)
        
        # Display Key Metrics
        total_incidents = len(df)
        high_severity = len(df[df["Severity"].isin(["HIGH", "CRITICAL"])])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Flagged Exceptions", total_incidents)
        c2.metric("High/Critical Threats", high_severity, delta_color="inverse")
        c3.metric("Monitored Zones", "Restricted Area #1")
        
        st.markdown("---")
        
        # Incident distribution charts
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("Severity Distribution")
            sev_counts = df["Severity"].value_counts()
            st.bar_chart(sev_counts)
            
        with col_chart2:
            st.subheader("Exception Type Frequency")
            type_counts = df["Type"].value_counts()
            st.bar_chart(type_counts)
            
        st.markdown("---")
        st.subheader("Historical Log Records (Interactive)")
        
        # Event detail viewer
        st.dataframe(df.sort_index(ascending=False), use_container_width=True)
        
        # Expander for reading LLM reasoning of selected events
        st.subheader("Read LLM Case Reasoning")
        selected_id = st.selectbox("Select Incident ID to inspect analysis", [l["alert_id"] for l in reversed(logs)])
        
        matched_log = next((l for l in logs if l["alert_id"] == selected_id), None)
        if matched_log:
            with st.expander(f"👁️ Analysis for Incident {selected_id}", expanded=True):
                st.markdown(f"**Anomaly Description:** {matched_log['anomaly']['description']}")
                st.markdown("---")
                st.markdown(matched_log["llm_analysis"])
