<div align="center">

# Video Surveillance AI Agent

### Intelligent CCTV Monitoring with YOLOv8 + LangGraph + Groq LLM

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/detector-YOLOv8-red.svg)](https://github.com/ultralytics/ultralytics)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/LLM-Groq-purple.svg)](https://groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

*An AI-powered surveillance system where specialized agents collaborate in real-time to detect anomalies, reason about threats, and trigger security responses — all from a live video feed.*

[**Quick Start**](#quick-start) · [**Architecture**](#architecture) · [**Dashboard**](#streamlit-dashboard) · [**Configuration**](#configuration)

</div>

---

## What Is This?

A multi-agent video surveillance system that combines:

1. **Computer Vision** — YOLOv8 + ByteTrack for real-time object detection and persistent tracking
2. **Rule-Based Anomaly Detection** — Mathematical heuristics for crowd, loitering, speed, and zone violations
3. **LLM Reasoning** — Groq-powered threat assessment and action planning
4. **Dual Interfaces** — OpenCV HUD for live monitoring + Streamlit dashboard for analytics

```
📹 Video Feed → 🔍 Vision Agent → 🧠 Anomaly Agent → 🤖 Reasoning Agent → ⚡ Action Agent
                   (YOLOv8)         (Heuristics)       (Groq LLM)         (Alerts)
```

The system is **cost-optimized** — the heavy CV model runs on every frame, but the expensive LLM is only called when anomalies are detected.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LangGraph StateGraph                             │
│                                                                         │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│   │  Vision Node │───▶│ Anomaly Node │───▶│ Reasoning    │             │
│   │              │    │              │    │ Node         │             │
│   │  YOLOv8 +    │    │  Rule-based  │    │              │             │
│   │  ByteTrack   │    │  heuristics  │    │  Groq LLM    │             │
│   └──────────────┘    └──────┬───────┘    └──────┬───────┘             │
│                              │                    │                     │
│                              │ no anomaly        │                     │
│                              ▼                    ▼                     │
│                           [END]           ┌──────────────┐             │
│                                           │ Action Node  │             │
│                                           │              │             │
│                                           │  Log alerts  │             │
│                                           │  Console HUD │             │
│                                           └──────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │ OpenCV   │        │ Streamlit│        │ JSONL    │
   │ HUD      │        │ Dashboard│        │ Logs     │
   └──────────┘        └──────────┘        └──────────┘
```

### Why Conditional Routing?

| Frame Type | Path | Cost |
|------------|------|------|
| **No anomaly** | Vision → Anomaly → END | ~0 tokens (CV only) |
| **Anomaly detected** | Vision → Anomaly → Reasoning → Action | ~500 tokens (LLM call) |

This means **95%+ of frames** skip the LLM entirely, keeping costs near zero.

---

## Agent Roles

| Agent | Technology | Responsibility |
|-------|------------|----------------|
| **Vision Agent** 🔍 | YOLOv8 + ByteTrack | Detects objects, assigns persistent track IDs, calculates velocity and trajectories |
| **Anomaly Agent** 🧠 | Python heuristics | Applies 5 rule-based checks: crowd, loitering, speed, zone intrusion, new objects |
| **Reasoning Agent** 🤖 | Groq Llama 3.3 70B | Evaluates threat level, provides contextual analysis, recommends actions |
| **Action Agent** ⚡ | Python | Generates alerts, logs to JSONL, prints colored console notifications |

### Anomaly Types Detected

| Anomaly | Trigger | Severity |
|---------|---------|----------|
| **Crowd Formation** | >N people in frame simultaneously | MEDIUM / HIGH |
| **Loitering** | Object stationary for >N frames | MEDIUM |
| **Speed Violation** | Object velocity exceeds threshold | HIGH |
| **Zone Intrusion** | Object enters restricted polygon | CRITICAL |
| **New Large Object** | Large static object appears suddenly | MEDIUM |

---

## Features

- **Real-Time Multi-Object Tracking** — Persistent IDs across frames with velocity mapping
- **5 Anomaly Detection Rules** — Crowd, loitering, speed, zone intrusion, abandoned objects
- **LLM Threat Reasoning** — Groq Llama 3.3 evaluates severity and recommends actions
- **Simulated Sandbox Mode** — Works without API key for instant testing
- **OpenCV Matrix HUD** — Live camera overlay with bounding boxes, trajectories, alerts
- **Streamlit Web Dashboard** — Dark UI with live feed, analytics, incident history
- **JSONL Event Logging** — Every alert persisted for forensic review
- **Configurable Thresholds** — All parameters adjustable via `.env` or `config.py`

---

## Quick Start

### Prerequisites

- Python 3.8+
- Webcam, video file, or RTSP stream
- (Optional) [Groq API key](https://console.groq.com) for LLM reasoning

### Installation

```bash
git clone https://github.com/KhalidKhan541/Video-Surveillance-AI-Agent.git
cd Video-Surveillance-AI-Agent

pip install -r requirements.txt

cp .env.example .env
# Edit .env with your Groq API key (optional — works without it in simulation mode)
```

### Run the OpenCV HUD

```bash
python main.py

# Or with a video file:
python main.py --source path/to/video.mp4

# Or with RTSP stream:
python main.py --source rtsp://username:password@ip:port/stream
```

Press `q` to exit.

### Run the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

Open **http://localhost:8501** in your browser.

---

## Streamlit Dashboard

The web dashboard has two tabs:

### Live Feed Monitor
- Real-time camera stream with detection overlay
- Confidence threshold slider
- Live object count and FPS display
- Active anomaly indicators

### Incident Analytics
- Severity distribution bar chart
- Historical anomaly timeline
- Detailed incident reports with LLM analysis
- Search and filter by date/severity

---

## Configuration

### Environment Variables (`.env`)

```env
# Groq LLM (optional — simulation mode if not set)
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Video source (0 = webcam, or file path / RTSP URL)
VIDEO_SOURCE=0

# YOLO model
YOLO_MODEL=yolov8n.pt
YOLO_CONFIDENCE=0.35
```

### Detection Thresholds (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CROWD_THRESHOLD` | 5 | People count to trigger crowd alert |
| `LOITER_FRAME_THRESHOLD` | 100 | Frames before loitering alert |
| `LOITER_MOVEMENT_TOLERANCE` | 30.0 | Pixel movement tolerance for "stationary" |
| `SPEED_THRESHOLD` | 200.0 | Max pixel displacement per frame |
| `NEW_OBJECT_MIN_AREA` | 50000 | Min bounding box area for new object alert |
| `RESTRICTED_ZONES` | `[]` | List of polygon coordinates for zone intrusion |
| `MAX_TRACK_HISTORY` | 300 | Position history per tracked object |

### Restricted Zones

Define custom zones in `config.py`:

```python
RESTRICTED_ZONES = [
    [(100, 100), (300, 100), (300, 400), (100, 400)],  # Zone 1
    [(500, 200), (700, 200), (700, 500), (500, 500)],  # Zone 2
]
```

---

## Project Structure

```
Video-Surveillance-AI-Agent/
├── .env                    # Environment config (Groq key, video source)
├── config.py               # Centralized thresholds and zone config
├── main.py                 # CLI entry point with OpenCV HUD
├── requirements.txt        # Python dependencies
│
├── state.py                # LangGraph SurveillanceState schema
├── graph.py                # StateGraph wiring and conditional routing
│
├── vision_agent.py         # Node 1: YOLOv8 + ByteTrack tracking
├── anomaly_agent.py        # Node 2: Rule-based anomaly detection
├── reasoning_agent.py      # Node 3: Groq LLM threat assessment
├── action_agent.py         # Node 4: Alert logging and notifications
│
├── core/
│   ├── tracker.py          # YOLOv8 tracking wrapper
│   └── anomaly_detector.py # Mathematical anomaly engine
│
├── dashboard/
│   └── app.py              # Streamlit web dashboard
│
└── logs/
    └── events.jsonl        # JSONL event history
```

---

## How It Works (Deep Dive)

### 1. State Schema

Every frame flows through a shared `SurveillanceState`:

```python
class SurveillanceState(TypedDict):
    frame_bytes: bytes              # JPEG-encoded frame
    frame_id: int                   # Frame counter
    detections: list[Detection]     # Raw YOLO output
    tracked_objects: list[TrackedObject]  # Objects with persistent IDs
    anomalies: list[Anomaly]        # Detected security events
    has_anomaly: bool               # Routing flag
    llm_analysis: str               # Groq reasoning output
    alerts: list[Alert]             # Generated alerts
```

### 2. Tracking Pipeline

```
Frame → YOLOv8 Detection → ByteTrack Association → Velocity Calculation → Position History
```

Each tracked object gets:
- **Persistent `track_id`** — same ID across frames
- **Velocity** — pixel displacement per frame
- **Frames tracked** — how long the object has been visible
- **Position history** — last N positions for trajectory analysis

### 3. Anomaly Detection Rules

```python
# Crowd: count people in frame
if person_count > CROWD_THRESHOLD:
    flag("crowd_formation")

# Loitering: object stationary for N frames
if frames_stationary > LOITER_FRAME_THRESHOLD:
    flag("loitering")

# Speed: velocity exceeds threshold
if velocity > SPEED_THRESHOLD:
    flag("speed_violation")

# Zone intrusion: point inside polygon
if point_in_restricted_zone(center):
    flag("zone_intrusion")

# New object: large static object appears
if new_object and area > NEW_OBJECT_MIN_AREA:
    flag("new_large_object")
```

### 4. LLM Reasoning Prompt

When anomalies are detected, the Reasoning Agent sends this to Groq:

```
You are an expert AI video surveillance analyst.
Here are the anomalies flagged by the rule-based engine:
[anomaly details]

Please provide:
1. Threat Level: [LOW | MEDIUM | HIGH | CRITICAL]
2. Incident Summary
3. Contextual Analysis
4. Action Items
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Object Detection** | YOLOv8 (Ultralytics) | Real-time object detection |
| **Object Tracking** | ByteTrack | Persistent ID assignment across frames |
| **Agent Orchestration** | LangGraph | State machine with conditional routing |
| **LLM Reasoning** | Groq (Llama 3.3 70B) | Threat assessment and action planning |
| **Video I/O** | OpenCV | Frame capture, rendering, display |
| **Web Dashboard** | Streamlit | Interactive monitoring interface |
| **Event Storage** | JSONL | Append-only incident logging |
| **Terminal UI** | Colorama | Colored console alerts |

---

## Requirements

```
ultralytics>=8.0.0
opencv-python>=4.8.0
langgraph>=0.2.0
langchain-core>=0.2.0
langchain-groq>=0.1.0
streamlit>=1.28.0
python-dotenv>=1.0.0
colorama>=0.4.6
numpy>=1.24.0
```

---

## Use Cases

- **Building Security** — Monitor entrances, lobbies, restricted areas
- **Parking Lots** — Detect vehicle speed violations, unauthorized access
- **Public Spaces** — Crowd monitoring, abandoned object detection
- **Industrial Sites** — Zone intrusion, safety compliance
- **Retail** — Loss prevention, foot traffic analysis

---

## Limitations

- **Single Camera** — Currently supports one video source; multi-camera support planned
- **No Persistence** — Tracking resets on restart; no database for long-term analytics
- **Simulation Mode** — Without Groq API, reasoning is rule-based simulation only
- **FPS dependent** — Loitering/speed thresholds are frame-rate sensitive

---

## Roadmap

- [ ] Multi-camera support with unified dashboard
- [ ] PostgreSQL backend for long-term analytics
- [ ] Email/SMS alert integration (Twilio, SendGrid)
- [ ] Custom YOLO model training for domain-specific objects
- [ ] WebSocket streaming for remote monitoring
- [ ] Mobile app for alert notifications
- [ ] Face recognition integration
- [ ] Automated report generation (PDF/HTML)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with YOLOv8 · LangGraph · Groq · Streamlit**

</div>
