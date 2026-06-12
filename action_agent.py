"""
Action Agent Node
=================
Performs alert logging, console notification, and triggers downstream actions
when anomalies are confirmed by LLM reasoning.
"""

from typing import Dict, Any
import json
import uuid
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LOG_FILE, LOG_DIR
from core.state import SurveillanceState

def action_node(state: SurveillanceState) -> Dict[str, Any]:
    """
    LangGraph node that logs detected anomalies and LLM analysis to file and console.
    
    Args:
        state: Current surveillance state
        
    Returns:
        State update dictionary containing the newly generated alerts list.
    """
    anomalies = state.get("anomalies", [])
    llm_analysis = state.get("llm_analysis", "")
    timestamp = state.get("timestamp", "")
    
    if not anomalies:
        return {"alerts": []}
        
    # Create the logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    alerts = []
    
    # Parse recommended actions and severity from LLM response if possible
    # We will also default to anomaly's severity
    primary_severity = anomalies[0]["severity"] if anomalies else "MEDIUM"
    if "Threat Level: HIGH" in llm_analysis:
        primary_severity = "HIGH"
    elif "Threat Level: CRITICAL" in llm_analysis:
        primary_severity = "CRITICAL"
    elif "Threat Level: LOW" in llm_analysis:
        primary_severity = "LOW"
        
    for anomaly in anomalies:
        alert_id = str(uuid.uuid4())[:8]
        
        # Build alert object
        alert = {
            "alert_id": alert_id,
            "anomaly": anomaly,
            "llm_analysis": llm_analysis,
            "recommended_action": "Refer to Action Items in LLM Analysis",
            "timestamp": timestamp or datetime.now().isoformat()
        }
        alerts.append(alert)
        
        # 1. Print formatted notification to console
        print_console_alert(alert, primary_severity)
        
        # 2. Append event to JSON lines log file for history/dashboard
        log_to_file(alert)
        
    return {"alerts": alerts}

def print_console_alert(alert: dict, severity: str):
    """Prints a beautiful colored terminal warning."""
    try:
        from colorama import init, Fore, Style
        init(autoreset=True)
        
        color = Fore.YELLOW
        if severity == "HIGH":
            color = Fore.RED
        elif severity == "CRITICAL":
            color = Fore.RED + Style.BRIGHT
        elif severity == "LOW":
            color = Fore.BLUE
            
        print("\n" + "="*60)
        print(f"{color}[ALERT {alert['alert_id']} - {severity} SEVERITY]{Style.RESET_ALL}")
        print(f"Time:      {alert['timestamp']}")
        print(f"Type:      {alert['anomaly']['anomaly_type']}")
        print(f"Details:   {alert['anomaly']['description']}")
        print("-"*60)
        print("🤖 LLM Reasoning & Assessment:")
        print(alert["llm_analysis"])
        print("="*60 + "\n")
    except ImportError:
        # Fallback if colorama is not available
        print(f"\n[ALERT {alert['alert_id']} - {severity}]")
        print(f"Time: {alert['timestamp']}")
        print(f"Anomaly: {alert['anomaly']['description']}")
        print(f"LLM Assessment:\n{alert['llm_analysis']}\n")

def log_to_file(alert: dict):
    """Append a single alert JSON object to the log file."""
    try:
        # We strip non-serializable fields (like raw frames/bytes)
        clean_alert = {
            "alert_id": alert["alert_id"],
            "anomaly": {
                "anomaly_type": alert["anomaly"]["anomaly_type"],
                "severity": alert["anomaly"]["severity"],
                "description": alert["anomaly"]["description"],
                "location": alert["anomaly"]["location"],
                "frame_id": alert["anomaly"]["frame_id"],
                "timestamp": alert["anomaly"]["timestamp"]
            },
            "llm_analysis": alert["llm_analysis"],
            "timestamp": alert["timestamp"]
        }
        
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean_alert) + "\n")
    except Exception as e:
        print(f"Error logging alert to file: {e}")
