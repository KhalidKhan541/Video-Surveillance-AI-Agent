"""
Reasoning Agent Node (Groq LLM)
================================
Uses Groq to reason about the detected anomalies and evaluate severity.
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GROQ_API_KEY, GROQ_MODEL
from core.state import SurveillanceState

# Initialize LLM lazily or fallback to mock if API key is not configured
_llm = None

def get_llm():
    """Initialize the ChatGroq model if API key is present, otherwise return None."""
    global _llm
    if _llm is not None:
        return _llm
        
    # Check if API key is default or placeholder
    if not GROQ_API_KEY or GROQ_API_KEY.strip() == "" or "your_groq_api_key" in GROQ_API_KEY.lower():
        print("⚠️  Groq API Key is not set or is using placeholder. Using local simulation mode.")
        return None
        
    try:
        from langchain_groq import ChatGroq
        # Pass api_key explicitly or rely on GROQ_API_KEY env var
        _llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)
        return _llm
    except Exception as e:
        print(f"⚠️  Error initializing ChatGroq: {e}. Falling back to simulation mode.")
        return None

def reasoning_node(state: SurveillanceState) -> Dict[str, Any]:
    """
    LangGraph node that uses Groq LLM to assess severity and details of anomalies.
    
    Args:
        state: Current surveillance state
        
    Returns:
        State update dictionary containing llm_analysis and updated messages.
    """
    anomalies = state.get("anomalies", [])
    if not anomalies:
        return {
            "llm_analysis": "No anomalies detected.",
            "messages": []
        }
        
    # Format description of all anomalies for the LLM prompt
    anomaly_summaries = []
    for idx, anomaly in enumerate(anomalies):
        summary = (
            f"Anomaly #{idx + 1}:\n"
            f"- Type: {anomaly['anomaly_type']}\n"
            f"- Reported Severity: {anomaly['severity']}\n"
            f"- Description: {anomaly['description']}\n"
            f"- Involved Objects: {anomaly['involved_objects']}\n"
            f"- Location Coordinates: {anomaly['location']}\n"
            f"- Timestamp: {anomaly['timestamp']}\n"
        )
        anomaly_summaries.append(summary)
        
    anomaly_text = "\n".join(anomaly_summaries)
    
    # Structure the prompt
    prompt = (
        "You are an expert AI video surveillance analyst. You monitor CCTV systems, "
        "detect security incidents, evaluate threats, and recommend immediate response actions.\n\n"
        "Here are the details of anomalies flagged by the Vision/Rule-based engine:\n"
        f"{anomaly_text}\n"
        "Please provide a comprehensive assessment in the following format:\n"
        "1. **Threat Level**: [LOW | MEDIUM | HIGH | CRITICAL]\n"
        "2. **Incident Summary**: (A brief 1-2 sentence description of what is happening)\n"
        "3. **Contextual Analysis**: (Reason why this may or may not be a true security concern. "
        "For example, multiple people gathering might be a line, a group meeting, or an unauthorized protest. "
        "A loitering person might be waiting for someone, or observing windows. Evaluate based on numbers and timings.)\n"
        "4. **Action Items**: (List 2-3 specific, actionable recommendations for security personnel, "
        "e.g., dispatch guards, monitor closely, sound alarm, ignore as false positive)\n"
    )
    
    llm = get_llm()
    
    if llm is None:
        # Generate a high-quality simulated reasoning response based on the anomaly details
        simulated_response = generate_simulated_reasoning(anomalies)
        analysis_text = simulated_response
    else:
        try:
            # Invoke ChatGroq LLM
            response = llm.invoke([HumanMessage(content=prompt)])
            analysis_text = response.content
        except Exception as e:
            print(f"❌ Groq API invocation failed: {e}")
            analysis_text = (
                f"❌ Error communicating with Groq API ({e}).\n"
                f"Please verify your GROQ_API_KEY in the .env file.\n\n"
                f"--- SIMULATED LOCAL ANALYSIS ---\n"
                f"{generate_simulated_reasoning(anomalies)}"
            )
            
    # Return updates including messages (for memory/chat context if desired)
    new_message = AIMessage(content=analysis_text)
    
    return {
        "llm_analysis": analysis_text,
        "messages": [HumanMessage(content=prompt), new_message]
    }

def generate_simulated_reasoning(anomalies: list[dict]) -> str:
    """Generates a realistic LLM-style assessment when no API key is available."""
    if not anomalies:
        return "No threats detected."
        
    primary_anomaly = anomalies[0]
    atype = primary_anomaly["anomaly_type"]
    desc = primary_anomaly["description"]
    severity = primary_anomaly["severity"]
    
    threat_level = severity
    
    if atype == "crowd_formation":
        summary = f"Simulated analysis: Detection of rapid group gathering. {desc}."
        analysis = (
            "A sudden gathering of individuals detected in the monitored frame. "
            "This could represent a crowd forming at an entrance, a group meeting, "
            "or a potential disturbance. In security monitoring, group clustering is "
            "flagged to ensure pedestrian egress pathways are not blocked and to monitor "
            "for escalations."
        )
        actions = (
            "- Monitor CCTV stream closely for next 5 minutes to evaluate crowd behavior.\n"
            "- Inform gate/entry staff to ensure doorways are clear.\n"
            "- If count continues to rise, dispatch a security patrol to verify the reason."
        )
    elif atype == "loitering":
        summary = f"Simulated analysis: Persistent object presence in tracking zone. {desc}."
        analysis = (
            "A tracked object has remained stationary beyond the acceptable dwell threshold. "
            "Loitering is a common precursor to tailgating, unauthorized entry, or vandalism. "
            "The subject may simply be waiting for transit or a colleague, but persistent static "
            "positioning warrants verification of intent."
        )
        actions = (
            "- Direct camera focus on the target subject.\n"
            "- If subject behavior is suspicious (checking doors/windows/locks), dispatch floor guard.\n"
            "- Log incident for patrol shift crossover."
        )
    elif atype == "speed_violation":
        summary = f"Simulated analysis: High-velocity movement detected. {desc}."
        analysis = (
            "An object is moving at speed exceeding regular pedestrian velocity. "
            "This could be a running person (indicating emergency, theft, panic, or fitness) "
            "or a fast-moving vehicle. Sudden acceleration in indoor or pedestrian zones "
            "usually flags immediate response."
        )
        actions = (
            "- Trace the direction of running to check if they are fleeing an incident.\n"
            "- Check adjacent cameras for exit vectors.\n"
            "- Alert duty sergeant of possible running/fleeing suspect."
        )
    elif atype == "zone_intrusion":
        threat_level = "CRITICAL"
        summary = f"Simulated analysis: Restricted perimeter breached. {desc}."
        analysis = (
            "An object has crossed the coordinates mapping to a defined restricted boundary. "
            "This represents an immediate security breach of a controlled zone. "
            "There is no baseline rationale for authorized presence at this location/time."
        )
        actions = (
            "- Immediate dispatch of nearest security unit to intercept and identify.\n"
            "- Trigger audible alert/warning speaker if available on-site.\n"
            "- Log timestamp and capture object trajectory for report."
        )
    elif atype == "new_large_object":
        summary = f"Simulated analysis: Sudden appearance of large object. {desc}."
        analysis = (
            "A new object of substantial dimensions has appeared in the scene and is static. "
            "This could indicate an abandoned bag, luggage, or an unauthorized vehicle. "
            "Abandoned items in public zones require systematic evaluation for safety protocols."
        )
        actions = (
            "- Review recording history to identify who deposited the item.\n"
            "- Inspect object visually for hazards via high-definition zoom.\n"
            "- Prepare public announcement/inquiry if item is left in common areas."
        )
    else:
        summary = f"Simulated analysis: Unspecified anomaly. {desc}."
        analysis = "General unusual event detected by rule-based sensor suite. Requires review."
        actions = "- Check the live stream.\n- Check tracking logs."

    response = (
        f"1. **Threat Level**: {threat_level}\n"
        f"2. **Incident Summary**: {summary}\n"
        f"3. **Contextual Analysis (SIMULATED)**: {analysis}\n"
        f"4. **Action Items**:\n{actions}"
    )
    return response
