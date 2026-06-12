"""
LangGraph StateGraph Configuration
==================================
Wires up the multi-agent pipeline using LangGraph StateGraph.
"""

from langgraph.graph import StateGraph, START, END

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state import SurveillanceState
from agents.vision_agent import vision_node
from agents.anomaly_agent import anomaly_node
from agents.reasoning_agent import reasoning_node
from agents.action_agent import action_node

def route_on_anomaly(state: SurveillanceState) -> str:
    """
    Conditional router edge: directs flow to reasoning node if anomalies
    exist, otherwise jumps directly to the end.
    
    Args:
        state: Current surveillance state
        
    Returns:
        "reasoning" if anomaly found, otherwise "end"
    """
    if state.get("has_anomaly", False):
        return "reasoning"
    return "end"

def create_surveillance_graph():
    """
    Assemble the surveillance StateGraph.
    
    Vision Agent ──> Anomaly Agent ──(anomaly?)──> Reasoning Agent ──> Action Agent
                                     └──(no?)────> [END]
    """
    builder = StateGraph(SurveillanceState)
    
    # Add agent nodes
    builder.add_node("vision", vision_node)
    builder.add_node("anomaly", anomaly_node)
    builder.add_node("reasoning", reasoning_node)
    builder.add_node("action", action_node)
    
    # Define primary edges
    builder.add_edge(START, "vision")
    builder.add_edge("vision", "anomaly")
    
    # Define conditional routing from anomaly detection
    builder.add_conditional_edges(
        "anomaly",
        route_on_anomaly,
        {
            "reasoning": "reasoning",
            "end": END
        }
    )
    
    # Complete reasoning-action path
    builder.add_edge("reasoning", "action")
    builder.add_edge("action", END)
    
    # Compile the graph
    graph = builder.compile()
    return graph

# Compiled graph instance for easy import
surveillance_graph = create_surveillance_graph()
