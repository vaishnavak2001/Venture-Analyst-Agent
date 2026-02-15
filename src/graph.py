"""
graph.py - The LangGraph State Machine for the Venture Analyst.
This connects all the three agents into a single pipeline:
    PDF Text ->[Extractor] ->[Verifier] ->[Analyst] ->Report

"""
from langgraph.graph  import StateGraph, END
from src.state import VentureAnalystState
from src.agents import extract_claims, verify_claims, generate_analysis

def build_graph():
    """
    Builds and compiles the LangGraph workflow.
    Flow:
        extract_claims -> verify_claims -> generate_analysis -> END
    """

    workflow = StateGraph(VentureAnalystState)
    workflow.add_node("extractor", extract_claims)
    workflow.add_node("verifier", verify_claims)
    workflow.add_node("analyst", generate_analysis)

    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "verifier")
    workflow.add_edge("verifier", "analyst")
    workflow.add_edge("analyst", END)

    app = workflow.compile()
    print("✅ LangGraph compiled successfully!")
    print("   Flow: Extractor → Verifier → Analyst → END")

    return app