"""
state.py -The Single Source of Truth for Venture Analyst Agent.

Every node in the LangGraph reads from and writes to this shared state.
Think of it as the "database schema" for the entire pipeline.
"""
from typing import TypedDict, List, Optional
class Claim(TypedDict):
    """
    A single claim extractes from the pitch deck.
    """
    claim_text: str
    source_page: str
    category: str

class VerifiedClaim(Claim):
    """
    A claim after it has been fact-checked agrainst web data.
    """
    claim_text: str
    source_page: str
    category : str
    search_query: str
    search_results: str
    verdict: str
    reasoning: str
class VentureAnalystState(TypedDict):
    """
    The Global state object for the entire Langgraph pipeline.
    Flow: PDF upload -> Extraction -> Verfication -> Analysis -> Report
    """
    pdf_text : str
    extracted_claims: List[Claim]
    verified_claims: List[VerifiedClaim]
    risk_score : int
    final_report :str

    startup_name : Optional[str]
    status: str
    error:List[str]
    