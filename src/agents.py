"""
agent.py - The AI Agents for The Venture Analyst.
contains:
1. ClaimsExtractor - Extracts structured claims from pitch deck text
2. ClaimsVerifier - (Phase 3)
3. InvestmentAnalyst - (Phase 4)
"""
import os
import json
from dotenv import load_dotenv
from typing  import List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun


from pydantic import BaseModel, Field

from src.state import VentureAnalystState,Claim
from src.utils import chunk_text

load_dotenv()

# llm & embedding setup
def get_llm(temperature:float = 0.0):
    """
    Intialize the Groq LLM (Llama-3.3-70b).
    """
    return ChatGroq(
        model="Llama-3.3-70b-versatile",
        temperature=temperature,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        
    )
def get_embeddings():
    """
    Initialize the HuggingFace embedding for FAISS.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#vector Store builder
def build_vector_store(pdf_text:str) -> FAISS:
    """
    Takes cleaned PDF text,chunks it, and loads it into FAISS.
    Args:
        pdf_text: The full cleaned text extracted from the pitch deck PDF.
    Returns:
        A FAISS vector store ready for similarity search.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ",""]
    )

    chunks = splitter.split_text(pdf_text)

    print(f"Created {len(chunks)} chunks from pitch deck")

    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    print(f" FAISS vector store built successfully")
    return vector_store

# Pydantic models for structured output
class ExtractedClaim(BaseModel):
    """ 
    Schema for a single Extracted Claim.
    """
    claim_text: str = Field(
        description="The exact claim or assertion made in the pitch deck"
    )
    
    source_page: str = Field(
        description="The page number or section where the claim was found, e.g.,'Page 3'"
    )
    category: str = Field(
        description="The category of the claim. Must be one of: Market Size, Traction, Revenue, Team, Technology, Competition, Financials, Other"
    )
class ClaimsExtractionResult(BaseModel):
    """
    Schema for the full extraction result 
    """
    startup_name: str = Field(
        description="The name of the startup from the pitch deck"
    )
    claims: List[ExtractedClaim] = Field(
        description="A list of all verifiable claims found in the pitch deck"
    )

    # the claims extractor agent

EXTRACTOR_PROMPT = """
You are a senior Venture Capital Analyst performing due diligence on a startup pitch deck. 
Your job is to extract ALL verifiable claims from th following pitch deck text. A "verifiable claim" is any statement that can be fact-checked against external data.

Focus on these categories:
1. **Market Size**- Claims about TAM, SAM, SOM, market growth rates
2. **Traction** - User numbers, growth rates,customer counts
3. **Revenue** - Revenue figures, projections, unit Economics
4. **Team** - Founder backgrounds,previous exits, experience claims
5. **Technology** - Patent claims,technical advantages, unique capabilities
6. **Competition** - Claims about competitors, market position, differentiation
7. **Financials** - Funding raised, runway claims, burn rate

RULES:
- Extract the EXACT claim as stated in the text
- Note which page the claim came from (look for "--- PAGE X ---" markers)
- if no page marker is nearby,write "Unkown"
- Only Extract claims that are VERIFIABLE (skip opinions and vague statements)
- Extract between 5-15 claims (focus on the most important ones)

PITCH DECK TEXT:
{context}

Respond with valid JSON matching this exact format:
{{
    "startup_name": "Name of the startup",
    "claims": [
        {{
            "claim_text": "The exact claim",
            "source_page": "Page X",
            "category": "Category name"
        }}
    ]
}}
Return ONLY the JSON. No other text.

"""
def extract_claims(state:VentureAnalystState) -> dict:
    """
    The claims Extractor Agent - Node for LangGraph.

    Reads: state["pdf_text"]
    Writes:state ["extracted_claims"], state["startup_name"],state["status"], state["error"]

    """
    print("\n CLAMS EXTRACTOR AGENT - Starting...")
    print("="*50)

    pdf_text = state["pdf_text"]
    if not pdf_text:
        return{
            "extracted_claims": [],
            "startup_name": "Unknown",
            "status": "Error: No PDF text found",
            "errors": state.get("errors",[])+["No PDF provided to extractor"]
        }
    
    print(("Building vector store from PDF text..."))
    vector_store= build_vector_store(pdf_text)

    print("Retrieving relavant sections...")
    search_queries=[
        "market size TAM revenue growth",
        "traction user customer growth",
        "team founder experience background",
        "competition competitive advantage",
        "financials funding revenue projections",
        "technology product innovation"

    ]
    all_retrieved_chunks = []
    seen_chunks = set()
    for query in search_queries:
        results = vector_store.similarity_search(query, k=3)
        for doc in results:
            if doc.page_content not in seen_chunks:
                all_retrieved_chunks.append(doc.page_content)
                seen_chunks.add(doc.page_content)
    context = "\n\n---\n\n".join(all_retrieved_chunks)
    print(f"Retrieved {len(all_retrieved_chunks)} unique relavant sections")

    print("\n Extracting claims with LLM...")
    llm= get_llm(temperature=0.0)
    prompt=ChatPromptTemplate.from_template(EXTRACTOR_PROMPT)
    chain=prompt | llm
    response = chain.invoke({"context": context})

    print("\n Parsing stuctured output...")
    try:
        raw_text = response.content.strip()

        first_brace = raw_text.find("{")
        last_brace = raw_text.rfind("}")

        if first_brace != -1 and last_brace != -1:
            json_string = raw_text[first_brace:last_brace + 1]
        else:
            json_string = raw_text

        print(f"   📝 Extracted JSON string ({len(json_string)} chars)")


        parsed = ClaimsExtractionResult.model_validate_json(json_string)
        claims_list = []
        for c in parsed.claims:
            single_claim=[
                {
                    "claim_text": c.claim_text,
                    "source_page": c.source_page,
                    "category": c.category
                }
            ]
            claims_list.append(single_claim)
        print(f"\n Successfully extracted {len(claims_list)} claims!")
        print(f" Startup Name: {parsed.startup_name}")
        for i,item in enumerate(claims_list,1):
            print(f"{i}.[{item[0]['category']}] {item[0]['claim_text'][:80]}...")
        return {
            "extracted_claims": claims_list,
            "startup_name": parsed.startup_name,
            "status": "Claims Extracted successfully",
            "errors": state.get("errors",[]),
        }
    except Exception as e:
        print(f"\n Error parsing LLM response: {e}")
        print(f"Raw response:{response.content[:500]}")

        return {
            "extracted_claims": [],
            "startup_name": "Unknown",
            "status": f"Error parsing claims: {str(e)}",
            "errors": state.get("errors",[])+[f"Extraction parsing error: {str(e)}"],
        }


# Initialize the search tool
search_tool = DuckDuckGoSearchRun()


VERIFIER_PROMPT = """You are a senior due diligence analyst fact-checking claims from a startup pitch deck.

You have been given:
1. A CLAIM from the pitch deck
2. SEARCH RESULTS from the web

Your job is to determine whether the search results SUPPORT, CONTRADICT, or are INCONCLUSIVE about the claim.

CLAIM:
{claim_text}

CLAIM CATEGORY: {category}

SEARCH RESULTS:
{search_results}

INSTRUCTIONS:
1. Compare the claim against the search results carefully
2. Look for specific numbers, dates, names, and facts
3. Consider that startup-specific claims (like ARR, customer count) may not be publicly verifiable
4. Be skeptical but fair

You MUST respond with valid JSON in this EXACT format:
{{
    "verdict": "Supported OR Contradicted OR Inconclusive",
    "reasoning": "2-3 sentences explaining your verdict with specific evidence from the search results"
}}

Return ONLY the JSON. No other text."""


def search_for_claim(claim_text: str, category: str) -> str:
    """
    Formulates a smart search query based on the claim and searches the web.

    Args:
        claim_text: The claim to verify.
        category: The category of the claim (helps refine the search).

    Returns:
        Search results as a string.
    """
    # Build a focused search query based on category
    if category in ["Market Size"]:
        # For market claims, search for the specific market and numbers
        query = claim_text[:150]  # Use the claim itself as the query
    elif category in ["Team"]:
        # For team claims, search for the person's name and company
        # Extract the person's name (usually at the start)
        query = claim_text[:100] + " LinkedIn"
    elif category in ["Technology"]:
        query = claim_text[:120] + " patent"
    elif category in ["Financials"]:
        query = claim_text[:120] + " funding announcement"
    elif category in ["Traction", "Revenue"]:
        query = claim_text[:120]
    else:
        query = claim_text[:150]

    print(f"      🔎 Searching: {query[:80]}...")

    try:
        results = search_tool.run(query)
        return results if results else "No results found."
    except Exception as e:
        print(f"      ⚠️ Search error: {e}")
        return f"Search failed: {str(e)}"


def verify_single_claim(claim_data: dict) -> dict:
    """
    Verifies a single claim against web search results.

    Args:
        claim_data: A dict with claim_text, source_page, category.

    Returns:
        A VerifiedClaim dict with verdict and reasoning.
    """
    claim_text = claim_data["claim_text"]
    category = claim_data["category"]
    source_page = claim_data["source_page"]

    # Step 1: Search the web
    search_results = search_for_claim(claim_text, category)

    # Step 2: Ask the LLM to analyze
    llm = get_llm(temperature=0.0)
    prompt = ChatPromptTemplate.from_template(VERIFIER_PROMPT)
    chain = prompt | llm

    response = chain.invoke({
        "claim_text": claim_text,
        "category": category,
        "search_results": search_results[:3000],  # Limit to avoid token overflow
    })

    # Step 3: Parse the verdict
    try:
        raw_text = response.content.strip()

        # Extract JSON robustly
        first_brace = raw_text.find("{")
        last_brace = raw_text.rfind("}")

        if first_brace != -1 and last_brace != -1:
            json_string = raw_text[first_brace:last_brace + 1]
        else:
            json_string = raw_text

        verdict_data = json.loads(json_string)

        return {
            "claim_text": claim_text,
            "source_page": source_page,
            "category": category,
            "search_query": claim_text[:100],
            "search_results": search_results[:1000],  # Store truncated results
            "verdict": verdict_data.get("verdict", "Inconclusive"),
            "reasoning": verdict_data.get("reasoning", "Could not parse reasoning."),
        }

    except Exception as e:
        print(f"      ⚠️ Parse error for claim: {e}")
        return {
            "claim_text": claim_text,
            "source_page": source_page,
            "category": category,
            "search_query": claim_text[:100],
            "search_results": search_results[:1000],
            "verdict": "Inconclusive",
            "reasoning": f"Error during verification: {str(e)}",
        }


def verify_claims(state: VentureAnalystState) -> dict:
    """
    The Verifier Agent — Node for LangGraph.

    Reads: state["extracted_claims"]
    Writes: state["verified_claims"], state["status"]
    """
    print("\n🕵️ FACT-CHECKER AGENT — Starting...")
    print("=" * 50)

    raw_claims = state.get("extracted_claims", [])

    # --- Flatten if claims are nested lists ---
    extracted_claims = []
    for item in raw_claims:
        if isinstance(item, list):
            # It's a nested list — unwrap it
            for sub_item in item:
                extracted_claims.append(sub_item)
        elif isinstance(item, dict):
            # It's already a proper dict
            extracted_claims.append(item)

    print(f"   📋 Found {len(extracted_claims)} claims to verify")

    if not extracted_claims:
        return {
            "verified_claims": [],
            "status": "Error: No claims to verify",
            "errors": state.get("errors", []) + ["No claims provided to verifier"],
        }

    verified_claims = []
    total = len(extracted_claims)

    for i, entry in enumerate(extracted_claims, 1):
        print(f"\n   📌 Verifying claim {i}/{total}: [{entry['category']}]")
        print(f"      \"{entry['claim_text'][:80]}...\"")

        verified = verify_single_claim(entry)
        verified_claims.append(verified)

        # Print the verdict
        verdict_emoji = {
            "Supported": "✅",
            "Contradicted": "❌",
            "Inconclusive": "❓",
        }
        emoji = verdict_emoji.get(verified["verdict"], "❓")
        print(f"      {emoji} Verdict: {verified['verdict']}")
        print(f"      💬 {verified['reasoning'][:100]}...")

    # Summary
    supported = sum(1 for v in verified_claims if v["verdict"] == "Supported")
    contradicted = sum(1 for v in verified_claims if v["verdict"] == "Contradicted")
    inconclusive = sum(1 for v in verified_claims if v["verdict"] == "Inconclusive")

    print(f"\n{'=' * 50}")
    print(f"📊 VERIFICATION SUMMARY")
    print(f"   ✅ Supported: {supported}")
    print(f"   ❌ Contradicted: {contradicted}")
    print(f"   ❓ Inconclusive: {inconclusive}")
    print(f"{'=' * 50}")

    return {
        "verified_claims": verified_claims,
        "status": f"Verification complete: {supported} supported, {contradicted} contradicted, {inconclusive} inconclusive",
        "errors": state.get("errors", []),
    }

# the investment analyst agent
ANALYST_PROMPT =""" You  are a senior Analyst  at a top-tier VC firm.
You have been given the results of a due diligence investigation on a startup pitch deck. Your job is to:
1. Analyze all verfied claims
2. Calculate a RISK SCORE (0-100, where 0 =No Risk, 100=Extremely Risky)
3. Write a professional investment Memo
HERE IS THE DUE DILIGENCE DATA:
STARTU NAME: {startup_name}
VERFIED CLAIMS:
{verified_claims_text}
SUMMARY STATISTICS:
- Total Claims Analyzed: {total_claims}
- Supported by External Data:{supported_claims}
- Contradicted by External Data:{contradicted_claims}
- Inconclusive (Not verifiable): {inconclusive_claims}

SCORING RUBRIC:
- Each CONTRADICTED claims adds 15-25 points to the risk score (depending on severity)
- Each INCONCLUSIVE claims adds 3-7 points(lack of verifiability is itself a risk)
- Each SUPPORTED claim reduces risk by 5-10 points (verfied claims reduce risk)
- Base risk score  starts at 50(neutral)
- Final score must be between 0 and 100

INSTRUCTIONS:
Generate a professional investment memo in the following EXACT format.Use Markdown.
You Must respond with valid JSON in this EXACT format:
{{
    "risk_score": <integer between 0-100>,
    "investment_memo": "<full markdown memo as a single string>"
}}
The investment_memo should follow this structure:
# Investment Memo: [startup_name]
## Executive Summary
(2-3 sentences)
## Risk Score:X/100
(1 sentence interpretation)
## Verified Claims Analysis
### Supported Claims
(List each with explanation)
### Contradicted Claims(RED FLAGS)
(List each with explanation - these are the most important)
### Inconclusive Claims
(List each with brief note)
## Key Risk Factors
(Numbered list of top 3-5 risks )
## Investment Recommendation
(PASS / PROCEED WITH CAUTION/STRONG CONSIDERATION)
(2-3 sentances explaining the recommendation)
Return ONLY the JSON. No other text.

"""
class AnalystResult(BaseModel):
    """Schema for Analyst's output."""
    risk_score: int = Field(
        description="Risk score from 0-100"
    )
    investment_memo: str = Field(
        description="The full investment memo in markdown format"
    )
def format_verified_claims_for_prompt(verfied_claims: list) ->str:
    """
    Formats the verified claims into a readable string fr the LLM prompt.
    """
    formatted = ""
    for i, vc in enumerate(verfied_claims,1):
        verdict_emoji = {
            "Supported": "✅",
            "Contradicted": "❌",
            "Inconclusive": "❓",
        }
        emoji = verdict_emoji.get(vc["verdict"], "❓")
        formatted += f"\n{i}. {emoji} [{vc['category']}] {vc['verdict']}\n "
        formatted += f"   Claim: {vc['claim_text']}\n"
        formatted += f"   Source: {vc['source_page']}\n"
        formatted += f"   Reasoning: {vc['reasoning']}\n"
    return formatted
def generate_analysis(state: VentureAnalystState) -> dict:
    """
    The Investment Analyst Agent - Node for LangGraph.

    Reads: state["verified_claims"], state["startup_name"]
    Writes: state["risk_score"], state["final_report"], state["status"]
    """
    print("\n📈 INVESTMENT ANALYST AGENT - Starting...")
    print("=" * 50)

    verified_claims = state.get("verified_claims", [])
    startup_name = state.get("startup_name", "Unknown Startup")

    flat_claims = []
    for item in verified_claims:
        if isinstance(item, list):
            for sub_item in item:
                flat_claims.append(sub_item)
        elif isinstance(item, dict):
            flat_claims.append(item)
    if not flat_claims:
        return {
            "risk_score": 50,
            "final_report": "# Error \nNo verified claims available for analysis.",
            "status": "Error: No verified claims found",
            "errors": state.get("errors", []) + ["No verified claims for analyst"],
        }

    
    supported_count = sum(1 for v in flat_claims if v["verdict"] == "Supported")
    contradicted_count = sum(1 for v in flat_claims if v["verdict"] == "Contradicted")
    inconclusive_count = sum(1 for v in flat_claims if v["verdict"] == "Inconclusive")
    total_claims = len(flat_claims)

    print(f"   📊 Analyzing {total_claims} verified claims...")
    print(f"      ✅ Supported: {supported_count}")
    print(f"      ❌ Contradicted: {contradicted_count}")
    print(f"      ❓ Inconclusive: {inconclusive_count}")
    
    verified_claims_text = format_verified_claims_for_prompt(verified_claims)

    print("\n   🤖 Generating Investment Memo...")
    # Prepare the prompt
    llm = get_llm(temperature=0.2)
    prompt = ChatPromptTemplate.from_template(ANALYST_PROMPT)
    chain = prompt | llm

    response = chain.invoke({
        "startup_name": startup_name,
        "verified_claims_text": verified_claims_text,
        "total_claims": total_claims,
        "supported_claims": supported_count,
        "contradicted_claims": contradicted_count,
        "inconclusive_claims": inconclusive_count,
    })

    # Parse the analyst's output
    try:
        raw_text = response.content.strip()

        first_brace = raw_text.find("{")
        last_brace = raw_text.rfind("}")

        if first_brace != -1 and last_brace != -1:
            json_string = raw_text[first_brace:last_brace + 1]
        else:
            json_string = raw_text
        parsed = AnalystResult.model_validate_json(json_string)
        risk_score = max(0, min(100, parsed.risk_score))  # Ensure between 0-100
        print(f"\n   ✅ Analysis complete!")
        print(f"   📊 Risk Score: {risk_score}/100")

        risk_level = "LOW" if risk_score < 35 else "MEDIUM" if risk_score < 65 else "HIGH"
        print(f"   ⚠️ Risk Level: {risk_level}")
        return {
            "risk_score": risk_score,
            "final_report": parsed.investment_memo,
            "status": f"Analysis complete - Risk Score: {risk_score}/100 ({risk_level})",
            "errors": state.get("errors", []),
        }
    except Exception as e:
        print(f"\n   ⚠️ Error parsing analyst output: {e}")
        print(f"   Raw response: {response.content[:500]}")

        return {
            "risk_score": 50,
            "final_report": f"# Error \nFailed to generate investment memo\n\nError:{str(e)}\n\nRaw LLM output:\n{response.content[:1000]}",
            "status": f"Error during analysis: {str(e)}",
            "errors": state.get("errors", []) + [f"Analyst parsing error: {str(e)}"],
        }