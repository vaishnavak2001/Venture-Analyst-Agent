"""
app.py — The Venture Analyst Dashboard (Streamlit Frontend)

A "Glass Box" UI that shows the AI's reasoning at every step.
Run: streamlit run app.py
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*Examining the path.*")
import streamlit as st
import time
from src.utils import extract_text_from_upload
from src.graph import build_graph
from src.state import VentureAnalystState


# ============================================
# 📐 PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="The Venture Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================
# 🎨 CUSTOM STYLING
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-top: 0;
    }
    .risk-high {
        background-color: #FEE2E2;
        border-left: 4px solid #EF4444;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-medium {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-low {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .claim-card {
        background-color: #F9FAFB;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# 📊 SIDEBAR — Upload & Controls
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/analytics.png", width=80)
    st.markdown("## 📊 The Venture Analyst")
    st.markdown("*Autonomous Due Diligence Agent*")
    st.markdown("---")

    st.markdown("### 📁 Upload Pitch Deck")
    uploaded_file = st.file_uploader(
        "Drop a PDF pitch deck here",
        type=["pdf"],
        help="Upload a startup pitch deck (PDF) for AI-powered due diligence analysis."
    )

    st.markdown("---")
    st.markdown("### ⚙️ How It Works")
    st.markdown("""
    1. 📄 **Extract** — AI reads the pitch deck
    2. 🔍 **Verify** — Claims checked against web data
    3. ⚖️ **Analyze** — Risk score & Investment Memo
    """)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - 🧠 Llama 3.3 70B (Groq)
    - 🔗 LangGraph
    - 📚 FAISS + HuggingFace
    - 🔎 DuckDuckGo Search
    - ✅ Pydantic Validation
    """)


# ============================================
# 🏠 MAIN AREA
# ============================================

# Header
st.markdown('<p class="main-header">📊 The Venture Analyst</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Due Diligence — Upload a pitch deck and get an instant risk assessment.</p>', unsafe_allow_html=True)
st.markdown("---")


# --- State Management ---
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.final_state = None


# --- Main Logic ---
if uploaded_file is None:
    # No file uploaded — show landing page
    st.markdown("## 👋 Welcome!")
    st.markdown("""
    Upload a startup pitch deck (PDF) using the sidebar to get started.

    The Venture Analyst will:
    - **Extract** all verifiable claims from the deck
    - **Fact-check** each claim against live web data
    - **Score** the startup's risk level (0-100)
    - **Generate** a professional Investment Memo

    > 💡 *This is an autonomous AI agent — it searches the web, reasons about evidence, and forms its own conclusions.*
    """)

    # Show sample output
    with st.expander("📋 See a Sample Analysis"):
        st.markdown("""
        **Startup:** NovaTech AI  
        **Risk Score:** 73/100 (HIGH)  
        **Claims Found:** 12  
        **Red Flags:** 2 contradicted claims  
        **Recommendation:** Proceed with Caution  
        """)

else:
    # File uploaded — run the analysis
    if not st.session_state.analysis_complete:

        # Step 1: Extract PDF text
        with st.status("🚀 Running Due Diligence Analysis...", expanded=True) as status:

            st.write("📄 **Step 1/3:** Extracting text from pitch deck...")
            try:
                pdf_text = extract_text_from_upload(uploaded_file)
                st.write(f"   ✅ Extracted {len(pdf_text)} characters")
            except Exception as e:
                st.error(f"❌ Failed to read PDF: {e}")
                st.stop()

            # Step 2: Build graph and run
            st.write("🔧 **Building AI Pipeline...**")
            graph = build_graph()

            initial_state: VentureAnalystState = {
                "pdf_text": pdf_text,
                "extracted_claims": [],
                "verified_claims": [],
                "risk_score": 0,
                "final_report": "",
                "startup_name": None,
                "status": "Starting pipeline...",
                "errors": [],
            }

            st.write("🧠 **Step 2/3:** Extracting and verifying claims (this takes ~60 seconds)...")
            final_state = graph.invoke(initial_state)

            st.write("⚖️ **Step 3/3:** Generating Investment Memo...")
            time.sleep(0.5)

            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        # Store results
        st.session_state.analysis_complete = True
        st.session_state.final_state = final_state

    # ============================================
    # 📊 DISPLAY RESULTS
    # ============================================
    final_state = st.session_state.final_state

    if final_state:
        startup_name = final_state.get("startup_name", "Unknown Startup")
        risk_score = final_state.get("risk_score", 50)
        verified_claims = final_state.get("verified_claims", [])

        # Flatten claims if needed
        flat_claims = []
        for item in verified_claims:
            if isinstance(item, list):
                for sub in item:
                    flat_claims.append(sub)
            elif isinstance(item, dict):
                flat_claims.append(item)

        # --- Risk Score Header ---
        st.markdown(f"## 🏢 {startup_name}")

        col1, col2, col3, col4 = st.columns(4)

        # Risk Score with color
        risk_color = "🔴" if risk_score >= 65 else "🟡" if risk_score >= 35 else "🟢"
        risk_label = "HIGH" if risk_score >= 65 else "MEDIUM" if risk_score >= 35 else "LOW"

        col1.metric(
            label="Risk Score",
            value=f"{risk_score}/100",
            delta=risk_label,
            delta_color="inverse" if risk_score >= 65 else "normal",
        )

        supported = sum(1 for v in flat_claims if v.get("verdict") == "Supported")
        contradicted = sum(1 for v in flat_claims if v.get("verdict") == "Contradicted")
        inconclusive = sum(1 for v in flat_claims if v.get("verdict") == "Inconclusive")

        col2.metric("✅ Supported", supported)
        col3.metric("❌ Contradicted", contradicted)
        col4.metric("❓ Inconclusive", inconclusive)

        st.markdown("---")

        # --- Tabs for different views ---
        tab1, tab2, tab3 = st.tabs(["📋 Investment Memo", "🔍 Claim Details", "📄 Raw Data"])

        # --- Tab 1: Investment Memo ---
        with tab1:
            report = final_state.get("final_report", "No report generated.")
            st.markdown(report)

        # --- Tab 2: Claim-by-Claim Details ---
        with tab2:
            st.markdown("### Verified Claims")

            for i, vc in enumerate(flat_claims, 1):
                verdict = vc.get("verdict", "Unknown")
                emoji = {"Supported": "✅", "Contradicted": "❌", "Inconclusive": "❓"}.get(verdict, "❓")

                with st.expander(f"{emoji} Claim {i}: [{vc.get('category', 'N/A')}] {vc.get('claim_text', '')[:80]}..."):
                    st.markdown(f"**Claim:** {vc.get('claim_text', 'N/A')}")
                    st.markdown(f"**Source:** {vc.get('source_page', 'N/A')}")
                    st.markdown(f"**Category:** {vc.get('category', 'N/A')}")
                    st.markdown(f"**Verdict:** {emoji} **{verdict}**")
                    st.markdown(f"**Reasoning:** {vc.get('reasoning', 'N/A')}")

                    st.markdown("---")
                    st.markdown("**🔎 Raw Search Results:**")
                    st.code(vc.get("search_results", "No search results"), language=None)

        # --- Tab 3: Raw Data ---
        with tab3:
            st.markdown("### Raw Pipeline Data")
            st.markdown("*This is the 'Glass Box' — full transparency into the AI's process.*")

            with st.expander("📄 Extracted PDF Text"):
                st.text(final_state.get("pdf_text", "")[:3000] + "...")

            with st.expander("📋 Extracted Claims (JSON)"):
                st.json(final_state.get("extracted_claims", []))

            with st.expander("🔍 Verified Claims (JSON)"):
                st.json(flat_claims)

            with st.expander("⚠️ Errors"):
                errors = final_state.get("errors", [])
                if errors:
                    for err in errors:
                        st.error(err)
                else:
                    st.success("No errors during pipeline execution!")

        # --- Reset Button ---
        st.markdown("---")
        if st.button("🔄 Analyze Another Pitch Deck"):
            st.session_state.analysis_complete = False
            st.session_state.final_state = None
            st.rerun()