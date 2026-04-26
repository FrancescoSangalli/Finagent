"""Streamlit entry point for the FinAgent financial analysis system."""

import json
import os
import tempfile

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.documents import Document

from src.graph import build_graph
from src.rag.vectorstore import index_documents

st.set_page_config(
    page_title="FinAgent — Financial Analysis",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("FinAgent")
    st.caption("Multi-agent financial document analysis")
    st.divider()

    st.subheader("Target Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF(s) to analyse",
        type=["pdf"],
        accept_multiple_files=True,
        help="10-K, 10-Q, earnings reports, etc.",
    )

    st.divider()

    st.subheader("Competitor Documents (optional)")
    competitor_files = st.file_uploader(
        "Upload competitor PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key="competitor_uploader",
        help="These are indexed into the RAG vectorstore for comparison queries.",
    )

    index_btn = st.button(
        "Index Competitor Docs",
        disabled=not competitor_files,
        use_container_width=True,
    )

    if index_btn and competitor_files:
        with st.spinner("Indexing competitor documents…"):
            texts = []
            for f in competitor_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                try:
                    from src.utils.pdf_parser import parse_pdf
                    result = parse_pdf(tmp_path)
                    texts.append(result["text"])
                except Exception as e:
                    st.warning(f"Could not parse {f.name}: {e}")
                finally:
                    os.unlink(tmp_path)

            if texts:
                try:
                    index_documents(texts, collection_name="competitor_docs")
                    st.success(f"Indexed {len(texts)} competitor document(s).")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")

    st.divider()
    st.caption("Stack: LangGraph · Groq LLaMA 3.3 70B · ChromaDB · HuggingFace")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Financial Document Analysis")

query = st.text_input(
    "What would you like to know?",
    placeholder=(
        "e.g. 'Full analysis of this 10-K' · "
        "'Assess the risk profile' · "
        "'Compare against competitors' · "
        "'Summarise the financial performance'"
    ),
)

analyze_btn = st.button(
    "Analyze",
    type="primary",
    disabled=not (uploaded_files and query.strip()),
    use_container_width=False,
)

if analyze_btn:
    # Save uploaded PDFs to temp files and build Document objects
    documents: list[Document] = []
    tmp_paths: list[str] = []

    for f in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        tmp_paths.append(tmp_path)
        documents.append(
            Document(
                page_content="",          # filled by document_parser agent
                metadata={"source": tmp_path, "filename": f.name},
            )
        )

    initial_state = {
        "query": query.strip(),
        "documents": documents,
        "parsed_data": {},
        "financial_analysis": {},
        "risk_report": {},
        "competitor_analysis": {},
        "final_report": "",
        "agent_log": [],
        "active_agents": [],
        "error": None,
    }

    with st.spinner("Running analysis pipeline…"):
        try:
            result = st.session_state.graph.invoke(initial_state)
            st.session_state.result = result
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.session_state.result = None
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

# ---------------------------------------------------------------------------
# Output area
# ---------------------------------------------------------------------------
result = st.session_state.result

if result:
    if result.get("error"):
        st.error(f"Agent error: {result['error']}")

    tab_report, tab_data, tab_trace = st.tabs(["Report", "Data", "Agent Trace"])

    with tab_report:
        final_report = result.get("final_report", "")
        if final_report:
            st.markdown(final_report)
        else:
            st.info("No report generated.")

    with tab_data:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Financial Analysis")
            fa = result.get("financial_analysis")
            if fa:
                st.json(fa)
            else:
                st.info("No financial analysis data.")

        with col2:
            st.subheader("Risk Report")
            rr = result.get("risk_report")
            if rr:
                st.json(rr)
            else:
                st.info("No risk report data.")

        if result.get("competitor_analysis"):
            st.subheader("Competitor Analysis")
            st.json(result["competitor_analysis"])

    with tab_trace:
        st.subheader("Agent Execution Trace")
        log = result.get("agent_log", [])
        if log:
            for i, entry in enumerate(log, start=1):
                st.markdown(f"**{i}.** {entry}")
        else:
            st.info("No agent trace available.")

elif not uploaded_files:
    st.info("Upload one or more PDF documents in the sidebar to get started.")
elif not query.strip():
    st.info("Enter a query above and click **Analyze**.")
