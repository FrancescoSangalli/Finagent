"""Unit tests for the LangGraph pipeline structure and execution."""

from unittest.mock import MagicMock, patch

import pytest

from src.state import AgentState

EXPECTED_NODES = {
    "router",
    "document_parser",
    "financial_analyst",
    "risk_assessor",
    "competitor_analyst",
    "report_writer",
}


def _base_state(**overrides) -> AgentState:
    """Return a minimal valid AgentState for testing."""
    state: AgentState = {
        "query": "Full analysis",
        "documents": [],
        "parsed_data": {},
        "financial_analysis": {},
        "risk_report": {},
        "competitor_analysis": {},
        "final_report": "",
        "agent_log": [],
        "active_agents": [],
        "error": None,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Graph construction tests
# ---------------------------------------------------------------------------

def test_graph_builds_without_error():
    """build_graph() compiles without raising any exception."""
    from src.graph import build_graph
    graph = build_graph()
    assert graph is not None


def test_graph_has_all_nodes():
    """The compiled graph contains all 6 required agent nodes."""
    from src.graph import build_graph
    graph = build_graph()
    # Compiled LangGraph exposes nodes via the underlying graph attribute
    nodes = set(graph.get_graph().nodes.keys())
    for expected in EXPECTED_NODES:
        assert expected in nodes, f"Missing node: '{expected}'"


# ---------------------------------------------------------------------------
# Full pipeline mock test
# ---------------------------------------------------------------------------

def _make_mock_response(content: str) -> MagicMock:
    """Return a mock LLM response with the given content string."""
    r = MagicMock()
    r.content = content
    return r


@patch("src.agents.report_writer.call_with_retry")
@patch("src.agents.report_writer.get_groq_llm")
@patch("src.agents.competitor_analyst.retrieve")
@patch("src.agents.risk_assessor.call_with_retry")
@patch("src.agents.risk_assessor.get_groq_llm")
@patch("src.agents.financial_analyst.call_with_retry")
@patch("src.agents.financial_analyst.get_groq_llm")
@patch("src.agents.document_parser.call_with_retry")
@patch("src.agents.document_parser.get_groq_llm")
@patch("src.agents.router.call_with_retry")
@patch("src.agents.router.get_groq_llm")
def test_full_pipeline_mock(
    mock_router_llm, mock_router_retry,
    mock_dp_llm, mock_dp_retry,
    mock_fa_llm, mock_fa_retry,
    mock_ra_llm, mock_ra_retry,
    mock_retrieve,
    mock_rw_llm, mock_rw_retry,
):
    """Full pipeline traverses all nodes and produces a non-empty final_report."""
    import json
    from src.graph import build_graph

    # Router returns full_analysis intent
    mock_router_retry.return_value = _make_mock_response("full_analysis")

    # DocumentParser: LLM returns company metadata
    mock_dp_retry.return_value = _make_mock_response(
        json.dumps({"company": "Acme Corp", "year": 2024, "type": "10-K"})
    )

    # FinancialAnalyst: LLM returns KPI JSON
    kpis = {
        "revenue": {"value": 500.0, "unit": "USD_millions", "yoy_change_pct": 8.0},
        "gross_margin_pct": 45.0,
        "operating_margin_pct": 20.0,
        "net_income": {"value": 75.0, "unit": "USD_millions"},
        "ebitda": {"value": 110.0, "unit": "USD_millions"},
        "debt_equity_ratio": 0.5,
        "cash_and_equivalents": {"value": 200.0, "unit": "USD_millions"},
        "revenue_growth_3yr_cagr": 12.0,
        "data_quality": "high",
    }
    mock_fa_retry.return_value = _make_mock_response(json.dumps(kpis))

    # RiskAssessor: LLM returns risk JSON
    risk = {
        "overall_risk": "medium",
        "flags": [
            {"category": "debt_covenant", "description": "Leverage ratio near limit", "severity": "medium"}
        ],
        "debt_concerns": "Leverage ratio approaching covenant threshold.",
        "litigation_summary": None,
        "key_dependencies": ["key customer A", "cloud provider B"],
    }
    mock_ra_retry.return_value = _make_mock_response(json.dumps(risk))

    # ReportWriter: LLM returns a markdown report
    mock_rw_retry.return_value = _make_mock_response(
        "# Acme Corp — M&A Investment Report\n\n"
        "## 1. Executive Summary\nSolid business.\n\n"
        "## 2. Financial Performance\nRevenue: $500M.\n\n"
        "## 3. Risk Assessment\nMedium risk.\n\n"
        "## 4. Competitive Position\nN/A.\n\n"
        "## 5. Investment Considerations\n- Consider acquisition.\n"
    )

    # Build and invoke the graph
    from langchain_core.documents import Document
    graph = build_graph()
    initial_state = _base_state(
        query="Full analysis of Acme Corp 10-K",
        documents=[Document(page_content="Annual report content...", metadata={"source": "fake.txt"})],
        parsed_data={"raw_text": "Annual report content...", "company": "", "year": None, "type": "", "tables": []},
    )

    result = graph.invoke(initial_state)

    assert result["final_report"], "final_report must not be empty"
    assert "Acme Corp" in result["final_report"]
    assert result["active_agents"] == [], "active_agents should be empty after pipeline completes"
    assert len(result["agent_log"]) >= 4
    assert result["error"] is None
