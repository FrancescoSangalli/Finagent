"""Unit tests for individual agents and utilities."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from src.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> AgentState:
    """Return a minimal valid AgentState for testing."""
    state: AgentState = {
        "query": "",
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


def _mock_llm_response(content: str) -> MagicMock:
    """Return a mock LLM response object with the given content."""
    response = MagicMock()
    response.content = content
    return response


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------

@patch("src.agents.router.call_with_retry")
@patch("src.agents.router.get_groq_llm")
def test_router_full_analysis(mock_llm, mock_retry):
    """Router maps 'full_analysis' intent to the correct 4-agent pipeline."""
    from src.agents.router import router, INTENT_TO_AGENTS

    mock_retry.return_value = _mock_llm_response("full_analysis")
    state = _base_state(query="Give me a complete analysis of this 10-K")
    result = router(state)

    assert result["active_agents"] == INTENT_TO_AGENTS["full_analysis"]
    assert "document_parser" in result["active_agents"]
    assert "financial_analyst" in result["active_agents"]
    assert "risk_assessor" in result["active_agents"]
    assert "report_writer" in result["active_agents"]
    assert len(result["active_agents"]) == 4


@patch("src.agents.router.call_with_retry")
@patch("src.agents.router.get_groq_llm")
def test_router_risk_only(mock_llm, mock_retry):
    """Router maps 'risk_only' intent to the 3-agent risk pipeline."""
    from src.agents.router import router, INTENT_TO_AGENTS

    mock_retry.return_value = _mock_llm_response("risk_only")
    state = _base_state(query="What are the main risk factors?")
    result = router(state)

    assert result["active_agents"] == INTENT_TO_AGENTS["risk_only"]
    assert "risk_assessor" in result["active_agents"]
    assert "financial_analyst" not in result["active_agents"]
    assert len(result["active_agents"]) == 3


@patch("src.agents.router.call_with_retry")
@patch("src.agents.router.get_groq_llm")
def test_router_compare(mock_llm, mock_retry):
    """Router maps 'compare' intent to the full 5-agent pipeline."""
    from src.agents.router import router, INTENT_TO_AGENTS

    mock_retry.return_value = _mock_llm_response("compare")
    state = _base_state(query="Compare this company against its competitors")
    result = router(state)

    assert result["active_agents"] == INTENT_TO_AGENTS["compare"]
    assert "competitor_analyst" in result["active_agents"]
    assert len(result["active_agents"]) == 5


@patch("src.agents.router.call_with_retry")
@patch("src.agents.router.get_groq_llm")
def test_router_unknown_intent_defaults_to_full_analysis(mock_llm, mock_retry):
    """Router falls back to full_analysis when LLM returns an unrecognised intent."""
    from src.agents.router import router, INTENT_TO_AGENTS

    mock_retry.return_value = _mock_llm_response("something_unexpected")
    state = _base_state(query="Tell me everything about this company")
    result = router(state)

    assert result["active_agents"] == INTENT_TO_AGENTS["full_analysis"]


@patch("src.agents.router.call_with_retry")
@patch("src.agents.router.get_groq_llm")
def test_router_llm_error_defaults_to_full_analysis(mock_llm, mock_retry):
    """Router falls back to full_analysis when LLM call raises an exception."""
    from src.agents.router import router, INTENT_TO_AGENTS

    mock_retry.side_effect = Exception("LLM unavailable")
    state = _base_state(query="Analyse this document")
    result = router(state)

    assert result["active_agents"] == INTENT_TO_AGENTS["full_analysis"]


# ---------------------------------------------------------------------------
# Rate limiter test
# ---------------------------------------------------------------------------

def test_groq_rate_limiter_respects_min_interval():
    """GroqRateLimiter enforces the minimum interval between consecutive calls."""
    from src.utils.groq_client import GroqRateLimiter

    limiter = GroqRateLimiter(max_rpm=60)  # 1 call/sec
    expected_interval = 60.0 / 60  # 1.0 second

    limiter.wait_if_needed()
    t0 = time.monotonic()
    limiter.wait_if_needed()
    elapsed = time.monotonic() - t0

    # Allow 20 ms tolerance for scheduling jitter
    assert elapsed >= expected_interval - 0.02, (
        f"Elapsed {elapsed:.3f}s is less than min interval {expected_interval}s"
    )


# ---------------------------------------------------------------------------
# PDF parser test
# ---------------------------------------------------------------------------

@patch("src.utils.pdf_parser.pdfplumber")
def test_pdf_parser_returns_expected_keys(mock_plumber):
    """parse_pdf returns a dict with 'text', 'tables', and 'pages' keys."""
    from src.utils.pdf_parser import parse_pdf

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Sample financial text"
    mock_page.extract_tables.return_value = []

    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = lambda s: mock_pdf
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_plumber.open.return_value = mock_pdf

    result = parse_pdf("fake_path.pdf")

    assert "text" in result
    assert "tables" in result
    assert "pages" in result
    assert result["pages"] == 1
    assert "Sample financial text" in result["text"]


# ---------------------------------------------------------------------------
# Financial analyst JSON parsing test
# ---------------------------------------------------------------------------

def test_financial_analyst_parses_plain_json():
    """_parse_json_response handles plain JSON without markdown fences."""
    from src.agents.financial_analyst import _parse_json_response

    payload = json.dumps({"revenue": {"value": 100.0, "unit": "USD_millions", "yoy_change_pct": 5.0}})
    result = _parse_json_response(payload)
    assert result["revenue"]["value"] == 100.0


def test_financial_analyst_parses_markdown_fenced_json():
    """_parse_json_response strips markdown code fences before parsing JSON."""
    from src.agents.financial_analyst import _parse_json_response

    payload = (
        "```json\n"
        '{"revenue": {"value": 200.0, "unit": "USD_millions", "yoy_change_pct": 10.0}, '
        '"gross_margin_pct": 42.5}\n'
        "```"
    )
    result = _parse_json_response(payload)
    assert result["gross_margin_pct"] == 42.5
    assert result["revenue"]["value"] == 200.0


def test_financial_analyst_parses_fenced_json_no_language_tag():
    """_parse_json_response handles fenced JSON blocks with no language tag."""
    from src.agents.financial_analyst import _parse_json_response

    payload = "```\n{\"net_income\": {\"value\": 50.0, \"unit\": \"USD_millions\"}}\n```"
    result = _parse_json_response(payload)
    assert result["net_income"]["value"] == 50.0
