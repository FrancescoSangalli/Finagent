"""Report writer agent: synthesizes all agent outputs into a final M&A-style report."""

import sys

from src.state import AgentState
from src.utils.groq_client import get_groq_llm, call_with_retry
from src.prompts.report import get_report_prompt


def report_writer(state: AgentState) -> AgentState:
    """Generate the final markdown M&A report from all prior agent outputs."""
    state["agent_log"] = state.get("agent_log") or []

    try:
        parsed = state.get("parsed_data") or {}
        company = parsed.get("company", "Unknown")
        year = parsed.get("year")
        financial_analysis = state.get("financial_analysis") or {}
        risk_report = state.get("risk_report") or {}
        competitor_analysis = state.get("competitor_analysis") or {}

        prompt = get_report_prompt(
            company=company,
            year=year,
            financial_analysis=financial_analysis,
            risk_report=risk_report,
            competitor_analysis=competitor_analysis,
        )

        llm = get_groq_llm()
        response = call_with_retry(llm, [{"role": "user", "content": prompt}])

        state["final_report"] = response.content.strip()
        state["agent_log"].append(
            f"ReportWriter: generated report ({len(state['final_report'])} chars)"
        )

    except Exception as e:
        state["error"] = f"ReportWriter failed: {e}"
        state["final_report"] = f"*Report generation failed: {e}*"
        print(f"[report_writer] Error: {e}", file=sys.stderr)

    # Advance the pipeline queue
    state["active_agents"] = state.get("active_agents", [])[1:]
    return state
