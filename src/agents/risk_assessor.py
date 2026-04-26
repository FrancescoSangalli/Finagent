"""Risk assessor agent: identifies red flags and scores overall risk."""

import sys
import json

from src.state import AgentState
from src.utils.groq_client import get_groq_llm, call_with_retry
from src.prompts.risk import get_risk_prompt

MAX_TEXT_CHARS = 12000


def _parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        block = parts[1] if len(parts) > 1 else text
        if block.startswith("json"):
            block = block[4:]
        text = block.strip()
    return json.loads(text)


def risk_assessor(state: AgentState) -> AgentState:
    """Assess risks from parsed_data and populate risk_report in state."""
    state["agent_log"] = state.get("agent_log") or []

    try:
        raw_text = (state.get("parsed_data") or {}).get("raw_text", "")
        if not raw_text:
            state["risk_report"] = {}
            state["agent_log"].append("RiskAssessor: no text available, skipping")
            state["active_agents"] = state.get("active_agents", [])[1:]
            return state

        excerpt = raw_text[:MAX_TEXT_CHARS]
        financial_analysis = state.get("financial_analysis") or {}
        prompt = get_risk_prompt(excerpt, financial_analysis)

        llm = get_groq_llm()
        response = call_with_retry(llm, [{"role": "user", "content": prompt}])
        risk_data = _parse_json_response(response.content)

        state["risk_report"] = risk_data
        num_flags = len(risk_data.get("flags", []))
        overall = risk_data.get("overall_risk", "unknown")
        state["agent_log"].append(
            f"RiskAssessor: overall_risk={overall}, flags={num_flags}"
        )

    except json.JSONDecodeError as e:
        state["error"] = f"RiskAssessor JSON parse error: {e}"
        state["risk_report"] = {}
        print(f"[risk_assessor] JSON parse error: {e}", file=sys.stderr)
    except Exception as e:
        state["error"] = f"RiskAssessor failed: {e}"
        state["risk_report"] = {}
        print(f"[risk_assessor] Unexpected error: {e}", file=sys.stderr)

    # Advance the pipeline queue
    state["active_agents"] = state.get("active_agents", [])[1:]
    return state
