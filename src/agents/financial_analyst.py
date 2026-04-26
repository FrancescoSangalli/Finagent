"""Financial analyst agent: extracts structured KPIs from parsed document text."""

import sys
import json

from src.state import AgentState
from src.utils.groq_client import get_groq_llm, call_with_retry
from src.prompts.analyst import get_analyst_prompt

# Max chars sent to LLM to avoid token limits
MAX_TEXT_CHARS = 12000


def _parse_json_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] contains the block content (possibly prefixed with 'json')
        block = parts[1] if len(parts) > 1 else text
        if block.startswith("json"):
            block = block[4:]
        text = block.strip()
    return json.loads(text)


def financial_analyst(state: AgentState) -> AgentState:
    """Extract financial KPIs from parsed_data and populate financial_analysis."""
    state["agent_log"] = state.get("agent_log") or []

    try:
        raw_text = (state.get("parsed_data") or {}).get("raw_text", "")
        if not raw_text:
            state["financial_analysis"] = {}
            state["agent_log"].append("FinancialAnalyst: no text available, skipping")
            state["active_agents"] = state.get("active_agents", [])[1:]
            return state

        excerpt = raw_text[:MAX_TEXT_CHARS]
        prompt = get_analyst_prompt(excerpt)

        llm = get_groq_llm()
        response = call_with_retry(llm, [{"role": "user", "content": prompt}])
        kpis = _parse_json_response(response.content)

        state["financial_analysis"] = kpis
        populated = sum(1 for v in kpis.values() if v is not None)
        state["agent_log"].append(
            f"FinancialAnalyst: extracted {populated}/{len(kpis)} KPIs, "
            f"data_quality={kpis.get('data_quality', 'unknown')}"
        )

    except json.JSONDecodeError as e:
        state["error"] = f"FinancialAnalyst JSON parse error: {e}"
        state["financial_analysis"] = {}
        print(f"[financial_analyst] JSON parse error: {e}", file=sys.stderr)
    except Exception as e:
        state["error"] = f"FinancialAnalyst failed: {e}"
        state["financial_analysis"] = {}
        print(f"[financial_analyst] Unexpected error: {e}", file=sys.stderr)

    # Advance the pipeline queue
    state["active_agents"] = state.get("active_agents", [])[1:]
    return state
