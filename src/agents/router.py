"""Router agent: classifies user intent and sets the active agent pipeline."""

import sys
from src.state import AgentState
from src.utils.groq_client import get_groq_llm, call_with_retry

INTENT_TO_AGENTS = {
    "full_analysis": ["document_parser", "financial_analyst", "risk_assessor", "report_writer"],
    "risk_only": ["document_parser", "risk_assessor", "report_writer"],
    "compare": ["document_parser", "financial_analyst", "risk_assessor", "competitor_analyst", "report_writer"],
    "financial_only": ["document_parser", "financial_analyst", "report_writer"],
}

ROUTER_PROMPT = """You are a financial analysis router. Classify the user query into exactly one intent.

Intents:
- full_analysis: comprehensive analysis including financials and risks
- risk_only: focused on risks, red flags, debt concerns
- compare: includes competitor comparison
- financial_only: focused on KPIs, revenue, margins, profitability

Respond with ONLY one of these words: full_analysis, risk_only, compare, financial_only

Query: {query}"""


def router(state: AgentState) -> AgentState:
    """Classify query intent and populate active_agents for the pipeline."""
    query = state["query"]
    intent = "full_analysis"

    try:
        llm = get_groq_llm()
        prompt = ROUTER_PROMPT.format(query=query)
        response = call_with_retry(llm, [{"role": "user", "content": prompt}])
        raw = response.content.strip().lower()

        for key in INTENT_TO_AGENTS:
            if key in raw:
                intent = key
                break

    except Exception as e:
        print(f"[router] LLM error: {e} — defaulting to full_analysis", file=sys.stderr)

    agents = INTENT_TO_AGENTS[intent]
    state["active_agents"] = agents
    state["agent_log"] = state.get("agent_log") or []
    state["agent_log"].append(f"Router: classified intent='{intent}', pipeline={agents}")
    return state
