"""Competitor analyst agent: compares target company against competitors via RAG."""

import sys

from src.state import AgentState
from src.utils.groq_client import get_groq_llm, call_with_retry
from src.rag.retriever import retrieve

COLLECTION_NAME = "competitor_docs"

COMPARE_PROMPT = """You are a financial analyst. Compare the target company against competitors.

Target company: {company}
Target financial metrics:
{financial_analysis}

Competitor context (retrieved passages):
{competitor_chunks}

Produce a concise comparison in JSON with this schema:
{{
  "status": "ok",
  "comparison": {{
    "revenue_position": "<string: how target compares on revenue>",
    "margin_position": "<string: how target compares on margins>",
    "risk_position": "<string: relative risk profile>",
    "summary": "<2-3 sentence narrative>"
  }}
}}

Return ONLY valid JSON, no markdown fences."""


def competitor_analyst(state: AgentState) -> AgentState:
    """Retrieve competitor data via RAG and produce a comparison analysis."""
    state["agent_log"] = state.get("agent_log") or []

    try:
        parsed = state.get("parsed_data") or {}
        company = parsed.get("company", "Unknown")
        financial_analysis = state.get("financial_analysis") or {}

        # Build RAG query from company name and key metrics
        revenue = (financial_analysis.get("revenue") or {})
        query = f"{company} revenue margins profitability competitive position"

        chunks = retrieve(query, COLLECTION_NAME, k=5)

        if not chunks:
            state["competitor_analysis"] = {"status": "no_data", "comparison": None}
            state["agent_log"].append(
                "CompetitorAnalyst: no competitor data in vectorstore, skipping comparison"
            )
            state["active_agents"] = state.get("active_agents", [])[1:]
            return state

        competitor_text = "\n\n---\n\n".join(chunks)
        prompt = COMPARE_PROMPT.format(
            company=company,
            financial_analysis=financial_analysis,
            competitor_chunks=competitor_text,
        )

        llm = get_groq_llm()
        response = call_with_retry(llm, [{"role": "user", "content": prompt}])

        import json
        raw = response.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            block = parts[1] if len(parts) > 1 else raw
            if block.startswith("json"):
                block = block[4:]
            raw = block.strip()

        result = json.loads(raw)
        state["competitor_analysis"] = result
        state["agent_log"].append(
            f"CompetitorAnalyst: comparison complete using {len(chunks)} retrieved chunks"
        )

    except Exception as e:
        state["competitor_analysis"] = {"status": "no_data", "comparison": None}
        state["error"] = f"CompetitorAnalyst failed: {e}"
        print(f"[competitor_analyst] Error: {e}", file=sys.stderr)

    # Advance the pipeline queue
    state["active_agents"] = state.get("active_agents", [])[1:]
    return state
