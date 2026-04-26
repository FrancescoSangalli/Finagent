"""Standalone evaluation script for the FinAgent pipeline."""

import json
import os
import sys
import time
from collections import defaultdict

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.graph import build_graph
from src.state import AgentState
from src.utils.yfinance_validator import get_ground_truth, compare_with_llm_output

CASES_PATH = os.path.join(os.path.dirname(__file__), "test_cases.json")

# Minimal synthetic document text — real eval should use actual PDFs
SAMPLE_TEXT = (
    "Apple Inc. Annual Report 2024. Fiscal Year ended September 2024. "
    "Total revenue: $391 billion. Gross margin: 46.2%. Operating income: $123 billion. "
    "Net income: $97 billion. Cash and equivalents: $65 billion. "
    "Total debt: $104 billion. Stockholders equity: $56 billion. "
    "The company faces regulatory scrutiny in the EU regarding App Store practices."
)

TICKER_FOR_EVAL = "AAPL"


def _make_initial_state(query: str) -> AgentState:
    """Build a minimal AgentState with synthetic document content."""
    from langchain_core.documents import Document

    doc = Document(page_content=SAMPLE_TEXT, metadata={"source": "synthetic", "filename": "AAPL_10K_2024.txt"})
    return {
        "query": query,
        "documents": [doc],
        "parsed_data": {
            "company": "Apple Inc",
            "year": 2024,
            "type": "10-K",
            "raw_text": SAMPLE_TEXT,
            "tables": [],
        },
        "financial_analysis": {},
        "risk_report": {},
        "competitor_analysis": {},
        "final_report": "",
        "agent_log": [],
        "active_agents": [],
        "error": None,
    }


def _extract_intent_from_agents(agents: list[str]) -> str:
    """Reverse-map an agent list back to its intent label."""
    from src.agents.router import INTENT_TO_AGENTS
    for intent, pipeline in INTENT_TO_AGENTS.items():
        if agents == pipeline:
            return intent
    return "unknown"


def run_evaluation() -> None:
    """Execute all evaluation metrics and print a summary report."""
    with open(CASES_PATH) as f:
        test_cases = json.load(f)

    graph = build_graph()

    routing_results = []
    kpi_completeness_results = []
    latencies = []
    consistency_results = []
    kpi_accuracy_results = []

    print("=" * 60)
    print("FinAgent Evaluation")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Routing accuracy + KPI completeness + latency
    # -----------------------------------------------------------------------
    print("\n[1/4] Routing accuracy, KPI completeness, latency...\n")

    for tc in test_cases:
        tc_id = tc["id"]
        query = tc["query"]
        expected_agents = tc["expected_agents"]
        expected_kpis = tc["expected_kpis"]

        state = _make_initial_state(query)
        t0 = time.monotonic()
        try:
            result = graph.invoke(state)
            elapsed = time.monotonic() - t0
        except Exception as e:
            print(f"  {tc_id}: PIPELINE ERROR — {e}")
            latencies.append(None)
            routing_results.append(False)
            continue

        latencies.append(elapsed)

        # Routing
        actual_agents = result.get("active_agents", [])
        routing_ok = actual_agents == expected_agents
        routing_results.append(routing_ok)
        status = "OK" if routing_ok else "FAIL"
        actual_intent = _extract_intent_from_agents(actual_agents)
        print(f"  {tc_id} [{status}] intent={actual_intent} | {elapsed:.1f}s")

        # KPI completeness
        fa = result.get("financial_analysis") or {}
        if expected_kpis:
            present = 0
            for kpi in expected_kpis:
                val = fa.get(kpi)
                if val is not None and val != {} and val != {"value": None}:
                    present += 1
            completeness = present / len(expected_kpis)
            kpi_completeness_results.append(completeness)
        else:
            kpi_completeness_results.append(None)

    # -----------------------------------------------------------------------
    # KPI accuracy against yfinance ground truth
    # -----------------------------------------------------------------------
    print(f"\n[2/4] KPI accuracy vs yfinance ground truth (ticker={TICKER_FOR_EVAL})...\n")

    # Run the financial_only pipeline once to get LLM KPI extraction
    fa_query = "Summarise the financial performance"
    fa_state = _make_initial_state(fa_query)
    try:
        fa_result = graph.invoke(fa_state)
        llm_kpis = fa_result.get("financial_analysis") or {}
        ground_truth = get_ground_truth(TICKER_FOR_EVAL)
        accuracy = compare_with_llm_output(ground_truth, llm_kpis)
        kpi_accuracy_results.append(accuracy)
        print(f"  Matched: {accuracy['matched']}/{accuracy['total']} "
              f"({accuracy['accuracy_pct']}% within 5% tolerance)")
        if accuracy["mismatches"]:
            for m in accuracy["mismatches"]:
                print(f"    Mismatch: {m}")
    except Exception as e:
        print(f"  KPI accuracy check failed: {e}")
        kpi_accuracy_results.append(None)

    # -----------------------------------------------------------------------
    # Consistency: same query, 3 runs, check intent stability
    # -----------------------------------------------------------------------
    print("\n[3/4] Consistency (3 runs of same query)...\n")

    consistency_query = test_cases[0]["query"]
    intents_seen = []
    for run in range(3):
        state = _make_initial_state(consistency_query)
        try:
            result = graph.invoke(state)
            intent = _extract_intent_from_agents(result.get("active_agents", []))
            intents_seen.append(intent)
            print(f"  Run {run+1}: intent={intent}")
        except Exception as e:
            print(f"  Run {run+1}: ERROR — {e}")
            intents_seen.append("error")

    consistent = len(set(intents_seen)) == 1
    consistency_results.append(consistent)

    # -----------------------------------------------------------------------
    # Summary report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    total_cases = len(routing_results)
    routing_ok_count = sum(routing_results)
    routing_pct = round(routing_ok_count / total_cases * 100, 1) if total_cases else 0
    print(f"\nRouting Accuracy     : {routing_ok_count}/{total_cases} ({routing_pct}%)")

    completeness_vals = [v for v in kpi_completeness_results if v is not None]
    if completeness_vals:
        avg_completeness = round(sum(completeness_vals) / len(completeness_vals) * 100, 1)
        print(f"KPI Completeness     : {avg_completeness}% (avg across cases with expected KPIs)")
    else:
        print("KPI Completeness     : N/A")

    if kpi_accuracy_results and kpi_accuracy_results[0]:
        acc = kpi_accuracy_results[0]
        print(f"KPI Accuracy ({TICKER_FOR_EVAL})   : {acc['accuracy_pct']}% ({acc['matched']}/{acc['total']} within 5%)")
    else:
        print(f"KPI Accuracy ({TICKER_FOR_EVAL})   : N/A")

    valid_latencies = [l for l in latencies if l is not None]
    if valid_latencies:
        avg_latency = round(sum(valid_latencies) / len(valid_latencies), 1)
        print(f"Avg Latency          : {avg_latency}s per pipeline run")
    else:
        print("Avg Latency          : N/A")

    consistency_pct = round(sum(consistency_results) / len(consistency_results) * 100, 1) if consistency_results else 0
    print(f"Consistency          : {'PASS' if consistent else 'FAIL'} ({consistency_pct}% stable intent across 3 runs)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_evaluation()
