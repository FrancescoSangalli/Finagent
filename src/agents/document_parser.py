"""Document parser agent: ingests PDFs and extracts structured metadata."""

import sys
import json

from src.state import AgentState
from src.utils.groq_client import get_groq_llm, call_with_retry
from src.utils.pdf_parser import parse_pdf

STRUCTURE_PROMPT = """Extract the following fields from this financial document excerpt.
Respond with ONLY valid JSON, no markdown, no explanation.

Fields:
- company: company name (string)
- year: fiscal year (integer)
- type: report type, e.g. "10-K", "10-Q", "earnings" (string)

Document excerpt (first 2000 chars):
{text}"""


def document_parser(state: AgentState) -> AgentState:
    """Parse uploaded PDF documents and populate parsed_data in state."""
    state["agent_log"] = state.get("agent_log") or []

    try:
        documents = state.get("documents") or []
        if not documents:
            state["parsed_data"] = {
                "company": "Unknown",
                "year": None,
                "type": "Unknown",
                "raw_text": "",
                "tables": [],
            }
            state["agent_log"].append("DocumentParser: no documents provided, parsed_data empty")
            state["active_agents"] = state.get("active_agents", [])[1:]
            return state

        combined_text = ""
        combined_tables = []

        for doc in documents:
            source = doc.metadata.get("source", "")
            if source.endswith(".pdf"):
                try:
                    result = parse_pdf(source)
                    combined_text += result["text"] + "\n"
                    combined_tables.extend(result["tables"])
                except Exception as e:
                    print(f"[document_parser] Failed to parse {source}: {e}", file=sys.stderr)
                    combined_text += doc.page_content + "\n"
            else:
                combined_text += doc.page_content + "\n"

        # Extract structure via LLM
        company = "Unknown"
        year = None
        report_type = "Unknown"

        try:
            llm = get_groq_llm()
            excerpt = combined_text[:2000]
            prompt = STRUCTURE_PROMPT.format(text=excerpt)
            response = call_with_retry(llm, [{"role": "user", "content": prompt}])
            raw = response.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            meta = json.loads(raw)
            company = meta.get("company", "Unknown")
            year = meta.get("year")
            report_type = meta.get("type", "Unknown")
        except Exception as e:
            print(f"[document_parser] LLM structure extraction failed: {e}", file=sys.stderr)

        state["parsed_data"] = {
            "company": company,
            "year": year,
            "type": report_type,
            "raw_text": combined_text,
            "tables": combined_tables,
        }
        state["agent_log"].append(
            f"DocumentParser: parsed {len(documents)} doc(s), company='{company}', year={year}, type='{report_type}'"
        )

    except Exception as e:
        state["error"] = f"DocumentParser failed: {e}"
        print(f"[document_parser] Unexpected error: {e}", file=sys.stderr)

    # Advance the pipeline queue
    state["active_agents"] = state.get("active_agents", [])[1:]
    return state
