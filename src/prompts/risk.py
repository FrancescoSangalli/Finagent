"""Prompt template for the RiskAssessor agent."""

RISK_PROMPT = """You are a financial risk analyst. Identify risks and red flags from the document below.

Return ONLY valid JSON matching this exact schema. Do NOT wrap your response in markdown code fences.

{{
  "overall_risk": "low|medium|high|critical",
  "flags": [
    {{
      "category": "<string: e.g. revenue_concentration, litigation, debt_covenant, regulatory, operational>",
      "description": "<string>",
      "severity": "low|medium|high"
    }}
  ],
  "debt_concerns": "<string or null>",
  "litigation_summary": "<string or null>",
  "key_dependencies": ["<string>"]
}}

Rules:
- overall_risk must reflect the aggregate severity across all flags
- flags should be ordered from highest to lowest severity
- debt_concerns: summarize any covenant violations, high leverage, or refinancing risk; null if none
- litigation_summary: summarize material legal proceedings; null if none
- key_dependencies: list 3-5 critical business dependencies (customers, suppliers, technology, geography)

Financial analysis context (for cross-referencing):
{financial_analysis}

Document text:
{text}"""


def get_risk_prompt(text: str, financial_analysis: dict) -> str:
    """Format the risk prompt with document text and financial analysis context."""
    return RISK_PROMPT.format(text=text, financial_analysis=financial_analysis)
