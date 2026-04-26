"""Prompt template for the ReportWriter agent."""

REPORT_PROMPT = """You are a senior M&A analyst. Write a professional investment report in markdown.

Use the structured data below. Include ALL five sections. Be concise but specific — use actual numbers.

---

## Input data

**Company:** {company} ({year})

**Financial Analysis:**
{financial_analysis}

**Risk Report:**
{risk_report}

**Competitor Analysis:**
{competitor_analysis}

---

## Required output format (markdown)

# {company} — M&A Investment Report

## 1. Executive Summary
[3-5 sentences: business overview, financial health snapshot, overall investment stance]

## 2. Financial Performance
[Key KPIs with values, YoY trends, margin analysis, cash position]

## 3. Risk Assessment
[Risk flags ordered by severity, overall risk rating, debt and litigation highlights]

## 4. Competitive Position
[Comparison vs competitors if data available; otherwise note data unavailability]

## 5. Investment Considerations
[3-5 bullet points: key factors for M&A decision, valuation considerations, recommended next steps]

---

Write ONLY the markdown report. No preamble, no explanation outside the report."""


def get_report_prompt(
    company: str,
    year,
    financial_analysis: dict,
    risk_report: dict,
    competitor_analysis: dict,
) -> str:
    """Format the report prompt with all agent outputs."""
    return REPORT_PROMPT.format(
        company=company,
        year=year or "N/A",
        financial_analysis=financial_analysis,
        risk_report=risk_report,
        competitor_analysis=competitor_analysis,
    )
