# FinAgent — Financial Analysis Multi-Agent System

## Overview

FinAgent is a multi-agent system that analyses financial documents (10-K, 10-Q, earnings reports) and produces structured M&A-style investment reports. It combines LLM-powered extraction with RAG-based competitor comparison, making it suitable for due diligence workflows where speed and structure matter.

---

## Architecture

```
User Query + PDF(s)
        │
        ▼
  ┌─────────────┐
  │   Router    │  ← classifies intent, sets agent pipeline
  └──────┬──────┘
         │
         ▼ (conditional, based on intent)
  ┌──────────────────┐
  │ DocumentParser   │  ← extracts text, tables, company metadata
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │ FinancialAnalyst │  ← extracts 9 structured KPIs (optional)
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │  RiskAssessor    │  ← identifies risk flags, scores overall risk (optional)
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │CompetitorAnalyst │  ← RAG retrieval from competitor vectorstore (optional)
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │  ReportWriter    │  ← synthesises final M&A-style markdown report
  └──────────────────┘

Intent → Pipeline mapping:
  full_analysis  : DocumentParser → FinancialAnalyst → RiskAssessor → ReportWriter
  risk_only      : DocumentParser → RiskAssessor → ReportWriter
  financial_only : DocumentParser → FinancialAnalyst → ReportWriter
  compare        : DocumentParser → FinancialAnalyst → RiskAssessor → CompetitorAnalyst → ReportWriter
```

---

## Tech Stack

| Component           | Tool                              | Note                              |
|---------------------|-----------------------------------|-----------------------------------|
| Orchestration       | LangGraph                         | Stateful agent graph              |
| LLM                 | Groq — LLaMA 3.3 70B Versatile    | Free tier, ~28 RPM limit          |
| LLM wrapper         | LangChain + langchain-groq        | ChatGroq interface                |
| Vector store        | ChromaDB (local)                  | Persisted in `./data/chromadb/`   |
| Embeddings          | sentence-transformers all-MiniLM-L6-v2 | On-device, no API needed     |
| PDF parsing         | pdfplumber                        | Text + table extraction           |
| Financial data      | yfinance                          | Ground-truth KPI validation       |
| SEC filings         | edgar                             | 10-K download via EDGAR           |
| UI                  | Streamlit                         | Single-file `app.py`              |
| Rate limiting       | tenacity                          | Exponential backoff, max 3 retry  |
| Testing             | pytest                            | Unit + integration tests          |
| Evaluation          | deepeval + custom run_eval.py     | Routing, KPI, latency metrics     |

---

## Setup

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd finagent

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration (.env)

```bash
cp .env.example .env
# Open .env and set your Groq API key:
# GROQ_API_KEY=gsk_...
```

### Download sample financial data (optional but recommended)

```bash
python data/download_samples.py
```

This downloads real 10-K filings from SEC EDGAR and structured financials via yfinance for AAPL, MSFT, NVDA (main) and AMD, GOOGL (competitors). Files are saved to `data/sample_10k/` and `data/competitor_docs/` (both git-ignored).

### Running the app

```bash
streamlit run app.py
```

---

## Usage

### Streamlit UI

1. **Upload target PDFs** — use the sidebar uploader to load one or more 10-K / 10-Q / earnings PDFs
2. **Upload competitor PDFs** (optional) — load competitor documents, then click **Index Competitor Docs** to embed them into the local ChromaDB vectorstore
3. **Enter a query** — type your analysis question in the main input
4. **Click Analyze** — the pipeline runs and results appear in three tabs:
   - **Report** — full M&A-style markdown report
   - **Data** — raw JSON for financial KPIs and risk flags
   - **Agent Trace** — step-by-step log of what each agent did

### Query examples

| Query | Expected intent |
|-------|----------------|
| `"Give me a full analysis of this 10-K"` | `full_analysis` |
| `"What are the main risk factors and litigation exposure?"` | `risk_only` |
| `"Summarise revenue growth, EBITDA, and margins"` | `financial_only` |
| `"Compare this company against its competitors"` | `compare` |
| `"Is this company a good M&A target? Full report please."` | `full_analysis` |

---

## Evaluation Results

Run the evaluation suite after installing dependencies and setting your API key:

```bash
python tests/eval/run_eval.py
```

| Metric | Value | Note |
|--------|-------|------|
| Routing Accuracy | — | % of test cases routed to correct agent pipeline |
| KPI Completeness | — | % of expected KPIs present and non-null in LLM output |
| KPI Accuracy (AAPL) | — | % of KPIs within 5% of yfinance ground truth |
| Avg Latency | — | Seconds per end-to-end pipeline run |
| Consistency | — | Intent stability across 3 runs of the same query |

*Fill in after running `python tests/eval/run_eval.py` with a real GROQ_API_KEY.*

To run unit tests:

```bash
pytest tests/
```

---

## Project Structure

```
finagent/
├── README.md
├── requirements.txt
├── .env.example
├── pyproject.toml
├── app.py                        # Streamlit entry point
├── src/
│   ├── __init__.py
│   ├── graph.py                  # LangGraph definition + routing logic
│   ├── state.py                  # AgentState TypedDict
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── router.py             # classifies intent, sets agent pipeline
│   │   ├── document_parser.py    # PDF ingestion, text and table extraction
│   │   ├── financial_analyst.py  # structured KPI extraction
│   │   ├── risk_assessor.py      # red flag identification and risk scoring
│   │   ├── competitor_analyst.py # competitor comparison via RAG
│   │   └── report_writer.py      # final M&A-style report synthesis
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py         # sentence-transformers setup
│   │   ├── vectorstore.py        # ChromaDB operations
│   │   └── retriever.py          # top-k similarity retrieval
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── analyst.py            # KPI extraction prompt
│   │   ├── risk.py               # risk assessment prompt
│   │   └── report.py             # M&A report synthesis prompt
│   └── utils/
│       ├── __init__.py
│       ├── groq_client.py        # Groq wrapper with rate limiting + retry
│       ├── pdf_parser.py         # pdfplumber utility
│       └── yfinance_validator.py # ground-truth KPI validation
├── tests/
│   ├── __init__.py
│   ├── test_agents.py            # unit tests for agents and utilities
│   ├── test_graph.py             # LangGraph structure and pipeline tests
│   └── eval/
│       ├── __init__.py
│       ├── test_cases.json       # 7 test cases across all 4 intents
│       └── run_eval.py           # standalone evaluation script
└── data/
    ├── sample_10k/               # downloaded 10-K filings (git-ignored)
    ├── competitor_docs/          # competitor documents for RAG (git-ignored)
    └── download_samples.py       # script to download real financial data
```
