"""AgentState definition for the FinAgent LangGraph pipeline."""

from typing import Optional, TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    query: str                      # original user question
    documents: list[Document]       # uploaded and parsed PDF documents
    parsed_data: dict               # structured output from DocumentParser
    financial_analysis: dict        # output from FinancialAnalyst
    risk_report: dict               # output from RiskAssessor
    competitor_analysis: dict       # output from CompetitorAnalyst
    final_report: str               # final M&A-style report
    agent_log: list[str]            # trace of each agent's decisions
    active_agents: list[str]        # agents activated by Router for this query
    error: Optional[str]            # error message if an agent fails
