"""LangGraph definition and routing logic for the FinAgent pipeline."""

from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.agents.router import router
from src.agents.document_parser import document_parser
from src.agents.financial_analyst import financial_analyst
from src.agents.risk_assessor import risk_assessor
from src.agents.competitor_analyst import competitor_analyst
from src.agents.report_writer import report_writer

AGENT_FUNCTIONS = {
    "document_parser": document_parser,
    "financial_analyst": financial_analyst,
    "risk_assessor": risk_assessor,
    "competitor_analyst": competitor_analyst,
    "report_writer": report_writer,
}


def get_next_agent(state: AgentState) -> str:
    """Return the next agent in the pipeline queue, or END if the queue is empty."""
    active = state.get("active_agents", [])
    if active:
        return active[0]
    return END


def _make_dispatcher(agent_name: str):
    """Return a conditional-edge target function that routes to a specific agent."""
    def _dispatch(state: AgentState) -> str:
        return agent_name
    _dispatch.__name__ = f"dispatch_{agent_name}"
    return _dispatch


def build_graph():
    """Build and compile the FinAgent LangGraph state machine."""
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("router", router)
    graph.add_node("document_parser", document_parser)
    graph.add_node("financial_analyst", financial_analyst)
    graph.add_node("risk_assessor", risk_assessor)
    graph.add_node("competitor_analyst", competitor_analyst)
    graph.add_node("report_writer", report_writer)

    # Entry point
    graph.set_entry_point("router")

    # After router: branch to the first agent in active_agents
    graph.add_conditional_edges(
        "router",
        lambda state: state["active_agents"][0],
        {
            "document_parser": "document_parser",
            "financial_analyst": "financial_analyst",
            "risk_assessor": "risk_assessor",
            "competitor_analyst": "competitor_analyst",
            "report_writer": "report_writer",
        },
    )

    # After each agent: advance to the next in the pipeline
    for node in ["document_parser", "financial_analyst", "risk_assessor",
                 "competitor_analyst", "report_writer"]:
        graph.add_conditional_edges(
            node,
            get_next_agent,
            {
                "document_parser": "document_parser",
                "financial_analyst": "financial_analyst",
                "risk_assessor": "risk_assessor",
                "competitor_analyst": "competitor_analyst",
                "report_writer": "report_writer",
                END: END,
            },
        )

    return graph.compile()
