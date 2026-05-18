from __future__ import annotations

from typing import Any, TypedDict

from hawkes_agent.llm import LiteLLMRouter
from hawkes_agent.recall import RecallMiddleware


class AgentState(TypedDict, total=False):
    messages: list[dict[str, str]]
    user_turn: str
    answer: str
    now: float
    namespace: str
    retrieved_segments: list[dict[str, Any]]
    adopted_ids: list[str]
    contradiction_candidates: list[dict[str, str]]
    contradicted_ids: list[str]
    prescreen_signal: float


def build_agent_graph(memory: RecallMiddleware, llm: LiteLLMRouter):
    """Build the 5-node graph described in the design doc.

    This is intentionally thin. The benchmark harness can run deterministic
    retrieval without invoking the graph, while production-style experiments can
    call this compiled graph for real LLM answers.
    """

    from langgraph.graph import END, StateGraph

    def recall_node(state: AgentState) -> AgentState:
        segments, _mu = memory.recall(
            state["user_turn"],
            now=state["now"],
            namespace=state["namespace"],
        )
        state["retrieved_segments"] = [s.__dict__ for s in segments]
        return state

    def main_llm_node(state: AgentState) -> AgentState:
        memory_block = "\n".join(
            f"[{s['id']}] {s['text']}" for s in state.get("retrieved_segments", [])
        )
        system = "Use retrieved memories only when helpful.\n" + memory_block
        state["answer"] = llm.main_answer(system, state["user_turn"])
        return state

    def adoption_update_node(state: AgentState) -> AgentState:
        # Reconstructing objects is avoided here because this node is meant as
        # a production hook; the deterministic benchmark uses RecallMiddleware
        # directly to preserve exact snapshots.
        return state

    def contradiction_micro_node(state: AgentState) -> AgentState:
        result = llm.classify_contradictions(
            user_turn=state["user_turn"],
            candidates=state.get("contradiction_candidates", []),
        )
        state["contradicted_ids"] = result.contradicted
        return state

    def should_check_contradictions(state: AgentState) -> str:
        return "contradiction" if state.get("prescreen_signal", 0.0) > 0.0 else "adoption"

    graph = StateGraph(AgentState)
    graph.add_node("recall", recall_node)
    graph.add_node("main_llm", main_llm_node)
    graph.add_node("contradiction_micro", contradiction_micro_node)
    graph.add_node("adoption_update", adoption_update_node)
    graph.set_entry_point("recall")
    graph.add_edge("recall", "main_llm")
    graph.add_conditional_edges(
        "main_llm",
        should_check_contradictions,
        {"contradiction": "contradiction_micro", "adoption": "adoption_update"},
    )
    graph.add_edge("contradiction_micro", "adoption_update")
    graph.add_edge("adoption_update", END)
    return graph.compile()
