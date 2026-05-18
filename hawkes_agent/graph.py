from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from hawkes_agent.memory import RetrievedSegment
from hawkes_agent.recall import RecallMiddleware

if TYPE_CHECKING:
    from hawkes_agent.llm import LiteLLMRouter


class AgentState(TypedDict, total=False):
    messages: list[dict[str, str]]
    user_turn: str
    turn_id: str
    type_class: str
    answer: str
    now: float
    namespace: str
    metadata: dict[str, Any]
    retrieved_segments: list[dict[str, Any]]
    adopted_ids: list[str]
    adoption_scores: dict[str, float]
    contradiction_candidates: list[dict[str, str]]
    contradicted_ids: list[str]
    prescreen_signal: float
    retrieval_counts: dict[str, Any]
    replay_events: dict[str, int]
    hot_top1_score: float
    hot_margin: float
    hot_score_entropy: float
    cold_trigger_reason: str | None


def _bump(state: AgentState, key: str, amount: int = 1) -> None:
    events = state.setdefault("replay_events", {})
    events[key] = int(events.get(key, 0)) + int(amount)


def _retrieved_from_state(state: AgentState) -> list[RetrievedSegment]:
    retrieved: list[RetrievedSegment] = []
    for raw in state.get("retrieved_segments", []):
        retrieved.append(
            RetrievedSegment(
                id=str(raw["id"]),
                text=str(raw["text"]),
                score=float(raw["score"]),
                cos_at_recall=float(raw["cos_at_recall"]),
                lambda_minus_snapshot=float(raw["lambda_minus_snapshot"]),
                t_created=float(raw["t_created"]),
                t_last_event=float(raw["t_last_event"]),
                type_class=str(raw["type_class"]),
                namespace=str(raw["namespace"]),
                metadata=dict(raw.get("metadata") or {}),
                retrieval_pool=raw.get("retrieval_pool"),
                bm25_at_recall=float(raw.get("bm25_at_recall", 0.0) or 0.0),
                hawkes_score=float(raw.get("hawkes_score", raw.get("score", 0.0)) or 0.0),
                cold_candidate_score=float(raw.get("cold_candidate_score", 0.0) or 0.0),
                rerank_score=float(raw.get("rerank_score", raw.get("score", 0.0)) or 0.0),
            )
        )
    return retrieved


def _candidate_payload(segments: list[RetrievedSegment]) -> list[dict[str, str]]:
    return [
        {
            "id": s.id,
            "text": s.text,
            "cos_at_recall": f"{s.cos_at_recall:.6f}",
        }
        for s in segments
    ]


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

    def adoption_prescreen_node(state: AgentState) -> AgentState:
        retrieved = _retrieved_from_state(state)
        adopted, scores = memory.score_adoption(state["answer"], retrieved)
        signal, suspicious = memory.prescreen_contradiction_signal(retrieved, adopted, scores)
        state["adopted_ids"] = adopted
        state["adoption_scores"] = scores
        state["contradiction_candidates"] = _candidate_payload(suspicious)
        state["contradicted_ids"] = []
        state["prescreen_signal"] = signal
        return state

    def memory_update_node(state: AgentState) -> AgentState:
        retrieved = _retrieved_from_state(state)
        memory.reinforce(retrieved, state.get("adopted_ids", []), now=state["now"])
        memory.suppress(retrieved, state.get("contradicted_ids", []), now=state["now"])
        return state

    def contradiction_micro_node(state: AgentState) -> AgentState:
        result = llm.classify_contradictions(
            user_turn=state["user_turn"],
            candidates=state.get("contradiction_candidates", []),
        )
        state["contradicted_ids"] = result.contradicted
        return state

    def should_check_contradictions(state: AgentState) -> str:
        if not memory.config.enable_contradiction_micro:
            return "memory_update"
        return "contradiction" if state.get("prescreen_signal", 0.0) > 0.0 else "memory_update"

    graph = StateGraph(AgentState)
    graph.add_node("recall", recall_node)
    graph.add_node("main_llm", main_llm_node)
    graph.add_node("adoption_prescreen", adoption_prescreen_node)
    graph.add_node("contradiction_micro", contradiction_micro_node)
    graph.add_node("memory_update", memory_update_node)
    graph.set_entry_point("recall")
    graph.add_edge("recall", "main_llm")
    graph.add_edge("main_llm", "adoption_prescreen")
    graph.add_conditional_edges(
        "adoption_prescreen",
        should_check_contradictions,
        {"contradiction": "contradiction_micro", "memory_update": "memory_update"},
    )
    graph.add_edge("contradiction_micro", "memory_update")
    graph.add_edge("memory_update", END)
    return graph.compile()


def build_memory_replay_graph(memory: RecallMiddleware, llm: LiteLLMRouter | None = None):
    """Build the full memory-update loop for replaying real dataset turns.

    The graph does not invent dialogue and does not answer the final benchmark
    question. It only applies the memory mechanism to observed turns:
    hot/cold recall, optional contradiction micro-classification, adoption
    reinforcement, contradiction suppression, and write.
    """

    from langgraph.graph import END, StateGraph

    def recall_node(state: AgentState) -> AgentState:
        segments, _mu, counts = memory.hot_cold_reranked_recall(
            state["user_turn"],
            now=state["now"],
            namespace=state["namespace"],
            threshold=None,
        )
        state["retrieved_segments"] = [s.__dict__ for s in segments]
        state["retrieval_counts"] = counts
        state["hot_top1_score"] = float(counts.get("hot_top1_score", 0.0) or 0.0)
        state["hot_margin"] = float(counts.get("hot_margin", 0.0) or 0.0)
        state["hot_score_entropy"] = float(counts.get("hot_score_entropy", 0.0) or 0.0)
        reason = counts.get("cold_trigger_reason")
        state["cold_trigger_reason"] = str(reason) if reason else None
        _bump(state, "retrieved", len(segments))
        _bump(state, "hot_retrieved", int(counts.get("hot", 0) or 0))
        _bump(state, "cold_retrieved", int(counts.get("cold", 0) or 0))
        _bump(state, "cold_triggered", int(counts.get("cold_triggered", 0) or 0))
        state["contradiction_candidates"] = []
        state["prescreen_signal"] = 0.0
        return state

    def adoption_prescreen_node(state: AgentState) -> AgentState:
        retrieved = _retrieved_from_state(state)
        update_text = state.get("answer") or state["user_turn"]
        adopted, scores = memory.score_adoption(update_text, retrieved)
        signal, suspicious = memory.prescreen_contradiction_signal(retrieved, adopted, scores)
        state["adopted_ids"] = adopted
        state["adoption_scores"] = scores
        state["contradiction_candidates"] = _candidate_payload(suspicious)
        state["contradicted_ids"] = []
        state["prescreen_signal"] = signal
        _bump(state, "adopted", len(adopted))
        return state

    def contradiction_micro_node(state: AgentState) -> AgentState:
        if llm is None or not memory.config.enable_contradiction_micro:
            state["contradicted_ids"] = []
            return state
        candidates = state.get("contradiction_candidates", [])
        if not candidates:
            state["contradicted_ids"] = []
            return state
        update_text = state.get("answer") or state["user_turn"]
        result = llm.classify_contradictions(
            user_turn=update_text,
            candidates=candidates,
        )
        state["contradicted_ids"] = result.contradicted
        _bump(state, "contradiction_micro_calls")
        _bump(state, "llm_calls")
        return state

    def memory_update_node(state: AgentState) -> AgentState:
        # Re-run recall segment reconstruction from the state snapshot. This
        # keeps the benchmark update tied to the graph output rather than doing
        # another vector search.
        retrieved = _retrieved_from_state(state)
        adopted = state.get("adopted_ids", [])
        memory.reinforce(retrieved, adopted, now=state["now"])
        contradicted = state.get("contradicted_ids", [])
        memory.suppress(retrieved, contradicted, now=state["now"])
        _bump(state, "contradicted", len(contradicted))
        return state

    def write_node(state: AgentState) -> AgentState:
        memory_text = state.get("answer") or state["user_turn"]
        memory.write_turn(
            id=state["turn_id"],
            text=memory_text,
            now=state["now"],
            namespace=state["namespace"],
            type_class=state.get("type_class"),
            metadata=state.get("metadata"),
        )
        _bump(state, "written")
        return state

    def should_check_contradictions(state: AgentState) -> str:
        if not memory.config.enable_contradiction_micro:
            return "memory_update"
        return "contradiction_micro" if state.get("prescreen_signal", 0.0) > 0.0 else "memory_update"

    graph = StateGraph(AgentState)
    graph.add_node("recall", recall_node)
    graph.add_node("adoption_prescreen", adoption_prescreen_node)
    graph.add_node("contradiction_micro", contradiction_micro_node)
    graph.add_node("memory_update", memory_update_node)
    graph.add_node("write", write_node)
    graph.set_entry_point("recall")
    graph.add_edge("recall", "adoption_prescreen")
    graph.add_conditional_edges(
        "adoption_prescreen",
        should_check_contradictions,
        {
            "contradiction_micro": "contradiction_micro",
            "memory_update": "memory_update",
        },
    )
    graph.add_edge("contradiction_micro", "memory_update")
    graph.add_edge("memory_update", "write")
    graph.add_edge("write", END)
    return graph.compile()
