import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import route_from_router, route_from_verify, prepare_research


def test_route_from_router_easy_goes_to_direct_answer():
    assert route_from_router({"route": "easy"}) == "direct_answer"


def test_route_from_router_medium_goes_to_search():
    assert route_from_router({"route": "medium"}) == "search"


def test_route_from_router_hard_goes_to_decompose():
    assert route_from_router({"route": "hard"}) == "decompose"


def test_route_from_router_meta_goes_to_corpus_info():
    assert route_from_router({"route": "meta"}) == "corpus_info"


def test_route_from_verify_dispatch():
    assert route_from_verify({"heal_action": "none"}) == "none"
    assert route_from_verify({"heal_action": "regenerate"}) == "regenerate"
    assert route_from_verify({"heal_action": "research"}) == "research"
    assert route_from_verify({}) == "none"


def test_prepare_research_sanitizes_markers():
    state = {
        "question": "q?",
        "unsupported_claims": ["The Transformer uses Adam [3].", ""],
    }
    r = prepare_research(state)
    assert r["retry_queries"] == ["The Transformer uses Adam ."]  # marker stripped


def test_prepare_research_falls_back_to_question_when_claims_empty():
    state = {"question": "what optimizer is used?", "unsupported_claims": []}
    r = prepare_research(state)
    assert r["retry_queries"] == ["what optimizer is used?"]


def test_prepare_research_does_not_touch_sub_questions():
    state = {
        "question": "q?",
        "unsupported_claims": ["claim one"],
        "sub_questions": ["original sub question"],
    }
    r = prepare_research(state)
    assert "sub_questions" not in r


# ---------- end-to-end wiring (build_graph, monkeypatched nodes) ----------


def test_graph_medium_route_flows_through_verify_to_end(monkeypatch):
    """medium -> search -> generate -> verify -> (none) -> END."""
    import src.graph as graph_module

    calls = []

    def fake_router(state):
        calls.append("router")
        return {"route": "medium", "iterations": 0, "trace": []}

    def fake_search(state):
        calls.append("search")
        return {"retrieved": [{"chunk_id": "c1"}], "iterations": 1, "trace": [], "retry_queries": []}

    def fake_generate(state):
        calls.append("generate")
        return {"answer": "the answer [1]", "citations": [], "trace": []}

    def fake_verify(state):
        calls.append("verify")
        return {"heal_action": "none", "critique_clean": True, "trace": []}

    monkeypatch.setattr(graph_module, "route_question", fake_router)
    monkeypatch.setattr(graph_module, "search_node", fake_search)
    monkeypatch.setattr(graph_module, "generate", fake_generate)
    monkeypatch.setattr(graph_module, "verify", fake_verify)

    app = graph_module.build_graph()
    result = app.invoke({"question": "q?"})

    assert calls == ["router", "search", "generate", "verify"]
    assert result["answer"] == "the answer [1]"


def test_graph_verify_regenerate_loops_back_to_generate_then_terminates(monkeypatch):
    """verify returns regenerate twice, then none: generate must run 3 times
    total (bounded by verify's own budget, not a graph-level counter) and the
    graph must terminate cleanly."""
    import src.graph as graph_module

    generate_calls = []
    verify_outcomes = iter(["regenerate", "regenerate", "none"])

    def fake_router(state):
        return {"route": "medium", "iterations": 0, "trace": []}

    def fake_search(state):
        return {"retrieved": [{"chunk_id": "c1"}], "iterations": 1, "trace": [], "retry_queries": []}

    def fake_generate(state):
        generate_calls.append(1)
        return {"answer": f"answer attempt {len(generate_calls)}", "citations": [], "trace": []}

    def fake_verify(state):
        outcome = next(verify_outcomes)
        return {"heal_action": outcome, "critique_clean": outcome == "none", "trace": []}

    monkeypatch.setattr(graph_module, "route_question", fake_router)
    monkeypatch.setattr(graph_module, "search_node", fake_search)
    monkeypatch.setattr(graph_module, "generate", fake_generate)
    monkeypatch.setattr(graph_module, "verify", fake_verify)

    app = graph_module.build_graph()
    result = app.invoke({"question": "q?"})

    assert len(generate_calls) == 3
    assert result["answer"] == "answer attempt 3"


def test_graph_verify_research_routes_through_prepare_research_back_to_search(monkeypatch):
    """verify returns research -> prepare_research -> search -> generate ->
    verify -> none -> END."""
    import src.graph as graph_module

    calls = []
    verify_outcomes = iter(["research", "none"])

    def fake_router(state):
        return {"route": "medium", "iterations": 0, "trace": []}

    def fake_search(state):
        calls.append("search")
        return {"retrieved": [{"chunk_id": "c1"}], "iterations": state.get("iterations", 0) + 1, "trace": [], "retry_queries": []}

    def fake_generate(state):
        calls.append("generate")
        return {"answer": "an answer", "citations": [], "trace": []}

    def fake_verify(state):
        calls.append("verify")
        outcome = next(verify_outcomes)
        extra = {"unsupported_claims": ["missing claim"]} if outcome == "research" else {}
        return {"heal_action": outcome, "critique_clean": outcome == "none", "trace": [], **extra}

    monkeypatch.setattr(graph_module, "route_question", fake_router)
    monkeypatch.setattr(graph_module, "search_node", fake_search)
    monkeypatch.setattr(graph_module, "generate", fake_generate)
    monkeypatch.setattr(graph_module, "verify", fake_verify)

    app = graph_module.build_graph()
    result = app.invoke({"question": "q?"})

    assert calls == ["search", "generate", "verify", "search", "generate", "verify"]
    assert result["retry_queries"] == []  # consumed by the second search_node call
    assert result["heal_action"] == "none"


if __name__ == "__main__":
    test_route_from_router_easy_goes_to_direct_answer()
    test_route_from_router_medium_goes_to_search()
    test_route_from_router_hard_goes_to_decompose()
    test_route_from_router_meta_goes_to_corpus_info()
    test_route_from_verify_dispatch()
    test_prepare_research_sanitizes_markers()
    test_prepare_research_falls_back_to_question_when_claims_empty()
    test_prepare_research_does_not_touch_sub_questions()
    print("=== all graph branch-function tests PASSED ===")
