import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import route_from_router, route_from_critique


def test_route_from_router_easy_goes_to_direct_answer():
    assert route_from_router({"route": "easy"}) == "direct_answer"


def test_route_from_router_medium_goes_to_search():
    assert route_from_router({"route": "medium"}) == "search"


def test_route_from_router_hard_goes_to_decompose():
    assert route_from_router({"route": "hard"}) == "decompose"


def test_route_from_critique_clean_ends():
    state = {"critique_clean": True, "iterations": 1}
    assert route_from_critique(state) == "end"


def test_route_from_critique_unsupported_below_cap_retries():
    state = {"critique_clean": False, "iterations": 2}
    assert route_from_critique(state) == "prepare_retry"


def test_route_from_critique_unsupported_at_cap_ends():
    state = {"critique_clean": False, "iterations": 3}
    assert route_from_critique(state) == "end"


def test_route_from_critique_boundary_two_below_three_cap():
    # iterations == MAX_ITERATIONS - 1 must still allow one more retry.
    state = {"critique_clean": False, "iterations": 2}
    assert route_from_critique(state) == "prepare_retry"


if __name__ == "__main__":
    test_route_from_router_easy_goes_to_direct_answer()
    test_route_from_router_medium_goes_to_search()
    test_route_from_router_hard_goes_to_decompose()
    test_route_from_critique_clean_ends()
    test_route_from_critique_unsupported_below_cap_retries()
    test_route_from_critique_unsupported_at_cap_ends()
    test_route_from_critique_boundary_two_below_three_cap()
    print("=== all graph branch-function tests PASSED ===")
