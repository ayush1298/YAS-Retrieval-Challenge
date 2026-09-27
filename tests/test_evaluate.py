from solution.evaluate import score, set_metrics
from solution.search import Result


def test_passage_query_recall_and_mrr():
    query = {"relevant_passages": ["B1-p02", "B1-p09"]}
    ranked = ["B1-p01", "B1-p02", "B1-p03", "B1-p04", "B1-p05", "B1-p06", "B1-p07", "B1-p08", "B1-p09"]
    assert score(query, Result("passages", ranked, [])) == {"R@5": 0.5, "R@10": 1.0, "MRR": 0.5}


def test_abstention_is_scored_both_ways():
    unanswerable = {"relevant_passages": [], "expected": "abstain"}
    answerable = {"relevant_passages": ["B1-p05"]}
    assert score(unanswerable, Result("abstain", [], [])) == {"R@5": 1.0, "R@10": 1.0, "MRR": 1.0}
    assert score(unanswerable, Result("passages", ["B1-p05"], ["B1"])) == {"R@5": 0.0, "R@10": 0.0, "MRR": 0.0}
    assert score(answerable, Result("abstain", [], [])) == {"R@5": 0.0, "R@10": 0.0, "MRR": 0.0}


def test_page_query_answered_with_passages_uses_pages_of_the_top_10():
    query = {"relevant_pages": ["B1", "S2"]}
    ranked = [f"B1-p{i:02d}" for i in range(1, 11)] + ["S2-p01"]
    result = Result("passages", ranked, ["B1", "S2", "G1"])
    assert score(query, result) == {"R@5": 0.5, "R@10": 0.5, "MRR": 1.0}
    assert set_metrics(query, result) == (0.0, 0.0)


def test_page_query_answered_with_pages():
    query = {"relevant_pages": ["B3", "S2", "S3", "G2"]}
    result = Result("pages", [], ["B3", "G2", "S2", "S3"])
    assert score(query, result) == {"R@5": 1.0, "R@10": 1.0, "MRR": 1.0}
    assert set_metrics(query, result) == (1.0, 1.0)
