# tests/test_verify.py
# Phase 7.5 — verify.py logic with a fake NLI model, so these run fast and don't
# load the real one. eval/eval_verify.py measures how well the real model does.

from types import SimpleNamespace

import numpy as np

from src.verify import split_passages, verify_claim, verify_claims


class FakeModel:
    """Entails a claim only if the passage contains the word 'yes'."""

    def __init__(self):
        # Same label order as the real model: entailment is not at index 0.
        config = SimpleNamespace(id2label={0: "contradiction", 1: "entailment", 2: "neutral"})
        self.model = SimpleNamespace(config=config)
        self.pairs = []

    def predict(self, pairs, apply_softmax, batch_size):
        self.pairs.extend(pairs)
        return np.array(
            [[0.0, 0.95, 0.05] if "yes" in premise else [0.1, 0.1, 0.8] for premise, _ in pairs]
        )


def test_split_passages_overlaps():
    assert split_passages("A. B. C. D.", window=3) == ["A. B. C.", "B. C. D."]


def test_split_passages_treats_lines_as_boundaries():
    assert split_passages("- one\n- two\n\n- three", window=1) == ["- one", "- two", "- three"]


def test_split_passages_short_and_empty_text():
    assert split_passages("A. B.", window=3) == ["A. B."]
    assert split_passages("  \n ", window=3) == []


def test_split_passages_window_zero_keeps_whole_text():
    assert split_passages("A. B. C. D.", window=0) == ["A. B. C. D."]


def test_verify_claim_reads_the_entailment_column_and_returns_best_passage():
    result = verify_claim("c", "no. no. no. yes.", window=1, model=FakeModel())
    assert result["supported"] is True
    assert result["score"] == 0.95
    assert result["evidence"] == "yes."


def test_verify_claim_passes_premise_first():
    model = FakeModel()
    verify_claim("the claim", "the chunk.", model=model)
    assert model.pairs == [("the chunk.", "the claim")]


def test_verify_claim_unsupported_below_threshold():
    result = verify_claim("c", "no.", model=FakeModel())
    assert result["supported"] is False


def test_verify_claim_threshold_is_respected():
    result = verify_claim("c", "yes.", threshold=0.99, model=FakeModel())
    assert result["supported"] is False


def test_verify_claim_empty_text_is_unsupported_without_calling_the_model():
    model = FakeModel()
    assert verify_claim("c", "", model=model)["supported"] is False
    assert model.pairs == []


def test_verify_claims_one_supporting_citation_is_enough():
    result = verify_claims(
        [{"text": "c", "citations": [1, 2]}], {1: "no.", 2: "yes."}, model=FakeModel()
    )
    assert result["supported"][0]["citation"] == 2
    assert result["faithfulness"] == 1.0


def test_verify_claims_splits_and_scores():
    claims = [
        {"text": "good", "citations": [1]},
        {"text": "bad", "citations": [2]},
    ]
    result = verify_claims(claims, {1: "yes.", 2: "no."}, model=FakeModel())
    assert [c["text"] for c in result["supported"]] == ["good"]
    assert [c["text"] for c in result["unsupported"]] == ["bad"]
    assert result["faithfulness"] == 0.5


def test_verify_claims_unknown_or_missing_citation_cannot_support():
    claims = [{"text": "a", "citations": [99]}, {"text": "b", "citations": []}]
    result = verify_claims(claims, {1: "yes."}, model=FakeModel())
    assert len(result["unsupported"]) == 2
    assert result["faithfulness"] == 0.0


def test_verify_claims_no_claims_has_no_faithfulness():
    assert verify_claims([], {}, model=FakeModel())["faithfulness"] is None
