import math

from originality_score.score import (
    compute_dataset_entropy,
    compute_perplexity,
    compute_embedding_diversity,
    compute_human_signal_score,
)


def test_entropy_basics() -> None:
    # Uniform distribution: maximum entropy
    texts = ["a b c", "d e f"]
    ent = compute_dataset_entropy(texts)
    assert math.isclose(ent, 1.0, rel_tol=1e-5)

    # Single token repeated: zero entropy
    ent_zero = compute_dataset_entropy(["x x x x"])
    assert math.isclose(ent_zero, 0.0, abs_tol=1e-6)


def test_perplexity_normalisation() -> None:
    texts = ["a b c", "d e f"]
    perp = compute_perplexity(texts)
    # In a uniform distribution of 6 unique tokens, perplexity should be 6 / 6 = 1
    assert math.isclose(perp, 1.0, rel_tol=1e-6)

    # Single token repeated: perplexity should be 1/1 = 1 (but normalised).  The entropy is zero, 2^0=1, vocab=1.
    perp_zero = compute_perplexity(["x x x x"])
    assert math.isclose(perp_zero, 1.0, abs_tol=1e-6)


def test_embedding_diversity() -> None:
    texts = ["this is a unique sentence", "another different sentence"]
    diversity = compute_embedding_diversity(texts)
    assert diversity > 0.0
    # identical documents should give zero diversity
    identical = compute_embedding_diversity(["hello world", "hello world"])
    assert math.isclose(identical, 0.0, abs_tol=1e-6)


def test_human_signal_score_range() -> None:
    texts = ["a quick brown fox", "jumps over the lazy dog"]
    score = compute_human_signal_score(texts)
    assert 0.0 <= score <= 1.0
    # Single document still yields a score
    score_single = compute_human_signal_score(["unique content here"])
    assert 0.0 <= score_single <= 1.0
