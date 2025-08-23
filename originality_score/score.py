"""Scoring functions for computing diversity and originality metrics.

The goal of these functions is to provide a lightweight, self-contained implementation
of metrics that can help diagnose whether a dataset contains enough human signal
to avoid model collapse.  They are designed to work on small machines without
requiring heavy external dependencies.  Only `numpy` and `scikit‑learn` are used,
alongside the Python standard library and `nltk` for simple tokenisation.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, List, Dict

import numpy as np  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer.

    Splits on whitespace and returns a list of tokens.  Lower‑cases the text
    before splitting.  This is intentionally simple; you may substitute a
    more sophisticated tokenizer depending on your language or use case.

    Args:
        text: The input string.

    Returns:
        A list of tokens.
    """
    return text.lower().split()


def compute_dataset_entropy(texts: Iterable[str]) -> float:
    """Compute the normalised Shannon entropy for a dataset of texts.

    All texts are concatenated into a single sequence of tokens.  The entropy
    of the token distribution is computed and normalised by the maximum
    possible entropy (which occurs when all unique tokens are equally
    probable).  The result is a value between 0 and 1, where 1 indicates
    maximum lexical diversity and 0 indicates all tokens are identical.

    Args:
        texts: An iterable of strings (e.g. lines, paragraphs or documents).

    Returns:
        Normalised entropy in the range ``[0, 1]``.
    """
    # Aggregate tokens from all texts
    tokens: List[str] = []
    for text in texts:
        tokens.extend(_tokenize(text))

    if not tokens:
        return 0.0

    counts = Counter(tokens)
    total = len(tokens)
    # Shannon entropy
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    # Maximum entropy occurs for uniform distribution over the unique tokens
    vocab_size = len(counts)
    if vocab_size <= 1:
        return 0.0
    max_entropy = math.log2(vocab_size)
    return entropy / max_entropy


def compute_perplexity(texts: Iterable[str]) -> float:
    """Compute a normalised perplexity metric for a dataset of texts.

    In the absence of an external language model, perplexity is derived from
    the token distribution in the dataset itself.  Perplexity is defined as
    ``2 ** H``, where ``H`` is the Shannon entropy in bits per token.  It
    represents the effective number of equally likely tokens.  The value is
    normalised by the vocabulary size to yield a result in ``[0, 1]``.

    Args:
        texts: An iterable of strings.

    Returns:
        Normalised perplexity in the range ``[0, 1]``.
    """
    tokens: List[str] = []
    for text in texts:
        tokens.extend(_tokenize(text))

    if not tokens:
        return 0.0

    counts = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    # Perplexity = 2^entropy (effective vocabulary size)
    perplexity = 2.0 ** entropy
    vocab_size = len(counts)
    if vocab_size == 0:
        return 0.0
    return perplexity / vocab_size


def compute_embedding_diversity(texts: Iterable[str]) -> float:
    """Compute the embedding diversity of a collection of texts.

    Texts are vectorised using a TF–IDF representation.  Cosine similarity
    between all pairs of documents is computed.  Diversity is defined as
    ``1 − mean(similarity)`` over all distinct pairs.  The result lies in
    ``[0, 1]``, where 1 indicates no similarity between any pair of documents
    and 0 indicates that all documents are identical.

    Args:
        texts: An iterable of strings representing documents.

    Returns:
        A diversity score in the range ``[0, 1]``.
    """
    docs = list(texts)
    n = len(docs)
    if n == 0:
        return 0.0
    if n == 1:
        # One document is trivially maximally diverse
        return 1.0

    # Create TF–IDF matrix (scikit‑learn will normalise rows to unit length)
    vectorizer = TfidfVectorizer()
    try:
        X = vectorizer.fit_transform(docs)
    except ValueError:
        # All documents are empty or contain only stopwords
        return 0.0

    # Compute cosine similarities; we only need the upper triangle excluding the diagonal
    sim_matrix = cosine_similarity(X)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(sim_matrix[i, j])
    if not sims:
        return 1.0
    mean_sim = float(sum(sims)) / len(sims)
    diversity = 1.0 - mean_sim
    # Bound the result in [0,1]
    if diversity < 0.0:
        diversity = 0.0
    elif diversity > 1.0:
        diversity = 1.0
    return diversity


def compute_human_signal_score(texts: Iterable[str], weights: Dict[str, float] | None = None) -> float:
    """Compute a composite human signal score for a list of texts.

    The score is a weighted sum of normalised Shannon entropy and embedding
    diversity.  Perplexity is not included separately because it is a
    monotonic function of entropy; however, the API allows future metrics to
    be added easily.

    Args:
        texts: An iterable of strings.
        weights: Optional dictionary specifying weights for the components.
            Allowed keys are ``"entropy"`` and ``"diversity"``.  If omitted,
            both components are weighted equally.

    Returns:
        A score between 0 and 1 indicating how much human signal is present.
    """
    if weights is None:
        weights = {"entropy": 0.5, "diversity": 0.5}

    # Normalise weights
    total_weight = sum(weights.values())
    if total_weight == 0:
        # Avoid division by zero; fall back to equal weights
        weights = {"entropy": 0.5, "diversity": 0.5}
        total_weight = 1.0
    for k in weights:
        weights[k] /= total_weight

    entropy = compute_dataset_entropy(texts)
    diversity = compute_embedding_diversity(texts)

    score = weights.get("entropy", 0.0) * entropy + weights.get("diversity", 0.0) * diversity
    # Bound the final score to [0,1]
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
