"""Command line interface for computing human signal scores.

This module defines a simple CLI entrypoint that reads one or more text
files, computes the human signal score for the combined dataset, and prints
the result.  You can install the package in editable mode (see README) to
make the ``originality-score`` command available on your system.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from .score import compute_human_signal_score


def _read_file(path: pathlib.Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        raise RuntimeError(f"Error reading {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI.

    Args:
        argv: Optional list of arguments.  Defaults to ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute a human signal score for one or more text files. "
            "The score reflects how diverse and original the combined content is."
        )
    )
    parser.add_argument(
        "paths",
        metavar="FILE",
        type=pathlib.Path,
        nargs="+",
        help="one or more text files to evaluate",
    )
    parser.add_argument(
        "-w",
        "--weight",
        dest="weights",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "override component weights, e.g. entropy=0.7,diversity=0.3. "
            "Weights are normalised automatically."
        ),
    )

    args = parser.parse_args(argv)

    # Parse weight overrides if provided
    weights: dict[str, float] | None = None
    if args.weights:
        weights = {}
        for item in args.weights:
            if "=" not in item:
                parser.error(f"Invalid weight specification: {item}")
            key, value = item.split("=", 1)
            try:
                weights[key.strip()] = float(value)
            except ValueError:
                parser.error(f"Invalid weight value for {key}: {value}")

    # Read all files
    texts: list[str] = []
    for path in args.paths:
        if not path.exists():
            parser.error(f"File not found: {path}")
        texts.append(_read_file(path))

    if not texts:
        parser.error("No input files were provided or all were empty.")

    # Compute individual metrics
    from .score import compute_dataset_entropy, compute_perplexity, compute_embedding_diversity

    entropy = compute_dataset_entropy(texts)
    perplexity = compute_perplexity(texts)
    diversity = compute_embedding_diversity(texts)
    score = compute_human_signal_score(texts, weights=weights)

    print("Documents:", len(texts))
    print(f"Token entropy (normalized): {entropy:.4f}")
    print(f"Perplexity (normalized):    {perplexity:.4f}")
    print(f"Embedding diversity:        {diversity:.4f}")
    print("--------------------------------")
    print(f"Human Signal Score:         {score:.4f}")

    return None


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])