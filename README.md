# Human Signal Score

This repository provides a simple prototype for computing a **Human Signal Score** for textual datasets.

Modern language models are often trained on data scraped from the web.  When large amounts of synthetic or duplicated text are fed back into the training pipeline, the diversity of the resulting models collapses.  Measuring and maintaining the diversity of your training data is therefore a critical first step toward preventing model collapse.

The `originality_score` package exposes functions to compute:

* **Shannon entropy** of a dataset – a normalised measure of lexical diversity based on token frequencies.
* **Perplexity** – derived from the entropy, representing the effective vocabulary size.
* **Embedding diversity** – a measure of semantic diversity computed from TF–IDF embeddings and pairwise cosine similarity.
* **Human Signal Score** – a weighted combination of the above metrics, scaled to the range `[0, 1]` where higher values indicate greater originality and diversity.

The package also contains a small command–line interface so you can evaluate one or more text files directly from your terminal.

## Installation

To install the package locally in editable mode, run the following commands from the project root:

```sh
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

The only external dependency is scikit‑learn, which is used for TF–IDF vectorisation and cosine similarity computation.  `nltk` is used for tokenisation in the entropy calculation.  These dependencies are declared in the `pyproject.toml` file.

## Usage

### Library

```python
from originality_score.score import compute_human_signal_score

texts = [
    "This is an example document with unique words.",
    "Another document contains different content and vocabulary.",
]

score = compute_human_signal_score(texts)
print(f"Human Signal Score: {score:.3f}")
```

You can also import the individual metrics:

```python
from originality_score.score import compute_dataset_entropy, compute_perplexity, compute_embedding_diversity

entropy = compute_dataset_entropy(texts)
perplexity = compute_perplexity(texts)
diversity = compute_embedding_diversity(texts)
```

### Command line

After installation, you can run the CLI on one or more text files.  The tool will read each file, compute a single score for the collection, and print the result.

```sh
originality-score examples/sample1.txt examples/sample2.txt

By default, the CLI prints individual metrics (entropy, perplexity,
embedding diversity) as well as the final human signal score.  You can
override the weighting of components using the `-w`/`--weight` option.  For
example:

```
originality-score -w entropy=0.7 -w diversity=0.3 examples/sample1.txt
```

## Development

If you wish to contribute to this project, please see the
[`CONTRIBUTING.md`](CONTRIBUTING.md) for instructions on how to set up your
development environment, run tests and follow the style guidelines.  The
repository also includes a `.pre-commit-config.yaml` which integrates
`black`, `ruff` and `pytest` into your workflow.

For a record of changes in each version, see the
[`CHANGELOG.md`](CHANGELOG.md).
```

## License

This project is licensed under the MIT License.  See the `LICENSE` file for details.
