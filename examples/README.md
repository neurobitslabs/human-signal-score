# Examples

This directory contains simple sample files and helper scripts that
demonstrate how to use the `originality_score` package from both the command line
and Python.

## Sample data

* `sample1.txt` – a short lorem ipsum paragraph.
* `sample2.txt` – a different paragraph from a public domain text.

## Running the CLI

From the project root, you can run:

```bash
python -m pip install -e .

# Score the two samples together
originality-score examples/sample1.txt examples/sample2.txt

# Or individually
originality-score examples/sample1.txt
```

## Python usage

```python
from originality_score import compute_human_signal_score

with open("examples/sample1.txt") as f:
    doc1 = f.read()
with open("examples/sample2.txt") as f:
    doc2 = f.read()
score = compute_human_signal_score([doc1, doc2])
print(f"Human Signal Score: {score:.3f}")
```
