# Contributing to Originality Score

First off, thanks for taking the time to contribute!  This project relies on
community input to grow and improve.  The following guidelines are meant to
make the contribution process as smooth and transparent as possible.

## Code Contributions

1. **Fork the repository** and create your branch from `main`.
2. **Install the dependencies**.  We recommend using a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

   The optional `dev` extra installs tools like `pytest`, `black` and `ruff`.

3. **Create a new branch** for your feature or bug fix:

   ```bash
   git checkout -b my-feature
   ```

4. **Write tests.**  Add tests in `originality_score/tests` that cover your
   change.  Run them locally with `pytest`.

5. **Format your code.**  We use [`black`](https://github.com/psf/black) and
   [`ruff`](https://github.com/astral-sh/ruff) for code style and linting.
   Please run `pre-commit run --all-files` before committing.

6. **Commit your changes.** Use clear, descriptive commit messages.

7. **Push to your fork** and submit a pull request.  Include a description of
   what you’ve changed and why.

## Reporting Bugs

Please use the GitHub issue tracker to report bugs.  Provide as much detail
as you can — a minimal reproducible example is ideal.

## Suggesting Features

Have an idea for a new metric or improvement?  Open an issue to start a
discussion.  We’re particularly interested in metrics that quantify
originality, diversity or collapse risk in innovative ways.

## Security Issues

If you believe you have found a security vulnerability, please do **not**
file a public issue.  Instead, see `SECURITY.md` for instructions on
responsible disclosure.