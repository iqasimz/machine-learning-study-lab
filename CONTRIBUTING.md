# Contributing

Open an issue describing a reproducible bug, unclear explanation or factual correction. For content changes, include the relevant primary source and a small worked example where useful.

Preserve the single canonical lesson used by sidebar and Interactive mode. Explain terms before equations, show inputs and outputs, and distinguish teaching calculations from measured training results. Do not advertise an architecture as trainable unless an executable backend and tests exist.

Use Python 3.11 and install with `python -m pip install -e '.[language]'`. Run the Python suite and relevant Node tests. Tests must not download data, use paid APIs or depend on the maintainer's saved runs. Mark optional hardware/integration requirements explicitly.

Keep generated experiments, private documents, weights and credentials out of commits. Rebuild the static preview after changing student-facing content. Describe any untested platform or behavior honestly in the pull request.
