# Contributing

Report a correction with the course, chapter, quoted passage and a reliable source or worked example. For interface issues, include your browser, screen size and steps to reproduce.

Course pages are in `site/courses/`. Keep explanations before notation, define symbols, preserve worked examples and check internal chapter links after edits. The shared reading layout is in `site/reader.css` and `site/reader.js`.

Run `python3 -m unittest discover -s tests -v` and `python3 scripts/check_release.py` before proposing a change. After an intentional, reviewed edit, update checksums with `python3 scripts/check_release.py --write-manifest`. Tests are not a substitute for technical review.

This repository publishes the homepage, five courses and their read-only lab setup/results/comparison views. Do not add training services, datasets, credentials, personal documents, Python lessons or extra courses without discussing a scope change first. Lab catalogs contain no private runs. Keep write operations blocked and training buttons disabled online.
