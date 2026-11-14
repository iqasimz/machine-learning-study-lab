# Machine Learning Study Lab

[Read the courses and explore the labs](https://iqasimz.github.io/machine-learning-study-lab/)

Five courses with explanations, worked calculations, diagrams, code examples and practice questions. Open a course from the homepage or expand its chapters in the sidebar.

Each course menu also opens its **training lab, results view and comparison page**. Explore available datasets, compatible models, settings, experiment summaries and the actual training-code excerpts. The same sidebar and visual layout are used throughout.

| Course | Chapters, including its final summary |
| --- | ---: |
| Machine Learning | 13 |
| Deep Learning Basics | 16 |
| Deep Learning Architectures | 8 |
| Natural Language Processing | 9 |
| Modern LLMs | 10 |

Each course ends with a summary and interview questions. There are 56 chapters altogether, including an optional dated frontier chapter in Modern LLMs. Small in-page demonstrations illustrate calculations; they do not train models or report benchmark results.

## Read online or offline

The website needs no account or Python environment. To read a downloaded copy, start a static server from this repository:

```sh
python3 -m http.server 8000 --directory site
```

Then open `http://localhost:8000/`. Modern browsers render the embedded MathML equations directly. No external fonts, analytics or paid APIs are required.

## What this release contains

Version **0.19.0** contains the homepage, five courses and read-only ML/DL/NLP/LLM lab views. **Training is local only**: the online buttons are disabled, downloads and other write operations are blocked, and no private saved runs are published. Results and comparison pages explain the workflow but have no personal runs to display. This static release does not include a training server or installation package. It does not detect or use your GPU.

Python/library/DSA pages, personal revision material, scientific-writing notes and the separate interview bank are not included. Course HTML, CSS and JavaScript live in `site/`; GitHub Pages publishes that directory. No downloaded models, datasets, credentials or internal audits are distributed.

Explanations and examples were developed with AI assistance. The material is being reviewed for accuracy and clearer teaching; corrections with a source or worked counterexample are welcome. Automated checks verify structure and links, not every scientific claim or a student's mastery.

## Checks

```sh
python3 -m unittest discover -s tests -v
python3 scripts/check_release.py
```

Node is used by the tests to check JavaScript syntax. Browsing the courses does not require Node. Original code and authored text are MIT-licensed; linked sources retain their own terms. See [third-party notices](THIRD_PARTY_NOTICES.md).
