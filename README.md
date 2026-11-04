# Machine Learning Study Lab

Study machine learning, deep learning, NLP and LLMs, then test the ideas in a local training lab.

The courses combine plain-language explanations, worked equations, selectable architecture diagrams, sources and practice questions. Browse the sidebar freely or follow Interactive mode. The two routes use the same lessons.

Each lesson includes a connected explanation before its worked calculations. The readings draw on textbooks such as *Dive into Deep Learning*, *Think Stats*, *Speech and Language Processing*, and *An Introduction to Statistical Learning*, alongside original papers and official implementation documentation. References are linked within each lesson. The [teaching chapters](learning_materials/teaching/README.md) contain the text served by both course modes.

## Choose how to use it

- **Online course:** the `site/` folder is a GitHub Pages-ready preview. Courses, diagrams and quizzes work in your browser. It does not train models, detect your GPU, call an inference API or contain somebody else's saved results.
- **Full local lab:** install this repository to train classical ML models, neural networks and small language models. Runs and downloaded assets stay on your computer. Initial installation and optional downloads need internet.

## Install the full lab

Use Python **3.11**. Clone the repository, then install it in a virtual environment:

```bash
git clone https://github.com/iqasimz/machine-learning-study-lab.git
cd machine-learning-study-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-tested.txt -e '.[language]'
python lab.py doctor
python lab.py serve
```

On Windows, create the environment with `py -3.11 -m venv .venv` and activate with `.venv\Scripts\Activate.ps1`. Then use the same `python -m pip ...` and `python lab.py ...` commands. The Windows workflow is documented, not tested on a Windows machine.

Open **http://127.0.0.1:8765**. To stop the server, press Ctrl-C in its terminal. Training never starts simply because you open a page. Only one dashboard-owned training worker runs at a time.

No Python installation is needed for the online course. Opening `site/index.html` directly as a file is not supported; use a web server or GitHub Pages.

### Hardware selection

Neural labs default to **Auto**: a usable CUDA GPU first, then Apple Metal/MPS, then CPU. A tiny forward/backward check verifies the selected backend. The dashboard and `doctor` show the result. Explicitly requesting an unavailable GPU raises an error rather than silently running on CPU.

Classical scikit-learn models remain CPU-only. Detection does not install drivers, combine GPUs or guarantee sufficient memory for every configuration. Model-specific unsupported operations may still fail: choose CPU or reduce batch/context/model size. No settings are silently changed after a run begins. Apple MPS completed a real forward/backward probe during release checks; NVIDIA policy is mock-tested rather than tested on a physical GPU.

For NVIDIA hardware, install the compatible GPU-enabled PyTorch build from the [official installation selector](https://pytorch.org/get-started/locally/) **before** installing this project. A CPU-only PyTorch wheel cannot use your GPU. Apple GPU use requires a compatible native macOS/PyTorch environment. Local CPU training and device-selection logic are tested; CUDA behavior is covered by mocked policy tests, not a physical NVIDIA run.

### Prepare optional data and a language model

Iris, Wine and small digits are available without a separate dataset download. The authored scratch-language fixture teaches the mechanics, not fluent language. For a pretrained language model and additional data, explicitly run:

```bash
python lab.py prepare --model
python lab.py model-prepare fashion_mnist
python lab.py model-prepare wine_quality
```

The language preparation downloads pinned SmolLM2-360M-Instruct weights (about 724 MB) and synthetic conversation data. LoRA adapts that existing model; a short run does not create language ability from scratch. The data may have been seen during its original training. Check the model and dataset terms in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

KaTeX and its license are bundled so mathematical notation works without a CDN. `python lab.py prepare --math` refreshes the pinned renderer explicitly; it is not needed for an intact release.

## What is implemented?

- 79 lessons: shared foundations, ML, DL, NLP, LLM training and post-training/systems.
- 102 interview questions, plus guided-course practice checks.
- Classical classification/regression, clustering, PCA and anomaly experiments.
- PyTorch linear/MLP/CNN/residual-CNN/image-autoencoder training.
- Small from-scratch decoder experiments and pretrained LoRA adaptation.
- Training progress, loss curves, checkpoints, errors and compatible-run comparisons.

NLP architectures, RLHF, distributed training and many advanced methods are explained in the course but are **not all executable trainers**. The lab is an educational project, not an accredited course, production training platform or guarantee of mastery. Numerical lesson examples are not benchmark evidence.

## Tests

```bash
python -m unittest discover -s tests -v
LAB_PYTHON="$(command -v python)" node tests/test_dashboard.cjs
node tests/test_course_state.cjs
node tests/test_chart_layout.cjs
node tests/test_static_preview.cjs
```

Node is needed for JavaScript development tests, not for using the local lab. Tests that need optional model downloads, datasets or a running HTTP server may skip on a fresh checkout. Automated interface tests use DOM stand-ins; they do not replace visual testing in a browser.

## Safety and contributions

Keep the server on loopback. Do not publicly tunnel it or expose it as a multi-user service: it has no user accounts or authentication. Do not load untrusted pickle/joblib artifacts. See [SECURITY.md](SECURITY.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

Original project code and teaching material are offered under the MIT license; third-party components retain their own terms. Corrections and contributions are welcome.
