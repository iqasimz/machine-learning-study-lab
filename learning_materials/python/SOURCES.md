# Sources and editorial rules

Prepared 6 November 2026. This pack contains original teaching prose, constructed numerical fixtures and original code. References support definitions/interfaces; they are not imported textbook chapters or permission to reuse copyrighted figures. The examples' execution evidence is separate from source/link research.

## Teaching sources

| Source | How it informed this pack |
| --- | --- |
| [Allen Downey, Think Python, third edition](https://allendowney.github.io/ThinkPython/) | Begin with values, small functions and tracing state; introduce references/classes through concrete behavior rather than a terminology list. Selected variables/functions/iteration/class material, not a claim to have reread the entire book in this pass. |
| [Python tutorial](https://docs.python.org/3/tutorial/) and [standard library](https://docs.python.org/3/library/) | Verify language behavior, files, containers, exceptions, dataclasses and argument parsing. Official tutorial assumes some programming knowledge; this pack supplies an earlier beginner explanation. |
| [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/installing-packages/) | Distinguish interpreter/environment, installation distribution, import package and project dependencies. |
| [Jake VanderPlas, Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) | Array → table → plot → estimator progression. Its online text has reuse restrictions and some APIs are older; no chapters/figures were copied. |
| [NumPy 1.26 documentation](https://numpy.org/doc/1.26/), [pandas 2.3 user guide](https://pandas.pydata.org/pandas-docs/version/2.3/user_guide/index.html), [SciPy guide](https://docs.scipy.org/doc/scipy/tutorial/index.html) | Shapes, views, broadcasting, labeled alignment, join cardinality, sparse storage and statistical calculations. Version-sensitive examples use the recorded environment, not whatever moving “stable” means later. |
| [Matplotlib quickstart](https://matplotlib.org/stable/users/explain/quick_start.html), [Seaborn guide](https://seaborn.pydata.org/tutorial.html) | Explain Figure/Axes and plotted quantities; generate original static figures from included data, not screenshots of source figures. |
| [ISLP](https://www.statlearning.com/) and [sklearn 1.7 guide](https://scikit-learn.org/1.7/user_guide.html) | Couple model assumptions with usable code contracts, baselines, selection and limitations. Specific API/metric references appear in each lesson. |
| [D2L](https://d2l.ai/) and [PyTorch tutorials](https://docs.pytorch.org/tutorials/) | Connect tensor dimensions, layers, differentiation and complete-loop control flow. The pack does not depend on D2L's supporting code/framework. |
| [PyTorch 2.9 docs](https://docs.pytorch.org/docs/2.9/index.html) | CE/BCE targets, SDPA masks/dropout, recurrent return values, module state and loading interfaces. |
| [NLTK book](https://www.nltk.org/book/) and [spaCy 101](https://spacy.io/usage/spacy-101) | Preserve text/source alignment, distinguish rules/tokenization from trained linguistic predictions and identify resource dependencies. No NLTK corpus/model was downloaded. |
| [SLP](https://web.stanford.edu/~jurafsky/slp3/) | NLP task/output distinctions, contextual representations, structured prediction, sequence generation and evaluation. No dependence on a moving chapter number. |
| [Transformers 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/index), [Tokenizers](https://huggingface.co/docs/tokenizers/components), [Datasets](https://huggingface.co/docs/datasets/process), [PEFT](https://huggingface.co/docs/peft/quicktour) | Local model construction/loading, tokenization, response targets and adapter responsibilities. Installed Transformers verification version is recorded separately. |
| [Sentence Transformers](https://sbert.net/docs/quickstart.html), [Gensim](https://radimrehurek.com/gensim/auto_examples/index.html), [FAISS](https://github.com/facebookresearch/faiss/wiki/Getting-started), [PyG](https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html) | Distinguish numerical utilities/representations/indexes/graph contracts from the trained models or complete systems they may support. Gensim/FAISS/PyG library-specific recipes are not executed merely because their concepts are described. |
| [Pydantic](https://docs.pydantic.dev/latest/concepts/models/), [HTTPX](https://www.python-httpx.org/), [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/), [MCP spec](https://modelcontextprotocol.io/specification) | Separate validation, permissions, communication and external side effects. All request examples use in-process mocks; the MCP message is explicitly an illustrative constructed request. |


## Primary method references

- [LoRA](https://arxiv.org/abs/2106.09685): low-rank correction and parameter counts, not a blanket quality guarantee.
- [VAE](https://arxiv.org/abs/1312.6114), [GAN](https://arxiv.org/abs/1406.2661), [DDPM](https://arxiv.org/abs/2006.11239), [SimCLR](https://arxiv.org/abs/2002.05709): distinct latent/adversarial/denoising/contrastive mechanisms. Worked calculations are deliberately limited, not purported reproductions of paper results.
- [DPO](https://arxiv.org/abs/2305.18290), [PPO](https://arxiv.org/abs/1707.06347), [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300): different preference/reward signals and objectives. Supplied-score arithmetic does not claim a trained policy or full implementation.
- [RAG](https://arxiv.org/abs/2005.11401): retrieval-plus-generation distinction. The pack's lexical fixture is not a replication of that system.

Selected book sections/documentation and primary method records were consulted for this preparation and the preceding report. This is not a claim that every linked book/page/paper was read cover to cover or that links establish support for every sentence. Concrete APIs/calculations are additionally tested; explanations still require editorial/student review.

## Output and asset evidence

The Python snippets are authored in the chapters, then extracted and checked by a maintainer-only script. Expected output is not called verified until that script passes. `verification/report.json` records hashes and installed versions; `verification/stdout.json` records captured stdout. Static plot files are generated from DS08/DS09's actual included arrays/counts.

LC01's pretrained answer uses the local lab's existing **HuggingFaceTB/SmolLM2-360M-Instruct** checkpoint, revision `a10cc1512eabd3dde888204e902eca88bddb4951`. The upstream artifact is Apache-2.0; [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct). File hashes are recorded, weights are not copied, and no training/download is performed. One arithmetic answer is a teaching sample, not a benchmark or proof of general correctness. Public release must preserve applicable upstream notices if assets/code are later redistributed.

The plotting/API dependencies absent from the lab environment were installed as pinned wheels into an isolated temporary directory for checking. The application's own installed dependencies and package files were not upgraded. Future integration should define a clean reproducible example environment rather than rely on that temporary directory.

## First-reader editing rules

Explain a task and its objects before showing an API. State whether weights/predictions are supplied or learned. Show actual input/output dimensions and the meaning of their axes. Explain the numerical result, not merely restate the method name. Include one realistic mistake or contrast where it helps.

Avoid false completeness: a named constructor is not a complete model lesson, a forward pass is not a trainer, an example count is not a learner outcome and a package import is not proficiency. The extension register lists material still needing deeper concrete recipes. No learner progress or qualification is awarded by reading this pack.
