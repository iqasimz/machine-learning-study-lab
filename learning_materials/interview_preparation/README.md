# Interview preparation

The website contains 102 stable interview items across ML, DL, NLP, language models and systems. This folder owns 66 of them; ML/DL and NLP have their own source files. These are realistic practice situations, not a prediction of any employer's exact questions or a claim about skills you already possess.

## Choose a topic

| Guide | What to practise |
| --- | --- |
| [Python, mathematics and research](FOUNDATIONS_AND_CODE.md) | Imports/environments, mutable objects, data structures, tensor bugs, loss, uncertainty, APIs, concurrency and project ownership |
| [Language-model training](MODELS_TRAINING_AND_ALIGNMENT.md) | Targets, templates, attention, failed training, resource use, adaptation, preferences and serving |
| [Retrieval and LLM systems](RAG_AGENTS_AND_SECURITY.md) | Search choices, evidence, access control, tools, state, evaluation, latency and recovery |
| [System-design scenarios](LLM_SYSTEM_DESIGN_CASEBOOK.md) | Self-contained support, approval, local-serving, training-budget, outage and release cases |
| [Practical exercises](PRACTICAL_ASSESSMENTS.md) | Small inputs, expected results and debugging/design reasoning; attempt before reading the answer |
| [ML/DL interviews](../ml_dl/INTERVIEW_QUESTIONS.md) | Data/model decisions, rare events, neural debugging and measured comparisons |
| [NLP interviews](../nlp/INTERVIEW_QUESTIONS.md) | Text baselines, recurrent/structured models, retrieval, extraction and language evaluation |

Try a short answer in your own words, then compare it with the explanation. Start with the concrete problem rather than listing framework names. Use a numerical example when it clarifies the idea; a diagram or a data-access boundary can be more useful than an equation for a systems question.

Follow-ups preserve specialist details without making every opening answer a checklist. For design, clarify the constraints and justify the simplest adequate approach. For coding, a small function you can test and modify independently is stronger evidence than a large module you cannot explain.

## Keep knowledge and experience separate

An example answer is not your biography. Distinguish what you designed, implemented, adapted, generated with assistance, reviewed and actually measured. Record assistance honestly and do not infer deployment or multi-GPU experience from a local simulation or a paper you read.

The [readiness tracker](READINESS_TRACKER.md) can separate theory, calculation, coding, debugging and design. The [coverage matrix](COVERAGE_MATRIX.md) preserves deeper course/project links; internal milestone codes are not prerequisites for understanding these answers. No exercise is automatically passed by reading it.

Use disposable inputs for failure drills. Real API spending, cluster jobs, external actions and public release need their own authorization and preparation. The website remains a learning application, not a security or professional-readiness certification.

## Sources

Full derivations and primary links live in the course. Useful implementation references include [PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html), [the Hugging Face LLM course](https://huggingface.co/learn/llm-course/chapter1/1), [PostgreSQL transaction isolation](https://www.postgresql.org/docs/current/transaction-iso.html) and [Google SRE monitoring](https://sre.google/sre-book/monitoring-distributed-systems/). Check release-specific APIs against the installed runtime.
