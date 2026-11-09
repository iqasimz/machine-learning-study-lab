# Interview preparation

The website has **62 concept-first topics in seven groups**. A topic explains what an idea means, gives a concrete example or calculation, and only then lists the questions an interviewer might ask about it. Underfitting, for example, is explained before presenting a poor-training-score question. Scenarios are examples at the end, not extra sidebar entries.

The seven groups are Machine learning, Statistics and metrics, Deep learning, NLP, LLMs, LLM applications, and Research and writing. Search by phrase or original question number. All 335 imported prompts retain a destination; related phrasings share an explanation, and ten implementation prompts link to existing code lessons. A destination is a study route, not a claim that every possible follow-up is exhaustively answered.

Scientific/general writing is consolidated into seven topics: drafting, AI/contributions, explanation/methods, introduction/sources, results/discussion, revision/feedback and email.

The complete readable question bank is [CONSOLIDATED_BANK.md](CONSOLIDATED_BANK.md). The files below offer additional practice; they are not separate website menus.

## Preserved source files

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

The [readiness tracker](READINESS_TRACKER.md) separates theory, calculation, coding, debugging and design. The [coverage map](COVERAGE_MATRIX.md) connects topics with lessons and questions. No exercise is automatically passed by reading it.

Use disposable inputs for failure drills. Real API spending, cluster jobs, external actions and public release need their own authorization and preparation. The website remains a learning application, not a security or professional-readiness certification.

## Sources

Full derivations and primary links live in the course. Useful implementation references include [PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html), [the Hugging Face LLM course](https://huggingface.co/learn/llm-course/chapter1/1), [PostgreSQL transaction isolation](https://www.postgresql.org/docs/current/transaction-iso.html) and [Google SRE monitoring](https://sre.google/sre-book/monitoring-distributed-systems/). Check release-specific APIs against the installed runtime.
