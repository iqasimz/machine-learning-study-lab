# Machine learning and deep learning

The ML and DL courses contain 30 lessons and 24 interview questions. The local labs let you compare classical methods and neural networks on shared data splits, inspect errors and follow recorded training curves. The model and dataset menus show what is executable.

Each topic is introduced before its controls and calculations. Diagrams show computational relationships; reading sections explain conceptual comparisons. Deeper notes provide derivations and alternatives without duplicating the main explanation.

Advanced sections cover recommendation, reinforcement learning, probability, vision, generative models, graphs, evaluation and deployment. Their placement is recorded in the [section registry](metadata/advanced_sections.json); the [course map](COURSE_MAP.md) gives the prerequisite route. Teaching an advanced method does not imply its trainer is implemented.

The route is **define the problem → build an honest baseline → compare classical methods → learn representations with neural networks → compare complete experiments → continue into the existing LLM course**. Deep learning is part of machine learning, not its automatic replacement.

Read [the course map](COURSE_MAP.md), then the numbered [ML](ml/README.md) and [DL](dl/README.md) lessons. They contain original definitions, process explanations, mathematics, hand calculations, practical scenarios, alternatives and checks—not just links. These are first-edition teaching drafts, not a certified substitute for a university course.

## Source files and teaching contracts

- [Reading route and visual companions](research/READING_ROUTE.md) and [source catalogue](metadata/sources.json).
- [Lesson catalogue](metadata/lessons.json): IDs, dependencies, outcomes, dataset and source references.
- [Parameter catalogue](metadata/parameters.json): every control proposed in this curriculum/lab plan, including optional later branches, with meaning, effect, interactions and limits.
- [Dataset catalogue](metadata/datasets.json): scope, permissions, leakage risks, split design and preparation status.
- [Experiment catalogue](metadata/experiments.json): controlled comparisons and exact evidence requirements.
- [Definitions and notation](DEFINITIONS_AND_NOTATION.md), [method comparisons](METHOD_COMPARISONS.md), [assessments](ASSESSMENTS.md) and [coverage limits](COVERAGE_AND_LIMITS.md).
- [Metadata contract](metadata/README.md).
- [Practical interview questions](INTERVIEW_QUESTIONS.md), twelve ML and twelve DL scenarios with explanations, caveats and follow-up answers.

Run `python3 validate_materials.py` here to check IDs, references, prerequisite order, content structure and selected arithmetic. It uses the standard library only; it does not download or train.

Stanford CS229/CS230/CS231n inform coverage and expected problem-solving depth. There is no affiliation, accreditation, equivalence claim or promise of interview mastery. The benchmark is whether a learner can derive, implement, debug and defend a controlled result.
