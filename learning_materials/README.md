# Course materials

These sources support the website's 174 lessons: 79 theory lessons across shared foundations, machine learning, deep learning, natural language processing, and LLM training and systems, plus 95 Python/code lessons. Sidebar browsing and Interactive mode use the same explanations, code, diagrams and worked calculations.

[Python and code](python/README.md) is integrated into the sidebar and Interactive mode, with checked non-training examples, recorded outputs, two plot images and four read-only training walkthroughs. Training happens only in the existing local labs. A worked example is not a promise of complete coverage of every library or architecture.

## Choose a subject

- [Shared mathematics and learning foundations](pre_llm/README.md)
- [Python, data libraries and model code](python/README.md)
- [Machine learning and deep learning](ml_dl/README.md)
- [Natural language processing](nlp/README.md)
- [LLM training](llm_training/README.md)
- [Post-training and systems](post_training/README.md)

The [course map](COURSE_MAP.md) lists the lessons. The [study plan](STUDY_PLAN.md), [mathematical reference](MATH_AND_TERMS_REFERENCE.md), [sources](RESEARCH_AND_RESOURCES.md), [assessments](ASSESSMENTS.md) and [interview questions](interview_preparation/README.md) provide additional practice.

Explanations introduce the topic before equations and diagrams. Numerical examples are teaching calculations; measured training results are produced separately by the local labs. See [coverage and limits](COVERAGE_AND_LIMITS.md) for the distinction between a covered topic and an executable trainer.

For standard-library calculation exercises:

```bash
cd learning_materials/pre_llm/practice
python3 -m unittest -v test_worked_math.py
python3 worked_math.py
```
