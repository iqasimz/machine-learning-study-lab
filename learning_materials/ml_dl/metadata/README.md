# Metadata contract

Schema version 1. All files are UTF-8 JSON. The 30 lesson records now say **integrated_course** with website_integrated=true. Parameter, dataset and experiment records retain broader design contracts; they must not be mistaken for the executable runtime registry.

The authoritative executable choices are training_lab/model_lab.py and /api/models/catalog. Actual download receipts live under model_data/ and fitted evidence under model_runs/ in a local installation. Curriculum metadata does not replace observed byte hashes, split fingerprints or measurements.

## Files and invariants

- `lessons.json`: unique stable lesson IDs, stage/order, source Markdown, prerequisites, learning outcomes, related datasets/parameters and source IDs. P1–P5/T0/T3/R1 are existing-course dependencies, not new IDs. The prerequisite graph must be acyclic.
- `parameters.json`: unique IDs; role; applicable methods; label; plain definition; mathematical meaning; proposed choices; cause/effect; interactions; failure modes; source IDs; implementation status. Choices are not API defaults. An irrelevant control must be disabled with a reason, not silently ignored.
- `datasets.json`: stable IDs; official source; task; known scope; license evidence; split/preprocessing rules; risks; resource tier; preparation status. No fake checksums: artifact hashes remain null until actual bytes are prepared.
- `experiments.json`: task, datasets, methods, changed factor, controls, metrics, hypotheses (not promised results), failure checks, evidence and estimated resource tier. No fictional results.
- `sources.json`: exact URL, kind, access result, date and use limits. An accessed page is not a successful model/data download.
- `advanced_sections.json`: ordered numbered source sections integrated into ten existing lectures. IDs identify the 27 advanced guide entries; they are not new course IDs, API controls or trainer registrations. The full content and primary links live in their owning Markdown. Advanced questions are maintained in `interactive_mode/advanced_checks.json` and do not replace the two core checks per lecture.

Human-readable lessons remain the source of teaching content. JSON supplies navigation/control contracts for later code, not a parallel paraphrased textbook. Future code must use allowlisted loaders and validated method-specific settings; never turn metadata into arbitrary Python evaluation, shell execution or remote-code imports.

The 98 cards cover the proposed visible teaching controls, not every keyword in every library API. Fixed backend settings (for example, mixture covariance stabilization, precise solver variant or normalization epsilon) must also be pinned in a method specification before execution. Adding a visible control requires a new complete card and a test proving its effect. Learned weights are defined in the lessons; they are not user-selected hyperparameters.

Changing a default, objective normalization, task definition, dataset split or loss reduction requires a versioned contract update. Preserve old recorded experiments rather than relabelling their results.
