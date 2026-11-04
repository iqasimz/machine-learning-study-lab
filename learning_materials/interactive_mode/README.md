# Interactive course assets

These assets support Model Lab's guided courses. There are 79 unique active lessons: 5 shared foundations, 16 ML, 14 DL, 26 NLP and 18 LLM lessons. The sidebar and Interactive mode use the same canonical reader with appropriate diagrams or conceptual reading sections and selected unique advanced notes. Original Markdown remains available for maintenance/recovery, not as a repeated full lecture below each diagram.

## Files and ownership

- `lesson_bridges.json`: connections, purposes and takeaways for earlier lessons, plus deliberate lab handoffs that never start training automatically. New NLP lesson connections come from its explicit metadata.
- `foundations_llm_checks.json`: original shared/language/LLM checks; checks for the inactive computer-orientation lesson are not served.
- `ml_checks.json`: 32 ML checks.
- `dl_checks.json`: 28 DL checks.
- `nlp_checks.json`: 48 NLP checks for N01–N24. The two existing NLP language foundations retain their earlier checks and IDs.
- `advanced_checks.json`: 25 additional checks for the extensions inside ten existing ML/DL lectures. Core question IDs and existing top-level paths are unchanged.

The active total is 183 checks, including the shared metrics checks owned in `guided_course.py`. Each question has a stable ID, a prompt, four distinct options with specific feedback, a zero-based correct-answer index, an explanation and a skill label. Numerical examples are authored teaching calculations, not new benchmark measurements.

## How content becomes screens

`src/training_lab/guided_course.py` supplies the same canonical reader produced by `lesson_reader.py` to both navigation modes. A lesson screen contains its linked explorer, guide operations, selected deeper source notes, owned parameter definitions and related links. Two core check screens follow the lesson, then advanced checks where authored. A connection introduces its purpose and a takeaway closes it; deliberate lab handoffs remain separate from reading. There are 266 top-level learning/check/lab steps.

Full-source parsing utilities retain code/display-math boundaries for maintenance and recovery. Their original chunking is not a second rendered teaching route. Canonical source/reader hashes and step fingerprints require review when affected material changes.

Shared foundations appear once, not at the beginning of each subject. P6/P7 belong to NLP; an old LLM cursor at those lessons migrates to NLP without erasing unchanged answer history. Progress is browser-local in `web/course.js`; the Python course API neither saves learner answers nor starts training. Clearing/denying storage or changing browser/address can lose local history.

## Editing and checks

Edit advanced content in source Markdown and its reader section map, primary operation explanations in the matching explorer, related/prerequisite metadata in its registry, and questions in the appropriate bank. Do not put correct-answer positions in visible feedback. NLP metadata is in `nlp/metadata/lessons.json`.

Run the lab's Python tests, `node tests/test_course_explorers.cjs`, `node tests/test_course_state.cjs` and `node tests/test_dashboard.cjs`. These check coverage, the 183 active question schemas, arithmetic, local history, navigation and actual equation rendering. DOM stand-ins do not replace human subject review or real-browser/mobile/accessibility checks.

Native grouped radio controls follow [W3C form grouping guidance](https://www.w3.org/WAI/tutorials/forms/grouping/). Browser persistence follows [MDN localStorage behavior and limits](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage). No external assessment service, answer-upload endpoint, AI tutor or mastery certification is involved.
