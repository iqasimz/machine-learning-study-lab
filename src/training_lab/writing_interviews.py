"""Writing study material. Historical entries remain; the live reader uses seven topics."""


def simple_items():
    topics = [
    ("W01", "How to plan and draft scientific writing", """
### Begin with the point, not polished sentences

Scientific writing explains a question, the evidence used to answer it and what follows from that evidence. Before drafting, write a short evidence note: **question, comparison, observation, interpretation**. This helps you see whether you have a defensible point before choosing impressive wording.

For example: “Does a generated record help a classifier because of its descriptive text, or because its generated category already predicts the label?” This gives the reader a reason to care about comparing text-only and category-only inputs.

### Turn that into a readable draft

Outline the ideas in the order a new reader needs them. Define the task before explaining the comparison. Write the section you understand well enough first; there is no rule that the abstract must be drafted first. A rough draft is allowed to be awkward. Revise the argument and missing explanations before polishing sentences.

### An interview answer to adapt honestly

“I first identify the reader, the question and the evidence. I make a short outline, draft an explanation, then check whether someone else can reconstruct the comparison. I revise the argument before polishing the wording, and verify numbers and citations.” If this is a new practice, say “This is the process I am now practising,” not that you have always done it.

### How an interviewer might ask about this

- **How do you approach writing a paper?** Explain your own actual drafting and revision process.
- **How do you turn experiments into a story?** Organise around the question and evidence, not the date each script ran.
"""),
    ("W02", "AI assistance, authorship and your own contribution", """
### Assistance does not remove responsibility

AI can suggest wording, identify unclear passages or propose criticism. It can also invent references, change a claim or produce prose you cannot defend. The author remains responsible for the evidence, interpretation and final text. Using AI to draft a section is more substantial than proofreading; describe it accurately.

Writing is also a research skill. If an assistant constructs every first explanation, you lose practice in deciding what matters and connecting the reasoning yourself. A useful learning approach is to draft the claim and paragraph, request specific criticism, then accept only changes you understand and verify.

### How to answer honestly

“I have used substantial AI assistance for coding and writing. I recognise that revising a generated draft is not the same practice as constructing the explanation myself. I am working on writing the core argument first and using assistance for targeted feedback. I check factual changes and references, follow the applicable policy and describe the assistance accurately.” Do not claim a new habit is established before practising it.

For contribution questions, name real decisions and work: research questions, experimental design, dataset choices, implementation/adaptation, validation and analysis. Distinguish your work from joint decisions and supervisory review. Do not imply every line was unaided merely because you directed a tool. Do not upload confidential material without permission.

[ACL publication ethics](https://www.aclweb.org/adminwiki/index.php/ACL_Policy_on_Publication_Ethics)

### How an interviewer might ask about this

- **What did you do yourself?** Describe specific contributions and assistance, rather than relying on first authorship alone.
- **How do you use AI in scientific writing?** Explain boundaries, verification and what you are learning to do independently.
"""),
    ("W04", "Explaining methods and writing clear paragraphs", """
### Define the object before the terminology

Start with what goes in, what comes out and why the operation is needed. Then introduce the technical name. A reader should not need your code to understand the main comparison.

Instead of “We exploit structured representations for robust transfer,” write: “A model labels parts of a passage as claims or premises. We test whether a later sentence classifier benefits from that record compared with reading the original passage.” The second version names the objects and question.

### A paragraph should develop one point

State the point, supply the necessary evidence or explanation, then explain the consequence. The next sentence should build on something the reader already knows. Define an abbreviation at first use and keep the same name for the same condition. Equations help after the quantities have meaning, not instead of that meaning.

### Methods need a reconstructable comparison

Explain data and labels, splits, inputs, fitting/selection, then evaluation. Say which information each system receives. Full settings can go in an appendix, but the central intervention must remain understandable in the main text. Do not present an exploratory choice as planned before seeing results.

### How an interviewer might ask about this

- **Explain your method to someone outside the project.** Begin with an input/output example and connect the steps.
- **How do you make a paper reproducible?** Distinguish a clear comparison from a list of commands.
"""),
    ("W06", "Introductions, related work and using sources", """
### What an introduction needs to establish

Give enough context to understand one unresolved question, explain why the closest existing work does not settle it, then state what your study tests. A broad history of AI or a claim to be first is not a substitute for that question.

For example, if a generated record improves classification, the open question may be whether it adds useful description or already contains a prediction. The introduction should make that distinction clear before listing experiments.

### Related work is a comparison

Group papers by the idea relevant to your question. Explain what they share with your work and what differs: task, data, information access, method or evaluation. It is not a sequence of abstracts and it is not enough that two papers both mention LLMs.

Read the source behind a claim. Check author/title/year, whether a newer version exists and whether the cited result actually supports your sentence. A plausible AI-generated citation is not a source. When paraphrasing, restate the meaning in your own explanation and still cite it; swapping synonyms does not remove the need for attribution.

### How an interviewer might ask about this

- **How is your work different from this paper?** Identify the actual distinguishing comparison, not a list of shared keywords.
- **How do you review literature or avoid plagiarism?** Explain source checking, synthesis and attribution.
"""),
    ("W08", "Results, discussion, figures and conclusions", """
### Give each part a different job

**Results** state what was observed under a comparison. **Discussion** explains what that observation means, why it may matter and which interpretation remains uncertain. **Limitations** identify boundaries of the evidence. **Conclusion** gives the main answer, not a second full results section.

“The category-only rule nearly matches the trained classifier” is a result. “Test generated categories as predictions before attributing gains to representation quality” is a possible consequence. “All generated representations are useless” would go beyond that evidence.

### Make numbers understandable

Identify the population, metric and averaging. A difference from .70 to .75 is .05 on the score scale, or five points on a 0–100 scale; it is not a 5% relative improvement. State the relevant uncertainty without repeating the same caution in every paragraph. A statistically significant change need not be useful enough to justify its cost.

### Tables, captions and abstracts

A table should answer a comparison question. Explain abbreviations and distinguish trained, zero-shot and historical systems where necessary. A caption should let a reader interpret the displayed values. The abstract states the question, approach, central finding and scope; the title should name the actual subject or finding without hype.

### How an interviewer might ask about this

- **What is your main contribution, beyond the score?** Explain what someone should understand or do differently.
- **How do you choose what to report?** Connect each display to the claim it supports.
"""),
    ("W11", "Revising a draft and responding to feedback", """
### Revise the explanation before the wording

Read as someone who does not know the project. Can you identify the question, labels, inputs, comparison and result? Mark an undefined term or unsupported inference before adjusting style. Reading a paragraph aloud can expose jumps hidden by familiar terminology.

### Treat criticism as a diagnosis to investigate

Separate a factual error, an unclear explanation and a genuine disagreement about interpretation. If a reader misunderstands a comparison, first check whether the text supplied the needed distinction. If two interpretations remain possible, ask what evidence would distinguish them.

For a reviewer response, restate the concern, give a direct answer and identify the change or reason for not changing. Do not promise an experiment that has not been done or claim that a revised sentence establishes a missing result. Check that a local edit does not leave inconsistent numbers or claims elsewhere.

### How an interviewer might ask about this

- **How do you respond to critical feedback?** Give a real example of what you changed and why.
- **What if you disagree with a reviewer or supervisor?** Clarify the exact claim and evidence rather than trying to win a wording dispute.
"""),
    ("W14", "Clear research emails and progress updates", """
### Give the reader a useful action

A research email should identify its purpose early, supply only the context needed and make the requested action clear. A progress update distinguishes completed work, interpretation, an open problem and the next decision. It should not be a diary of everything attempted.

### Example

“I finished the matched comparison on the four datasets. The new control reduces the apparent advantage for one generator but not the other. I have checked the prediction counts; I still need to inspect the five failed targets. Could we discuss whether the main text should lead with the generator difference? I attached the comparison table.”

The reader can see what is known, what is not and what response is useful. This is more informative than “Everything is going well; I ran many experiments.” Do not call preliminary results final or send an oversized bundle without identifying the relevant file.

### How an interviewer might ask about this

- **How do you keep supervisors informed?** Explain a concise, decision-oriented update.
- **How would you ask a collaborator for help?** State the problem, checks already made and the specific assistance needed.
"""),
    ]
    return [{"id":identifier, "title":title, "category":"research", "category_label":"Research and writing",
             "source":"interview_preparation/CONSOLIDATED_BANK.md", "markdown":body.strip(), "related_lessons":[]}
            for identifier,title,body in topics]


def items():
    result = []
    def add(identifier, category, title, text):
        result.append({"id": identifier, "category": category,
                       "category_label": "Writing interviews" if category == "writing" else "Writing course",
                       "title": title, "source": "writing_interviews.py", "markdown": text})

    add("W01", "writing", "How do you approach scientific writing?", """
### The idea

Writing is building a checkable argument for a reader. Before choosing polished sentences, decide what question the evidence answers and which comparison supports the answer.

### A concrete answer to adapt honestly

“I start by identifying the reader and the claim. I make a small evidence note: the question, the comparison, the result and the main limit. I outline the explanation in the order the reader needs, then write a rough version myself. I revise the argument first, check whether a new reader can reconstruct the comparison, and only then polish the language. I verify numbers and references against their sources.”

### If that is not yet your established practice

Say “This is the process I am now practising,” not “I have always done this.” A preparation answer is not evidence of a past habit. Try explaining a real paragraph you wrote and one change you made after feedback.
""")
    add("W02", "writing", "How should AI assistance fit into writing?", """
### The idea

An assistant can suggest wording or criticism, but the author remains responsible for the claim, evidence, interpretation and references. The learning question is also important: outsourcing every first draft removes practice in forming your own argument.

### A concrete answer to adapt honestly

“I have used substantial AI assistance. I recognise that adjusting a generated draft is not the same practice as constructing an explanation myself. I am working on drafting the claim and paragraph first, then asking for targeted criticism or language suggestions. I check every factual change and citation, follow the relevant policy and disclose the assistance accurately.”

### What not to say

Do not call substantive drafting assistance proofreading, promise unaided authorship that did not happen, or claim a new habit is established before practising it. Do not upload confidential reviews or manuscripts without permission. Policies differ by venue and role.

[ACL publication ethics](https://www.aclweb.org/adminwiki/index.php/ACL_Policy_on_Publication_Ethics)
""")
    add("W03", "writing", "How do you turn experiments into a paper outline?", """
### The idea

The order you ran experiments is not necessarily the order that explains them. The paper needs a question, necessary definitions, a distinguishing comparison, an observation and its consequence.

### Example

You first try a complex classifier, then discover that a simple rule nearly matches it. A useful outline introduces the competing explanations, defines the rule and classifier inputs, compares their predictions, and discusses what the agreement changes. It need not narrate every failed script in date order.

### Interview answer

“I organise around the comparison that supports the contribution, while stating which decisions were exploratory. I draft whichever section I understand well enough first, often methods or results, and revise the introduction and abstract to match the final evidence.” Clear presentation does not authorise inventing a prospective research history.
""")
    add("W04", "writing", "How do you explain a technical method to a first reader?", """
### The idea

Define the object and the question before symbols or abbreviations. Give an input/output example, then explain each operation's purpose and only the mathematics needed to understand it.

### Example

Instead of “We exploit structured representations for robust transfer,” say: “A model labels parts of a passage as claims or premises. We test whether a later sentence classifier benefits from that record, compared with reading the original passage.” Now the reader knows what was created, what consumes it and what the comparison asks.

### Interview answer

“I check whether a reader can name the input, output and purpose without seeing my code. I introduce a small example, connect the steps, then add notation. I keep a technical term when it is useful and define it rather than replacing it with a vague slogan.”
""")
    add("W05", "writing", "What is the difference between a result and a contribution?", """
### The idea

A result is an observation under a design. A contribution is what the observation or resource adds to understanding or practice. A method, diagnostic, dataset or careful negative comparison can contribute without setting a new leaderboard record.

### Example

“The simple rule agreed with the classifier on nearly all evaluated cases” is a result. “Generated category labels should be tested as standalone predictions before attributing gains to representation quality” is a possible consequence supported by that design. Neither establishes that every generated representation is useless.

### Interview answer

“I state the observation with its population and comparison, then explain what a researcher should do differently because of it. If that implication needs a stronger experiment, I narrow it rather than make the wording more confident.”
""")
    add("W06", "writing", "How do you write an introduction without hype?", """
### The idea

Give the reader enough context to understand a specific unresolved question. A broad history of AI or a first-ever claim rarely identifies the actual uncertainty your experiment resolves.

### Example

Start with two explanations for a classification gain: better descriptive information, or a useful generated label already encoding the answer. Explain why existing comparisons do not separate them, then describe your distinguishing test and the scoped finding.

### Interview answer

“I work backwards from the evidence to the question it resolves, then check the closest literature. The introduction should make the need for the comparison understandable, not pretend the entire field was waiting for my system.” Do not rewrite exploratory discovery as if every conclusion had been predicted.
""")
    add("W07", "writing", "How do you write methods someone else can inspect?", """
### The idea

Methods establish what information reaches each system and how training, selection and final measurement are separated. They are not a diary of terminal commands.

### Example

“We train on one source corpus, choose checkpoints on that source's validation split, and test on each other corpus. We keep shared passages in the same split. The encoder and training budget stay fixed while its input changes.” A reader can now inspect the intended intervention.

### Interview answer

“I explain task and label definition, data/splits, inputs, fitting and selection, then evaluation and aggregation. Full prompts, settings and run details can go in an appendix or repository, but the central comparison must be reconstructable from the paper.”
""")
    add("W08", "writing", "How do you report numbers without misleading the reader?", """
### The idea

A number needs a system, comparison, metric, evaluation population and averaging rule. Distinguish absolute change, relative change and uncertainty.

### Example

A macro-F1 change from 0.70 to 0.74 is +0.04 on that scale, or four score points on a 0–100 scale. Its relative increase is about 5.7%. It is not automatically four percentage points of accuracy. An interval over test resamples and variation across training seeds measure different uncertainty.

### Interview answer

“I report the actual comparison and units, check the unrounded source values, and explain what the uncertainty procedure varies. I do not recompute a supposedly exact contrast by subtracting rounded table cells.”
""")
    add("W09", "writing", "How do results and discussion differ?", """
### The idea

Results say what was observed; discussion explains the consequence and alternatives. Listing the same numbers again with stronger adjectives does not add interpretation.

### Example

Result: a role-based decision agrees closely with a trained reader. Discussion: the generated role may already be supplying a prediction, so a representation claim needs a control for that channel. A separate causal mechanism remains unestablished.

### Interview answer

“I give the numerical comparison a home in Results. In Discussion I explain what it changes, what alternative explanations remain and which practical or research decision follows. I do not repeat the same full caveat in every section.”
""")
    add("W10", "writing", "How do you position related work and verify citations?", """
### The idea

Compare the question, evidence and scope of the closest work, not merely names and publication years. A citation points to support; it does not prove you read or verified it.

### Example

One paper holds out whole tasks; another holds out datasets for one task. Both concern transfer, but their evidence is not interchangeable. Record the relevant section/table and use the published version when available.

### Interview answer

“I keep a note of the source's claim, setup, result and limits, then explain the precise connection. I verify titles, authors, identifiers and which sentence a citation supports. I do not cite a model's invented reference or inherit a secondary source's bibliography without checking.”
""")
    add("W11", "writing", "How do you revise a draft after criticism?", """
### The idea

Revise in order of consequence: argument, reader understanding, then language. A missing control is not repaired by shortening sentences.

### Example

If a reviewer says two systems use different label definitions, first inspect whether the central inference survives. Add a matched comparison if feasible or narrow the claim. Then define the conditions clearly and fix captions. Grammar comes after that scientific repair.

### Interview answer

“I identify the exact inference being challenged, check the evidence, and revise that before polishing the wording. I use feedback to find what the reader could not reconstruct. I can disagree with a stronger unsupported criticism while accepting its valid part.”
""")
    add("W12", "writing", "How do you answer a reviewer without overpromising?", """
### The idea

Identify the concern, state agreement or disagreement, give evidence, and specify the actual change or remaining limitation. A proposed action is not a completed revision.

### Example

“We agree that the comparison changes both links and model depth. We narrowed the claim to the complete processing system. A capacity-matched control would be needed to isolate the links.” Use “narrowed” only after making that edit; otherwise say you will narrow it.

### Interview answer

“I respond to the strongest true part of the criticism and show exactly where the manuscript changed. If a requested study is infeasible, I explain the limit without inventing results or pretending the objection proves an untested alternative cause.”
""")
    add("W13", "writing", "What makes a figure, table or caption useful?", """
### The idea

Choose the visual for the comparison a reader must understand. Explain symbols, axes, populations and uncertainty. A decorative process arrow does not substitute for an experimental design.

### Example

A confusion-count table can explain a recall/precision trade-off better than six unexplained F1 decimals. A graph diagram should say whether an edge means adjacency, alignment or a semantic relation. Error bars must state whether they describe seeds, bootstrap intervals or something else.

### Interview answer

“I ask what question the visual answers, make its units and conditions self-contained, and write the main pattern beside it. I do not put historical and current-model results into a single apparent ranking without the distinctions needed to interpret them.”
""")
    add("W14", "writing", "How do you write a clear email or project update?", """
### The idea

Begin with what the recipient needs to know or decide. Provide only the context needed for that action, name attachments and make the request unambiguous. General professional writing has a different job from a methods section.

### Example

“The registration form is ready for review. Could you confirm the examiner field and sign pages 1–2? I have attached the current PDF; no signature has been inserted.” This is more useful than a long chronology of preparing the form.

### Interview answer

“I identify the audience, purpose and action first. For updates I separate completed work, evidence, blockers and the next decision. I do not bury a request under background or imply that approval has already been given.”
""")
    add("W15", "writing", "How do you paraphrase without losing the source's meaning?", """
### The idea

Understand the claim, make a factual note, write in a structure suited to your explanation, then compare it with the source. Changing synonyms while retaining a distinctive sentence is not enough; the idea still needs attribution.

### Example

A source finding improved performance on one benchmark does not support “the method is generally superior.” Your paraphrase must retain the relevant population and comparison, even if a broader sentence sounds smoother.

### Interview answer

“I check both meaning and provenance. I quote sparingly where exact wording matters; otherwise I explain the idea in my own argument and cite it. I verify that a language edit has not changed causality, statistical meaning or scope.”
""")
    add("W16", "writing", "How do you defend an abstract or title?", """
### The idea

The title identifies the object and contribution. The abstract states the problem, comparison, central observation and consequence, with the boundary that changes its meaning. Neither is a place to sell a stronger paper than the evidence supports.

### Example

“Evaluating Generated Labels in Cross-Domain Classification” identifies a study. “Solving Generalisation with Intelligent Reasoning” makes several additional claims that require evidence. A one-generator, post-hoc win should not replace the diagnostic result in an abstract merely because its number is larger.

### Interview answer

“I check that the title, abstract, tables and conclusion tell the same scoped story. If I cannot explain a sentence aloud and point to supporting evidence, I revise it before treating the draft as ready.”
""")

    add("WC01", "writing-course", "1. Start with evidence, not polished prose", """
Scientific writing makes a question and its answer inspectable. Before drafting, write five rough lines: question, comparison, observation, interpretation and boundary. Those are different jobs. 'We used an LLM' is an activity, not a result or contribution.

### Work through an example

Suppose an elaborate text classifier and a simple rule usually predict the same labels. The question is whether the elaborate representation contributes beyond the rule. The comparison fixes the evaluated texts and checks their predictions. The observation is their agreement, not proof of identical internal processing. The consequence may be that future evaluations need the simple control.

### Build your evidence sheet

For each number, save the metric, population, split, input, selection policy, aggregation and exact source. Mark decisions made after seeing outcomes. A script being frozen today does not make yesterday's analysis preregistered. Separate the observation from possible explanations: a useful generated label can help classification without being a correct semantic annotation.

### Practise

Take one comparison you understand. Write one question sentence, one result sentence and one implication sentence. Then identify the additional experiment needed for a stronger claim. Do this in rough prose yourself before asking a tool for a rewrite. The goal is practice constructing the argument, not performing an unaided-authorship ritual for appearances.
""")
    add("WC02", "writing-course", "2. Make sentences and paragraphs follow the reader", """
A paragraph should help a reader understand one connected point. Introduce the object before judging it; establish familiar context before adding an unfamiliar distinction. These are useful reader expectations, not rigid rules forcing every paragraph into an identical template.

### Repair an unclear sentence

Unclear: “The structural channel enabled robust transfer.” Clearer: “The classifier received the target sentence and a generated claim label. We compared it with the same classifier reading the original passage.” The second version tells the reader what information changed. It does not need a more impressive adjective.

### Connect the explanation

After defining an input, explain who uses it, what comes out and why the comparison matters. Use stable names for experimental conditions; avoiding repetition by switching between three synonyms can make one condition look like three different systems. Prefer verbs that identify actions over long noun chains. Define abbreviations at the first necessary use.

### Practise

Write 100 words explaining a method to someone outside the project. Read it aloud. Mark each pronoun whose referent is unclear and each technical word that arrives before its definition. Revise those points. Passive voice is sometimes appropriate; the issue is whether the actor, action and object can be understood, not banning a grammatical form.
""")
    add("WC03", "writing-course", "3. Explain methods, metrics and uncertainty", """
A methods account establishes the task definition, data/splits, model inputs, fitting/selection and measurement. Explain the crucial comparison in the main text. Detailed prompts, full configurations and hashes can be supporting material; the repository cannot replace the experiment's central logic.

### A controlled comparison

“We train on one source dataset and evaluate on the other three, select checkpoints on the source validation split, and repeat with three seeds. The model and budget stay fixed while the input changes.” This is not training on three datasets jointly and leaving one out. Name the actual held-out unit and keep related passages or patients together when independence requires it.

### Interpret the number

Define the positive class before precision, recall and F1. State whether a mean weights classes, datasets, seeds or individual observations. A rise from 0.70 to 0.74 is +0.04 on the 0–1 score scale, not an unqualified “4% improvement.” Relative increase is 0.04/0.70, about 5.7%.

### Explain uncertainty

A paired bootstrap draws the same evaluation units for both systems; clustering carries related observations together. A confidence procedure has long-run coverage under assumptions, not a probability assigned to the fixed parameter after observing this one interval. Training seeds vary fitted models; test resampling varies the evaluated sample. Multiple-comparison correction controls a specified testing problem, not data leakage or hindsight.

### Practise

Write one methods paragraph and two result sentences. Check the original unrounded evidence. Label a hypothetical arithmetic example as hypothetical, and label exploratory analyses as exploratory rather than changing their history through wording.
""")
    add("WC04", "writing-course", "4. Build the introduction and related-work argument", """
Related work should establish the distinction your experiment needs. For a close source, record its question, claim, evidence, scope and the section/table you checked. Then explain whether your work agrees, extends it, changes a condition or tests an alternative explanation.

### Find a real gap

If existing studies compare a complete generated record with a sentence-only classifier, they may not reveal whether the record's label-like field is doing the work. That specific uncertainty motivates a control. It is stronger than an unsupported claim that nobody has ever studied generated representations.

### Assemble the introduction

Orient the reader, explain the unresolved distinction, identify the comparison that addresses it, and preview the scoped contribution. You may discover the right question during exploration; do not pretend it was all predicted before the experiments. A literature list or a broad history of AI is not automatically the context this reader needs.

### Verify sources

Read the relevant method/results, not just search snippets. Check author names, versions and identifiers. Put citations near their supported claims. Credit an idea when paraphrasing, and do not use another paper's citations as proof you examined the underlying studies.

### Practise

Write 80 words comparing your study with one paper you actually read. Ask whether its authors would recognise their own result in your description. Correct a straw baseline before polishing the paragraph.
""")
    add("WC05", "writing-course", "5. Give results, discussion and limitations different jobs", """
Results identify a comparison and its observed pattern. Discussion explains the consequence and plausible alternatives. A limitation names an inference the design cannot support. Keeping those jobs separate reduces repetitive caveats without hiding important boundaries.

### Explain a result through counts

In a hypothetical 20-positive test set, a detector finding 15 positives but raising 45 false alarms has recall 15/20 and precision 15/60. That count table can explain a deployment decision better than an unexplained decimal. Its caption should specify the population, metric and conditions; the paragraph should interpret the important pattern rather than recite every cell.

### Keep causality separate

If performance falls on social media, conversational style is one hypothesis. Topic, context availability and annotation definitions may also change. Report the failure, then describe the hypothesis as a hypothesis. An added graph reader that changes edges and capacity supports the combined system, not an isolated edge effect.

### Place detail deliberately

Keep a caveat beside the claim when moving it would mislead. Put extended settings and secondary results in supporting material. A diagram earns its space by clarifying inputs or dependencies; label what an edge means. Error bars need their procedure and unit. Repeating the same full statistical-and-limitation paragraph everywhere makes the argument harder, not safer.

### Practise

Write a caption, a result paragraph and a discussion paragraph for one figure. Each should add something different. A useful negative result needs an informative baseline and consequence, not just an unsuccessful run.
""")
    add("WC06", "writing-course", "6. Revise, respond and take responsibility", """
Drafting order can be flexible. You may write methods/results first, use an early abstract as a plan, or revise several sections together. Before sharing, make the final question, evidence, title and conclusion agree.

### Three revision passes

First inspect the argument: missing controls, unsupported inference and exploratory decisions. Next inspect the reader's path: definitions, condition names, captions and connections. Finally edit sentences: grammar, tense, pronouns, unnecessary repetition and emphasis. Language polish cannot repair a scientific design gap.

### Respond accurately

A response to criticism should identify the concern, give evidence, describe the actual change and state any remaining limit. Say “we will revise” for a proposal and “we revised” only after doing it. Disagree with an unsupported claim precisely; do not invent an experiment or equate disagreement with hostility.

### Use tools without surrendering authorship

For learning, try a rough paragraph before asking for targeted criticism. For real work, verify factual changes and citations, comply with the relevant venue/institution policy and describe substantive assistance honestly. Author-writing permissions do not authorise sharing a confidential review. Human authors retain responsibility; coauthor approval, data rights and ethics checks are separate facts to confirm.

### Final exercise

Write a short abstract yourself, revise one paragraph in the three passes and answer one objection. Explain why your revision improved the argument. These are practice tasks, not a certification that reading six pages establishes writing proficiency.

### Further reading

[Mensh and Kording: structuring papers](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005619), [Simon Peyton Jones: research-paper writing](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/), [Purdue: paragraphing](https://owl.purdue.edu/owl/general_writing/academic_writing/paragraphs_and_paragraphing/index.html), [ACL publication ethics](https://www.aclweb.org/adminwiki/index.php/ACL_Policy_on_Publication_Ethics).

These lessons are original explanations and exercises, not reproductions of the linked articles. The private local CV section can additionally show the learner's original research-specific writing course; no private examples are included here.
""")
    return result
