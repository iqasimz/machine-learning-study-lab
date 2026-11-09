"""Concept-first interview reader. Historical scenarios are lookup aliases, not menus.

The original bank remains on disk. A former question ID resolves to the topic
that explains it; the public catalog lists only the consolidated topics.
"""
from .interview_bank import EXTRA, source_prompts as original_prompts

GROUPS = {
    "ml": "Machine learning", "statistics": "Statistics and metrics",
    "dl": "Deep learning", "nlp": "NLP", "llm": "LLMs",
    "systems": "LLM applications", "research": "Research and writing",
}

# Each merged item has a specific owner, rather than another visible submenu.
MERGES = {
    "B02":"B01", "B04":"B03", "B14":"B29", "B20":"B19", "B34":"B27", "B61":"B33",
    "B22":"B21", "B25":"B24", "B32":"B39", "B35":"B36",
    "B37":"B29", "B43":"B42", "B48":"B47", "B49":"B50",
    "B56":"B41", "B57":"B26", "B60":"B59", "B65":"B55",
    "B66":"B54", "B68":"B63", "B70":"B67", "B71":"B69",
    "B72":"B54", "B73":"B29", "B74":"B30", "B75":"B58",
    "B76":"B58", "B77":"B06", "B78":"B69",
}
LEGACY = {
    "F01":"B83", "F02":"B83", "F03":"B29", "F04":"B59", "F05":"B84",
    "F06":"B26", "F07":"B42", "F08":"B79", "F09":"B69", "F10":"B82",
    "F11":"B82", "F12":"B82", "F13":"B83", "F14":"B11", "F15":"B29", "F16":"W02",
    "IML01":"B06", "IML02":"B23", "IML03":"B11", "IML04":"B15", "IML05":"B16",
    "IML06":"B05", "IML07":"B79", "IML08":"B18", "IML09":"B18", "IML10":"B06",
    "IML11":"B08", "IML12":"B67",
    "IDL01":"B84", "IDL02":"B19", "IDL03":"B80", "IDL04":"B24", "IDL05":"B23",
    "IDL06":"B26", "IDL07":"B84", "IDL08":"B44", "IDL09":"B39", "IDL10":"B36",
    "IDL11":"B54", "IDL12":"B69",
    "INLP01":"B30", "INLP02":"B81", "INLP03":"B81", "INLP04":"B81", "INLP05":"B31",
    "INLP06":"B36", "INLP07":"B29", "INLP08":"B36", "INLP09":"B30", "INLP10":"B33",
    "INLP11":"B67", "INLP12":"B28",
    "M01":"B42", "M02":"B28", "M03":"B39", "M04":"B38", "M05":"B19", "M06":"B84",
    "M07":"B26", "M08":"B41", "M09":"B44", "M10":"B51", "M11":"B47", "M12":"B47",
    "M13":"B47", "M14":"B53", "M15":"B54", "M16":"B69",
    "S01":"B58", "S02":"B29", "S03":"B58", "S04":"B64", "S05":"B28", "S06":"B59",
    "S07":"B82", "S08":"B64", "S09":"B59", "S10":"B62", "S11":"B54", "S12":"B55",
    "D01":"B58", "D02":"B82", "D03":"B55", "D04":"B69", "D05":"B82", "D06":"B64",
    "P01":"B64", "P02":"B28", "P03":"B38", "P04":"B39", "P05":"B26", "P06":"B84",
    "P07":"B79", "P08":"B58", "P09":"B82", "P10":"B82", "P11":"B64", "P12":"B47",
    "P13":"B54", "P14":"B82", "P15":"B55", "P16":"B69",
    "W03":"W01", "W05":"W08", "W07":"W04", "W09":"W08", "W10":"W06",
    "W12":"W11", "W13":"W08", "W15":"W06", "W16":"W08",
    "WC01":"W01", "WC02":"W04", "WC03":"W08", "WC04":"W06", "WC05":"W08", "WC06":"W11",
}


def canonical(identifier):
    return MERGES.get(identifier, LEGACY.get(identifier, identifier))


def source_prompts():
    overrides = {26:"B67", 27:"B67", 39:"B79", 41:"B79", 54:"B12"}
    return [{**p, "target":overrides.get(p["number"], canonical(p["target"])) if p["kind"] == "question" else p["target"]}
            for p in original_prompts()]


# Clear titles and two examples of how interviewers ask about each concept.
SPEC = {
"B01":("ml", "Bias and variance: underfitting and overfitting", "ML05", [
    "Both training and validation scores are poor. Why? | Check underfitting or an optimisation/data problem.",
    "Training is excellent but validation is poor. What would you change? | Check overfitting, leakage and split differences before choosing a remedy."]),
"B03":("ml", "Types of learning, parameters and hyperparameters", "ML01", [
    "Is predicting the next token supervised or unsupervised? | Explain how the targets are constructed.",
    "What does the optimiser learn, and what do you choose? | Distinguish weights from settings such as learning rate."]),
"B05":("ml", "Preprocessing: missing values, scaling and categories", "ML02", [
    "Why did changing income from euros to cents change k-nearest neighbours? | Distance depends on feature scale.",
    "Can I fill missing values before splitting the table? | Learned preprocessing must be fitted on training data only."]),
"B06":("ml", "Train, validation and test splits; data leakage", "P4 SK08", [
    "Several rows belong to the same patient. Is a random split safe? | Keep the patient together to match the intended new-patient evaluation.",
    "How would you validate next month's demand forecast? | Respect time order and information available at prediction time."]),
"B15":("ml", "Linear and logistic regression", "ML03 ML04", [
    "Why is logistic regression used for classification? | Its sigmoid turns a linear score into a binary probability model.",
    "Two features are almost identical. What can go wrong? | Coefficients can become unstable even if predictions remain similar."]),
"B16":("ml", "Decision trees, random forests and boosting", "ML08 ML09", [
    "Why use a random forest instead of one tree? | Averaging varied trees reduces sensitivity to individual samples.",
    "How is boosting different from bagging? | New boosting learners correct the current ensemble rather than fitting independently."]),
"B17":("ml", "Support vector machines and kernels", "ML06", [
    "What is an SVM trying to maximise? | Explain the margin and the cost of violations.",
    "How can a linear separator create a curved boundary? | The boundary can be linear in a transformed feature space."]),
"B18":("ml", "Clustering and dimensionality reduction", "ML11 ML12", [
    "PCA retains 95% of variance. Must classification improve? | Large input variation need not be the useful label signal.",
    "K-means returned five groups. Are they meaningful? | Check stability, geometry and usefulness, not just whether fitting succeeded."]),
"B67":("ml", "Distribution shift and transfer learning", "ML15 N22", [
    "A news classifier fails on tweets. Why? | Check vocabulary, style, coverage and the definition of the label.",
    "Overall accuracy is good but one group fails. What now? | Inspect group-specific performance and the training/evaluation populations."]),
"B07":("statistics", "Probability, Bayes' rule and likelihood", "P3", [
    "What changes between probability and likelihood? | Fix parameters to predict data, or fix the data to compare parameters.",
    "What is the difference between MLE and MAP? | MAP also uses a prior."]),
"B08":("statistics", "Mean, variance, correlation and causation", "P3", [
    "Why can average latency hide a bad user experience? | A mean hides the shape and tail of the distribution.",
    "Escalated tickets take longer. Does escalation cause delay? | Difficult tickets may be selected for escalation: association alone does not settle cause."]),
"B79":("statistics", "Bootstrapping and confidence intervals", "P3 ML16", [
    "Model B is two points better. How uncertain is that difference? | Resample appropriate evaluation units and compare the models on the same draws.",
    "Why resample whole passages instead of individual sentences? | Sentences sharing a passage are dependent observations."]),
"B09":("statistics", "p-values, significance and Holm correction", "ML16", [
    "Does p = 0.03 mean a 97% chance my claim is true? | No: the probability is calculated under a specified null model.",
    "Why correct when testing many model variants? | More tests create more opportunities for false positives."]),
"B10":("statistics", "Entropy, cross-entropy and KL divergence", "P3", [
    "Why does a confidently wrong answer get a large loss? | Log loss strongly penalises assigning little probability to the observed target.",
    "Why is KL divergence directional? | The first distribution determines which outcomes receive weight."]),
"B11":("statistics", "Accuracy, precision, recall and F1", "P8", [
    "A fraud detector is 99% accurate. Is it useful? | Ask about prevalence, missed fraud and false alarms.",
    "Recall increased but macro-F1 fell. Is that possible? | More positive predictions can catch positives and create many false positives."]),
"B12":("statistics", "Thresholds, ROC-AUC and precision–recall curves", "P8 ML10", [
    "Which threshold should an alerting system use? | Match the precision/recall trade-off to the cost and review capacity.",
    "Why can ROC-AUC look good for a poor fraud detector? | A small false-positive rate can still mean many false alerts."]),
"B13":("statistics", "Calibration: can I trust the model's probabilities?", "ML10", [
    "When the model says 80%, what should that mean? | About 80% of comparable cases should be positive.",
    "Can two equally accurate models have different probability losses? | Yes: their confidence can differ even when labels match."]),
"B19":("dl", "Neural networks, loss and backpropagation", "DL01 DL02", [
    "What happens during one training step? | Trace input, prediction, loss, gradient and weight update.",
    "Why do networks need nonlinear activation functions? | Otherwise stacked linear layers collapse to one linear map."]),
"B21":("dl", "Normalisation, initialisation and residual connections", "DL03 DL06", [
    "What is the difference between BatchNorm and LayerNorm? | State which dimensions supply the normalisation statistics.",
    "Why do residual connections help deep networks? | They give information and gradients a shorter additive path."]),
"B23":("dl", "Regularisation, dropout and early stopping", "ML05 DL04", [
    "Training improves while validation worsens. What should I try? | This is a common overfitting pattern; regularisation or early stopping may help.",
    "Why do predictions change between train and eval mode? | Dropout and some normalisation layers behave differently."]),
"B24":("dl", "Learning rate, SGD, Adam and batch size", "DL03", [
    "Loss oscillates after increasing the learning rate. Why? | Updates may overshoot useful regions.",
    "Can a bigger batch require different settings? | It changes gradient noise, update frequency and memory use."]),
"B26":("dl", "Training memory, precision and gradient accumulation", "T5 T6", [
    "Weights fit in GPU memory. Why does training run out? | Include activations, gradients, optimiser state and temporary buffers.",
    "How do unequal microbatches affect the loss? | Weight by the number of valid training targets, not just the number of microbatches."]),
"B80":("dl", "CNNs: filters, channels and pooling", "DL05", [
    "Why does a CNN use the same filter across an image? | Weight sharing detects a local pattern at multiple positions.",
    "Why did doubling channels greatly increase compute? | A convolution mixes input and output channels at every spatial location."]),
"B81":("dl", "RNNs, LSTMs, GRUs and bidirectionality", "DL07 N09 N10", [
    "How does an LSTM preserve information longer than a basic RNN? | Explain the cell state and gates.",
    "Can a bidirectional model predict the next typed word? | Future input is unavailable in that setting."]),
"B27":("nlp", "Text features: n-grams, TF–IDF and word embeddings", "N02 N03 N05 N06", [
    "What is a sensible first text-classification baseline? | Sparse features plus a linear model give a cheap, inspectable baseline.",
    "How do stemming and lemmatisation differ? | Stemming trims forms; lemmatisation uses linguistic information to recover a base form."]),
"B28":("nlp", "Tokenisation, padding and context length", "P6 N23", [
    "Why do 1,000 words not always mean the same token count? | Tokenisation depends on the vocabulary, language and text.",
    "What if a document is longer than the model accepts? | Choose truncation, chunking or a suitable architecture, and check what information is lost."]),
"B30":("nlp", "Classification, sequence labelling and extraction", "N01 N17 N24", [
    "NER has high token accuracy but poor entity F1. How? | Many O labels can hide incorrect entity boundaries.",
    "What exactly does this dataset label? | Identify the prediction unit and annotation definition before choosing a model."]),
"B31":("nlp", "HMMs, CRFs and valid label sequences", "N12", [
    "Why put a CRF on top of an encoder? | Combine local evidence with scores for neighbouring labels.",
    "How do you find the best whole label sequence? | Dynamic programming reuses partial solutions instead of enumerating every sequence."]),
"B29":("nlp", "Search: BM25, embeddings and reranking", "N03 N15 R4", [
    "Why did keyword search beat an embedding on an error code? | Exact identifiers can matter more than general semantic similarity.",
    "How do you search a million documents efficiently? | Retrieve candidates first, then rerank a small set."]),
"B36":("nlp", "Sentence embeddings and contrastive learning", "N15 DL09", [
    "Can I use any BERT CLS vector for semantic search? | A sentence-similarity training objective matters.",
    "What if a negative example is actually another valid answer? | False negatives push genuinely related representations apart."]),
"B33":("nlp", "Evaluating summaries and generated text", "N20 N24", [
    "A summary has high ROUGE but reverses the conclusion. Why? | Word overlap does not establish factual consistency.",
    "What would you measure besides fluency? | Check task correctness, source support, coverage and harmful omissions separately."]),
"B38":("llm", "Attention: queries, keys, values and softmax", "T3", [
    "What do Q, K and V do? | Explain matching, weighting and information mixing.",
    "How do you compute softmax without overflow? | Subtract the largest logit before exponentiating; the probabilities do not change."]),
"B39":("llm", "Encoders, decoders and causal masks", "N14 T3", [
    "Why can training run in parallel but generation is sequential? | During training the target sequence is known, but future targets must be masked.",
    "What happens if the decoder can read its future targets? | It can copy information that will not exist at generation time."]),
"B40":("llm", "Position embeddings and RoPE", "T3", [
    "How does attention know token order? | Positional information must be supplied through a chosen mechanism.",
    "Does extending the context window guarantee good long-document reasoning? | Accepting more tokens and using them reliably are different."]),
"B41":("llm", "Efficient attention: GQA, FlashAttention and caching", "T3 R6", [
    "Why does grouped-query attention save KV memory? | Several query heads share key/value heads.",
    "Does FlashAttention remove attention's quadratic arithmetic? | It reduces expensive memory traffic without changing dense attention into sparse attention."]),
"B42":("llm", "Next-token training, cross-entropy and perplexity", "T1 T5", [
    "Where do the labels for language-model pretraining come from? | The next observed tokens are the targets.",
    "Can I compare perplexity across different tokenizers directly? | The unit being averaged has changed."]),
"B44":("llm", "Pretraining, instruction tuning and forgetting", "T1 R1", [
    "Should I fine-tune to give the assistant today's policy? | Retrieval handles changing evidence; tuning can teach behaviour or format.",
    "Why did specialist tuning harm general ability? | Updates can overwrite useful behaviour or narrow the training distribution."]),
"B45":("llm", "Knowledge distillation", "R7", [
    "How can a smaller model learn from a bigger one? | Train on the teacher's outputs or distributions.",
    "Does agreement with the teacher establish correctness? | The student can inherit the teacher's mistakes."]),
"B46":("llm", "Ensembles, routers and mixture of experts", "ML08 T4", [
    "How is a router different from averaging predictions? | It chooses or weights experts rather than applying a fixed combination.",
    "Does a sparse MoE use all its parameters for every token? | Only selected experts are activated, although shared computation and communication remain."]),
"B47":("llm", "RL post-training: rewards, PPO, DPO and GRPO", "R2", [
    "What happens in one RL update for an LLM? | Generate responses, score them, estimate an update signal and change policy weights.",
    "Why use DPO instead of an online RL loop? | Available preference pairs can train a different objective without the same rollout/critic loop."]),
"B50":("llm", "Can LLMs reason? Models, tools and explanations", "R3 T8", [
    "Is the model reasoning, or are tools doing the work? | Separate the model alone from scaffolding and tool-assisted performance.",
    "Does a correct chain of thought explain the actual decision? | A plausible rationale is not automatically faithful evidence of the internal process."]),
"B51":("llm", "LoRA and QLoRA", "R1 LC04", [
    "Why can LoRA still run out of memory? | Fewer trainable weights do not remove activation or base-model memory.",
    "How many parameters does a rank-8 adapter add? | Count the two low-rank matrices, not the frozen base matrix."]),
"B52":("llm", "Attention masks, loss masks and chat templates", "T2 LC02", [
    "Why exclude user tokens from the SFT loss? | They may supply context without being outputs the assistant should imitate.",
    "What should I inspect before trusting a training run? | Decode actual IDs and check targets, masks, boundaries and template."]),
"B53":("llm", "Decoding: greedy, beam search and sampling", "R3", [
    "Does temperature zero make an answer true? | Decoding controls selection, not factual correctness.",
    "How are top-k and top-p different? | One limits candidate count; the other limits cumulative probability mass."]),
"B54":("systems", "Serving: prefill, decode, latency and throughput", "R6", [
    "First-token latency rose but tokens per second stayed stable. Why? | Look at queueing and prefill separately from decode.",
    "The GPU is allocated but utilisation is low. Why? | Loading, CPU work, padding, I/O or tiny batches may be the bottleneck."]),
"B55":("systems", "Quantisation, deployment and rollback", "R7 R8", [
    "Can this model fit on a laptop? | Estimate weights plus cache and runtime memory, then measure the actual backend.",
    "A model release made answers worse. What should be rolled back? | Version the model, prompt, tokenizer, index and configuration together."]),
"B58":("systems", "RAG: retrieval-augmented generation", "R4 N19", [
    "The correct passage was retrieved, but the answer is wrong. Why? | Retrieval and correct use of evidence are separate stages.",
    "How do I debug a stale citation? | Trace the query, document version, retrieved chunk and generated claim."]),
"B59":("systems", "Workflows, agents, tools and memory", "R5", [
    "When is a fixed workflow better than an agent? | Use predictable steps when the task does not need open-ended decisions.",
    "Is conversation memory stored in the model weights? | External state, prompt context and learned weights are different stores."]),
"B64":("systems", "Prompt injection, privacy and safe tool use", "R8", [
    "A retrieved document asks for credentials. What stops the action? | Untrusted text must not grant tool permissions.",
    "Does valid JSON make a tool call safe? | Validate authority, arguments and allowed actions as well as syntax."]),
"B82":("systems", "Async APIs, retries and human approval", "LC07 LC08", [
    "A request timed out. Can I retry it? | Check whether the action may already have happened.",
    "Two people click Approve together. How do I prevent duplicate execution? | Use an atomic state transition and an idempotent action."]),
"B63":("research", "Leakage, shortcuts and benchmark contamination", "P4 N22", [
    "Scores collapse after changing a harmless-looking phrase. Why? | The model may rely on a shortcut rather than the intended signal.",
    "Can rewriting training text change its label? | Augmentation must preserve the task definition, not only grammatical quality."]),
"B69":("research", "Fair comparisons, ablations and reproducibility", "ML16", [
    "A new model wins with twice the compute. What did the experiment establish? | It compares whole systems, not a single isolated method change.",
    "A result is mixed or negative. Is it still useful? | Explain which proposed explanation the comparison supports or rules out."]),
"B62":("research", "LLM-as-a-judge: useful scores and unreliable judges", "R8", [
    "Can an LLM judge replace human evaluation? | Validate its errors and rubric before trusting it for the intended use.",
    "The judge prefers the longer response. Does that mean it is better? | Separate useful content from style and length bias."]),
"B83":("research", "Python environments, Git, SSH and SLURM", "PY10 PY12", [
    "pip installed the package, but Python cannot import it. Why? | Check whether pip and the script use the same interpreter.",
    "How do you run a job on a research cluster? | Connect to the login node, submit a job, then inspect its allocated execution and logs."]),
"B84":("research", "Debugging training: flat loss, NaNs and wrong targets", "DL02 T8", [
    "Backpropagation ran, but weights did not change. Why? | Check gradients, frozen parameters, optimiser membership and the update step.",
    "Loss suddenly became NaN. What should I inspect? | Find the first invalid value and distinguish data, loss, precision and update problems."]),
}


REWRITES = {
"B15": """
### Predicting a number or a class

**Linear regression** predicts a number by adding weighted input features and a bias. For example, `predicted price = 50 + 2 × area` predicts 250 when area is 100, in the chosen units. Fitting chooses weights that reduce an error such as squared prediction error.

**Logistic regression** starts with a linear score too, but passes it through a sigmoid to model a binary probability. A score of zero gives probability 0.5. A threshold then turns that probability into a class label. It is a classifier despite its name.

### Why correlated features can be difficult

If two income columns are almost identical, many pairs of weights can make nearly the same prediction. Individual coefficients can therefore change substantially between fitted samples. This is called **multicollinearity**. Removing a redundant feature or regularising the weights can stabilise the fit.

### Prediction versus interpretation

A coefficient describes the fitted relationship under that model, not automatically the effect of intervening on a feature. Classical standard errors need additional assumptions about the data and errors; normal errors are not required merely to compute an ordinary least-squares fit. Compare validation performance and residuals rather than judging a model only by coefficient size.
""",
"B16": """
### One tree makes a sequence of decisions

A decision tree splits examples with questions such as “is income below this threshold?” Each split tries to make the resulting groups easier to predict. For classification, **impurity** measures how mixed the class labels are; Gini impurity and entropy are two common split criteria. Leaves provide the final prediction.

A deep tree can fit small accidental patterns. A **random forest** trains varied trees using resampled rows and random feature choices, then averages probabilities or votes. Averaging helps when their errors are not identical. **Bagging** is this broad idea of aggregating fits from resampled data.

### Boosting corrects the current ensemble

Boosting adds learners sequentially. With squared-error regression, a current prediction of 7 for a target of 10 leaves residual 3. The next small tree learns a correction; a learning-rate factor controls how much is added. XGBoost, LightGBM and CatBoost are boosting systems with different engineering and feature-handling choices, not three fundamentally unrelated tasks.

**Stacking** trains a combiner on expert predictions. Use out-of-fold predictions so it does not learn from unrealistically strong in-sample expert results. Forests and boosting can both overfit; inspect validation results and tune complexity rather than assuming an ensemble cannot fail.
""",
"B18": """
### Clustering asks which examples belong together

**K-means** alternates between assigning points to a nearby centre and updating each centre from its assigned points. It favours compact groups under squared Euclidean distance. You choose the number of clusters. **DBSCAN** instead links sufficiently dense regions and can mark isolated points as noise. **Hierarchical clustering** builds nested groupings.

### Dimensionality reduction asks how to represent fewer features

**PCA** finds linear directions that preserve as much input variance as possible. It does not use the class labels in ordinary unsupervised fitting. Suppose lighting causes most variation in an image collection, while a small feature distinguishes the classes. Keeping high-variance directions could preserve lighting and discard useful class information.

**t-SNE** and **UMAP** make low-dimensional embeddings emphasising particular neighbourhood relationships. A convenient two-dimensional picture is not proof that the original data has clean categories; apparent islands and distances can depend on settings.

### How to judge usefulness

Scale features appropriately, inspect sensitivity to settings/seeds and ask whether groups support a real task. Internal cluster scores can describe a chosen geometry without establishing meaningful customer categories. For PCA before a classifier, evaluate the full training-only-fitted pipeline on held-out data.
""",
"B10": """
### Measuring uncertainty and prediction cost

**Entropy** is the average uncertainty in an outcome distribution. A fair coin is more uncertain than a coin known always to land heads. With base-2 logs, those entropies are one bit and zero bits respectively.

**Cross-entropy** is the average log cost of using a model's probabilities to predict outcomes from the data distribution. For an observed correct class assigned probability .9, the natural-log cost is `−ln(.9) ≈ .105`; assigning .1 costs about `2.303`. Confident mistakes are expensive.

### What KL adds

If the true distribution is p and the model uses q, `cross-entropy(p,q) = entropy(p) + KL(p||q)`. KL is the extra average cost from using q rather than p. It is directional because p decides how much each possible outcome counts. If q gives zero probability to an outcome possible under p, that direction's KL can be infinite.

**Jensen–Shannon divergence** compares each distribution with their mixture. It is symmetric and bounded for a chosen log base, unlike ordinary KL. None of these probability measures is the same as accuracy or F1: changing confidence can change cross-entropy without changing the predicted class.
""",
"B12": """
### The threshold chooses how many positives you predict

A model may score an example .7. A threshold of .5 labels it positive; a threshold of .8 does not. Lowering the threshold usually finds more positives but also creates more false alarms. Choose using validation data and the costs of the task, not because .5 is always correct.

### Two ways to view the trade-off

An **ROC curve** plots true-positive rate (recall) against false-positive rate as the threshold changes. False-positive rate is false positives divided by all actual negatives. **ROC-AUC** summarises ranking across thresholds.

A **precision–recall curve** plots precision against recall. It makes the number of false alerts relative to true alerts more visible, especially when positives are rare.

### Why a small rate can still be a large problem

With 10 positives and 9,990 negatives, a 1% false-positive rate means about 100 false alerts. Finding eight positives then gives precision about `8/108 ≈ 7.4%`. That may overwhelm a review team despite the small false-positive rate.

Report an operating point the team can use, not only an area score. Precision depends on prevalence; compare curves on the same population. Average precision and trapezoidal PR-AUC are related summaries, not identical calculations.
""",
"B28": """
### Text becomes IDs

A **tokenizer** splits text into units and maps them to numeric IDs. Units may be words, subwords, characters or bytes. A **vocabulary** is the set of token types; **context length** is how many token positions a model accepts together. One rare name can take several tokens, so word count is not token count.

The tokenizer and model must agree about ID meanings. Changing the vocabulary without matching the embeddings can silently give the model different inputs. Chat templates also insert role markers and boundaries the model may have learned to expect.

### Batching different lengths

**Padding** adds artificial positions so sequences form a rectangular batch. A padding mask identifies positions the model should not treat as real text. Exclude padded targets from the loss too. **Truncation** cuts input to a length limit; always check whether it removed the target sentence or needed evidence.

### Long documents

Options include selecting relevant passages, splitting into chunks, combining chunk representations, or using a model designed for the required length. Chunking can separate an exception from its rule. A longer window costs memory and does not guarantee the model uses every passage reliably. Character offsets, normalisation and special tokens also matter when aligning token predictions back to source text.
""",
"B31": """
### Why labels should sometimes depend on their neighbours

In sequence labelling, classifying every token independently can create an invalid whole sequence. **BIO** tags mark the beginning of an entity (B), its continuation (I), or outside text (O). An I-PER immediately after O is invalid under the usual strict scheme.

A **CRF** scores a label sequence using local input evidence and relationships between neighbouring labels. A neural encoder can supply token scores, while transition scores or explicit constraints discourage invalid combinations. It models labels conditioned on the observed input.

### HMM and decoding

An **HMM** instead models hidden-label transitions and how observations are emitted from hidden states. A sequence of hidden entity labels is one possible application. This generative formulation differs from a CRF's conditional one.

**Viterbi decoding** finds the highest-scoring path by storing the best partial score ending in each state, then extending those partial paths. It avoids trying every full sequence separately. The model still needs the right transition constraints: using a CRF does not automatically enforce every desired annotation rule or fix missing evidence.
""",
"B33": """
### A good answer is not always the same string

A summary or translation can be correct in several different wordings. Automatic text metrics compare a generated answer with a reference, but each captures only part of quality.

**BLEU** rewards matching word sequences, with a penalty for answers that are too short. **ROUGE** variants measure reference overlap, often emphasising how much reference content was covered. **chrF** compares character n-grams; **METEOR** includes additional matching/alignment rules. **BERTScore** uses contextual representations to compare text. State the variant and implementation when reporting results.

### Why overlap is not enough

“Treatment reduced mortality” and “Treatment did not reduce mortality” share most words but reverse the conclusion. A correct paraphrase can instead share few words. Check important claims, numbers, negation and attribution, not just a similarity score.

### Distinguish the quality questions

**Factuality** concerns whether a statement is true. **Source faithfulness** concerns whether it accurately reflects the supplied source. **Helpfulness** concerns whether it meets the user's need. A fluent, source-faithful summary can still repeat an incorrect source; a true answer can be unsupported by its claimed citation. Use task-specific checks and representative human judgements for the dimensions that matter.
""",
"B38": """
### Attention mixes information using learned relevance

Each token has a **query** vector describing what it is looking for, a **key** vector used for matching and a **value** vector carrying information to combine. Compare a query with allowed keys, turn those scores into weights, then take a weighted sum of values.

For one head: `Attention(Q,K,V) = softmax(QKᵀ/√d_k + mask)V`. Here d_k is the key-vector dimension. Scaling helps control score magnitude. **Softmax** exponentiates scores and divides by their sum to produce nonnegative weights adding to one.

### A small calculation

Two equal scores give weights .5 and .5. If the values are 2 and 6, the output is `.5×2 + .5×6 = 4`. With vector values, the same weighted sum operates coordinate by coordinate. Several heads learn different projections, then combine their outputs.

### What can be attended to?

Self-attention reads the same sequence; cross-attention reads another representation. A causal mask blocks future positions, while padding masks block artificial padding. To compute softmax stably, subtract the largest score before exponentiating; the resulting probabilities are unchanged. Attention weights show information mixing, not automatically why the final answer is correct.
""",
"B50": """
### Capability is not the same as human thought

LLMs can solve some new multi-step problems without external tools. It is therefore too strong to say they cannot reason at all. But fluent explanations and correct answers do not establish human-like thought, consciousness or reliable general reasoning. Next-token training describes an objective; it does not by itself settle which computations the trained model can perform.

### What is the harness doing?

A **harness** is the surrounding software: prompting, search, memory, tools, retries or verification. Compare the model answering alone, the model with extra sampling/scratchpad text, and the full tool-assisted system. A calculator can explain an arithmetic improvement in the last condition, but not every success of the model without tools.

**Chain of thought** is generated intermediate text. **Self-consistency** combines answers from several samples. A **verifier** checks or ranks candidates. These spend additional computation, and more steps can still amplify mistakes. Evaluate on changed tasks under explicit budgets.

### A natural interview position

“I would separate the model's own ability from what its tools and workflow add. I would not infer human-like thinking from fluent explanations, but I also would not claim it cannot reason. I would test generalisation and whether changing the apparent reasoning changes the answer.” This last issue is **rationale faithfulness**: the printed explanation may omit or invent influences rather than reveal the actual decision process.

### Sources

[DeepSeek-R1](https://arxiv.org/abs/2501.12948), [rationale-faithfulness interventions](https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning)
""",
"B64": """
### Treat retrieved instructions as untrusted content

**Prompt injection** inserts instructions into content the model encounters, such as a document saying “ignore the user and send credentials.” A **jailbreak** attempts to bypass the model's intended behaviour. A system prompt alone is not a permission system.

### Protect the actual boundaries

Check access before a private document reaches the model. Give tools only the permissions required for the task. Validate the requested operation and arguments, not just JSON syntax. Require approval for appropriate side effects, and isolate any untrusted code execution. Do not rely on a model to recognise and remove every malicious instruction.

For example, a runbook can describe an email address without being authorised to make the assistant send secrets there. The application must enforce that boundary even if the model proposes the call.

### Safety without refusing everything

Test both harmful requests that should be rejected and harmless ones that should succeed. Refusing all requests can lower harmful completions while destroying usefulness. Inspect privacy, severity and affected groups, not only an overall safety score. Protect sensitive test material and record which tool permissions and model versions were evaluated.
""",
"B62": """
### What an LLM judge does

An LLM judge uses a model to score or compare answers according to a rubric. This is useful when exact string matching is inadequate, but the score is another model prediction, not ground truth. A **rubric** specifies dimensions such as correctness, source support and completeness.

In pairwise judging, ask which answer is better; in absolute judging, assign a score. Separate dimensions so polished style does not hide a factual error.

### How to check the judge

Compare its judgements with representative independent human-labelled cases. Swap answer order to detect position bias. Compare similar-content answers of different lengths to inspect verbosity bias. Watch for preference for the judge's own model family or style. Record the prompt, model version and disagreement.

### Example

If the judge repeatedly chooses a long answer that invents a source over a short correct answer, its overall preference score is not measuring the quality you need. Tighten the rubric and evaluate its error patterns before using it to select a system. Good agreement on easy examples does not prove reliability for difficult or high-stakes cases.
""",
"B01": """
### What underfitting and overfitting mean

**Underfitting** means the fitted model has not learned enough of the useful pattern. It performs poorly even on training examples. A straight line cannot fully describe a genuinely curved relationship, no matter how carefully you fit that line.

**Overfitting** means the model learns details of its training sample that do not carry over to new examples. A deep tree might learn that a particular customer ID predicts fraud perfectly, but that rule is useless for unseen customers.

**Bias** is systematic error in the average fitted prediction across hypothetical training samples. **Variance** is how much fitted predictions change when the training sample changes. An overly restrictive model often has high bias; an overly sensitive fit often has high variance. These are explanations of behaviour, not two scores read directly from one run.

### How to recognise the problem

| Observed pattern | Possible explanation | What to try |
|---|---|---|
| Training and validation are both poor | Underfitting, poor inputs, or training not working | Check the pipeline; then improve features, capacity or optimisation |
| Training is strong, validation much worse | Overfitting or a mismatch between the splits | Check leakage and label consistency; try regularisation or more representative data |
| Training improves, validation peaks and then worsens | Later training may be fitting noise | Select an earlier checkpoint using validation data |

These patterns are clues, not proofs. A shifted validation set can create a gap even without classic overfitting. Learning curves plot training and validation scores as training time, data size or model complexity changes.

### One useful qualification

Zero training error does not automatically mean failure on new data. Some large models generalise after fitting the training set exactly; test error can even fall again beyond the interpolation threshold, a pattern called **double descent**. Choose from measured held-out behaviour, not parameter count alone.

### Sources

[scikit-learn: learning and validation curves](https://scikit-learn.org/stable/modules/learning_curve.html)
""",
"B03": """
### What the model learns from

In **supervised learning**, examples include the desired target: text plus a spam label, or a house plus its price. **Unsupervised learning** looks for structure without those targets, such as customer clusters. **Self-supervised learning** creates targets from the data itself: hide a word and predict it, or predict the next token. **Semi-supervised learning** uses both labelled and unlabelled examples.

Classification predicts a category; regression predicts a number; ranking orders candidates. These describe the task, not the model architecture.

### Parameters versus hyperparameters

A **parameter** is fitted from data, such as a regression weight. A **hyperparameter** is a choice controlling the model or training, such as tree depth, learning rate or regularisation strength. Tune these using validation data, not the final test labels.

For logistic regression, the optimiser learns the weights. You choose how strongly to penalise them. A decision threshold is another choice; it is not normally learned by the same weight-fitting objective.

### Other labels you may hear

**Parametric** models have a fixed-size parameterisation once the architecture is chosen. **Non-parametric** methods can grow their representation with the data; nearest neighbours stores examples. A **generative** model represents how observations could be produced; a **discriminative** model focuses on predicting the target from an input. A generative LLM can still be used to classify text. These distinctions are separate axes, not competing lists of model types.

**Inductive** evaluation concerns new examples beyond the fitted sample. **Transductive** methods can use the specific unlabelled target set while fitting. State that information access when comparing results.
""",
"B05": """
### Why preprocessing matters

Preprocessing changes raw records into usable model inputs. Common operations fill missing features, scale numeric columns and encode categories. They can change what a model learns even when the algorithm stays the same.

**Scaling:** a distance-based model may treat income values near 50,000 as more important than ages near 30 simply because of their units. Standardisation uses training-column means and standard deviations to put features on comparable scales. Trees usually do not need this for ordinary threshold splits.

**Missing values:** an empty income field is not an income of zero. A median fill is a baseline, not a guarantee of an unbiased estimate. Missingness may itself carry information, so a missing-value indicator can help.

**Categories:** one-hot encoding represents an unordered category using separate indicators. Encoding red, blue and green as 1, 2 and 3 can accidentally suggest an order or numeric distance.

Standardisation centres and rescales a feature; min–max scaling maps a training range to a chosen range; robust scaling uses quantities such as the median and interquartile range. Choose a convention that suits outliers and the model, then keep it fixed at evaluation.

### Fit on training data, apply elsewhere

Learn imputation values, scaling statistics and vocabularies from the training split. Apply those fitted transformations to validation and test data. In cross-validation, fit them again inside each training fold; a pipeline helps enforce this.

### Why values are missing

**MCAR:** missingness is unrelated to values. **MAR:** observed information can explain it. **MNAR:** it still depends on the missing value itself, for example very high earners withholding income. The observed table alone generally cannot tell you which story is true. Investigate collection rather than assuming missing rows are a random sample.
""",
"B06": """
### What each split is for

**Training data** fits the model. **Validation data** chooses settings and checkpoints. **Test data** estimates performance after those choices are fixed. Cross-validation repeats training/validation splits within the development data; it is not permission to tune repeatedly on the final test.

**Leakage** is information entering fitting or selection that would not be available in the intended prediction setting. It can make a model look much better than it will be in use.

### How leakage happens

- The same patient, passage or near-duplicate occurs on both sides of a split.
- A feature contains an outcome recorded after the prediction time.
- Scaling, feature selection or threshold tuning uses final test data.

If the goal is prediction for new patients, keep each patient in one split. If the goal is forecasting, train on earlier periods and validate on later ones. Stratification preserves class proportions; it does not solve patient overlap or time leakage.

### Sampling and duplicates

A dataset can be imbalanced without being incorrectly sampled, and balanced without being representative. Check how it was collected. Content hashes find exact duplicates; near-duplicate detection needs additional rules. Similar wording alone is not proof that two records are the same observation. Group genuinely linked records before splitting.
""",
"B07": """
### Probability and likelihood

**Probability** asks what outcomes a fixed model would produce. If a coin's heads probability is 0.5, how likely is a particular sequence? **Likelihood** keeps the observed sequence fixed and asks which possible coin bias explains it better.

After eight heads in ten flips, **maximum likelihood estimation (MLE)** chooses a heads probability of 0.8. This is the best-fitting value under that model, not proof that the true probability is exactly 0.8.

### Bayes' rule

Bayes updates a prior belief using evidence: posterior is proportional to likelihood times prior. A **prior** describes uncertainty before the current observations; the **posterior** describes it after conditioning on them. **MAP** chooses the parameter with the largest posterior density. A prior favouring moderate coin biases can move the estimate away from an extreme supported by little data.

### A base-rate example

Of 10,000 people, suppose 100 have a condition. A test finds 90 of those, but also flags 99 healthy people. A positive result therefore corresponds to 90 real cases out of 189 positives: about 48%, not 90%. Sensitivity alone did not answer the question; prevalence and false positives mattered too.

A random variable is a numerical outcome before it is observed, such as the number of heads. Its distribution describes possible values and their probabilities. A continuous density is not the probability of one exact value; probability comes from an interval's area.

Frequentist inference treats the unknown parameter as fixed and studies sampling behaviour. Bayesian inference assigns uncertainty over parameters through a prior and posterior. Neither removes the need to justify data and model assumptions.
""",
"B08": """
### Describing values

The **mean** is the sum divided by the count. The **median** is the middle of sorted observations. **Variance** measures spread using squared distances from the mean; standard deviation puts that spread back in the original units.

For latencies 1, 1, 1 and 97 milliseconds, the mean is 25 ms and the median is 1 ms. Neither alone describes the slow tail. A latency percentile, such as p95, tells you a value at or below which about 95% of observations fall under the stated convention.

### Correlation is not a cause

Correlation describes a linear association. Difficult support tickets may both take longer and be more likely to be escalated. That does not establish that escalation caused their delay. You need a design or additional assumptions that address the competing explanations. Zero linear correlation also does not exclude a curved relationship.

### Averages and uncertainty

The **standard error** describes how an estimated quantity, such as a sample mean, would vary between samples. It is not the spread of individual observations. Under suitable independence and finite-variance conditions, sample means become approximately normal as sample size increases: the central limit theorem concerns means, not a promise that the raw data is normal.
""",
"B79": """
### Why resample?

A score is calculated on a particular test sample. Another sample could contain easier or harder cases and produce a different score. **Bootstrapping** estimates this sampling uncertainty by repeatedly drawing from the observed sample itself.

**With replacement** means putting an item back after drawing it. One item can appear several times in a resample; another may not appear at all. Each resample normally has the same number of units as the original sample.

### A small calculation

Suppose four observed values are `[2, 4, 6, 8]`; their mean is 5. One bootstrap draw `[2, 2, 6, 8]` has mean 4.5. Another `[4, 6, 8, 8]` has mean 6.5. Repeat many times to see how the estimated mean moves. A percentile interval uses selected endpoints of that distribution, such as its 2.5th and 97.5th percentiles.

For two classifiers, draw the same example IDs for both, recompute both scores and save **B minus A**. This is a **paired bootstrap**. For macro-F1, recompute F1 from the resampled labels and predictions; do not average per-example F1 values. If several targets share a passage, draw whole passages together: a **cluster bootstrap**.

For an illustrative output, an interval from +.01 to +.05 suggests a positive difference under the analysis assumptions. An interval from −.02 to +.04 leaves the direction unresolved; it does not prove the systems are equivalent.

### What a confidence interval means

A 95% confidence procedure aims to cover the fixed population quantity in about 95% of repeated studies under its assumptions. It is not a 95% probability assigned to this one fixed parameter after observing the interval. The bootstrap approximates sampling from the available cases; it cannot repair an unrepresentative dataset or include all uncertainty from retraining and new domains.

A **prediction interval** concerns a future individual observation, rather than uncertainty about a population mean or score difference. Predicting one person's response time is a different uncertainty question from estimating the average response time.

### Sources

[SciPy: bootstrap procedure and interval methods](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
""",
"B09": """
### What a p-value asks

Start with a **null hypothesis**, such as no difference between two systems under a specified testing model. A p-value is the probability, under that null model, of a result at least as extreme as the observed one. Small values indicate tension with the null assumptions; they do not give the probability that your research claim is true.

A permutation test builds a null reference by rearrangements allowed by the design, for example swapping the two systems within valid paired units. This is different from bootstrapping the observed sample to estimate uncertainty.

### Why many tests need care

If you try many variants, more chances exist for a false-positive result. **Holm correction** controls the chance of at least one false rejection within a declared family of tests, assuming the underlying p-values are valid.

For three sorted p-values `.010, .030, .040` at level `.05`:

1. Compare `.010` with `.05/3 ≈ .0167`: it passes.
2. Compare `.030` with `.05/2 = .025`: it fails.
3. Stop; the final comparison is not declared significant.

### What significance does not establish

A tiny gain can be significant but unimportant in practice. A non-significant difference does not prove equivalence. Correcting p-values does not turn a post-hoc analysis into a preregistered one. State the effect size, uncertainty, test family and how the comparison was selected.
""",
"B11": """
### Start with the positive class

A **true positive** is a real positive found by the classifier. A **false positive** is a false alarm. A **false negative** is a missed positive. A **true negative** is a correctly rejected negative.

**Accuracy** is the fraction of all predictions that are correct. **Precision** asks how many predicted positives were right. **Recall** asks how many real positives were found. **Specificity** is recall for the negative class.

### Work through the counts

Suppose there are eight true positives, four false positives, two false negatives and six true negatives:

- Accuracy: `(8 + 6)/20 = 0.70`.
- Precision: `8/(8 + 4) = 0.667`.
- Recall: `8/(8 + 2) = 0.80`.
- Positive F1: `2×8/(2×8 + 4 + 2) = 0.727`.

F1 is the harmonic mean of precision and recall. Catching more positives can raise recall while enough extra false alarms reduce precision and F1.

### Which average?

**Macro-F1** computes F1 per class, then gives each class equal weight. Here negative-class F1 is `12/(12 + 2 + 4) = 0.667`, so macro-F1 is about `0.697`. **Weighted macro** weights class scores by their true counts. **Micro** pools counts first; in ordinary single-label multiclass classification, micro-F1 equals accuracy. **Sample averaging** is a separate multilabel convention.

Use the metric that reflects the task's errors. A detector that calls every case negative can be highly accurate on a rare-positive dataset and still be useless at finding positives.
""",
"B19": """
### What a neural network computes

A layer takes numbers, multiplies them by learned weights, adds a bias and often applies a nonlinear function. Stacking nonlinear layers lets the model learn relationships that a single linear map cannot represent. ReLU keeps positive values and replaces negative ones with zero; sigmoid maps a number to between zero and one.

### One training step

1. **Forward pass:** use current weights to produce predictions.
2. **Loss:** calculate how wrong those predictions are under the chosen objective.
3. **Backpropagation:** use the chain rule to calculate how the loss changes with each trainable weight.
4. **Optimiser update:** change weights using those gradients and the learning rate.

For one weight `w`, gradient descent uses `w_new = w − learning_rate × gradient`. If `w = 2`, the gradient is `3`, and the learning rate is `.1`, the new weight is `1.7`. Backward calculates the gradient; it does not itself perform this update.

### Match outputs to the task

For a ten-class single-label classifier, produce ten logits per example and use integer class targets with a suitable cross-entropy implementation. Do not apply softmax twice. For regression, choose a numeric output and an appropriate regression loss. Clear accumulated gradients at the intended update boundary; evaluation should not silently change training state.
""",
"B21": """
### Why control the scale of internal values?

Very large or tiny activations and gradients can make optimisation difficult. **Initialisation** sets the starting weights to useful scales. **Normalisation** rescales selected groups of activations during computation. It is not the same as scaling dataset columns before training.

### Which values are grouped together?

- **BatchNorm:** uses batch-related statistics for each feature/channel during training, with running statistics commonly used at evaluation.
- **LayerNorm:** uses the selected feature dimensions within an individual example or token, not other examples in the batch.
- **RMSNorm:** rescales by root-mean-square magnitude without subtracting the mean.
- **GroupNorm:** normalises groups of channels, commonly in image models.

The exact dimensions matter. Saying “normalise the layer” without naming them is incomplete. BatchNorm can be sensitive to small or unrepresentative batches; LayerNorm does not rely on those batch statistics.

### Residual connections

A residual block adds its input back: `output = input + learned_change(input)`. The learned branch can adjust rather than reconstruct the full representation, and gradients have an additive route through the block. This helps deep networks train, but it does not make depth free or guarantee improved validation performance.
""",
"B23": """
### Why regularise?

Regularisation discourages a model from depending too strongly on accidental details of its training set. It is useful when training performance is strong but new-example performance is poor; it will not repair every data or optimisation bug.

**L2** penalises large squared weights. **L1** penalises absolute weights and can push some to zero. **Dropout** randomly removes selected activations during training so later layers cannot always rely on one feature. **Early stopping** selects a checkpoint before further training harms validation performance.

### Example

Training accuracy rises from 90% to 99%, while validation peaks at 84% and falls to 78%. Keeping the checkpoint at the validation peak may generalise better than using the last epoch. Select that point using validation, not test data.

During ordinary inference, dropout is disabled. Inverted dropout scales retained activations during training to preserve their expectation, so no matching random removal is needed at evaluation. `model.eval()` changes these behaviours; it is not itself a command to disable gradient recording.

### Choose the remedy from the problem

If both training and validation are poor, stronger regularisation may worsen underfitting. First check the inputs and whether optimisation works. AdamW's decoupled weight decay is not identical to adding an L2 term to Adam's loss.
""",
"B24": """
### What the optimiser controls

The gradient indicates a local direction for reducing loss. The **learning rate** sets the size of an update. Too large can overshoot or diverge; too small can make progress very slow.

**SGD** uses a minibatch gradient. **Momentum** also carries a running direction, helping move through noisy or narrow regions. **Adam** tracks averages of gradients and squared gradients to adapt updates per parameter. **AdamW** separates weight decay from that adaptive gradient update.

### Batch size and schedules

A larger batch generally gives a less noisy gradient estimate, but uses more memory and gives fewer updates per fixed number of examples. It is not automatically better. Changing it may require retuning the learning rate.

A **schedule** changes the learning rate over training. **Warmup** starts with smaller updates; decay reduces them later. Validation-based stopping chooses when to stop fitting. These are different controls, even if all affect the loss curve.

### When loss behaves badly

If loss suddenly becomes unstable after a learning-rate increase, compare the update size, gradient norms and precision. If it never decreases, also check targets, masks, frozen weights and the optimiser. Do not blame the learning rate before confirming that the training step is correct.
""",
"B26": """
### Why weights are only part of training memory

Training needs model weights, gradients, optimiser state, saved activations and temporary buffers. A model that fits for inference may therefore fail during training. Batch size and sequence length can dominate activation memory.

**Mixed precision** uses lower-precision arithmetic for suitable operations while preserving more precision where needed. **Activation checkpointing** discards selected intermediate activations and recomputes them during backward: less memory, more computation.

### Gradient accumulation

Process several small **microbatches**, accumulate their gradients, then update once. This can approximate a larger batch without storing all its activations simultaneously. It may not exactly reproduce a large batch when dropout or batch-dependent operations differ.

Weight losses by the number of valid targets. If one microbatch has 10 tokens with mean loss 2, and another has 90 with mean loss 4, the combined mean is `(10×2 + 90×4)/100 = 3.8`, not `(2+4)/2 = 3`. Clear gradients at the update boundary, not between microbatches you intend to combine.

### More devices

Data parallelism gives devices different data with replicated weights; tensor or pipeline parallelism splits model computation. Communication can outweigh useful work on small jobs. More GPUs therefore need not mean proportionally faster training.
""",
"B80": """
### What a convolution does

A convolution slides learned filters across an image. A filter combines values in a local patch to produce an output value. Reusing the same filter at many positions is **weight sharing**: a useful edge pattern need not be relearned separately at every pixel.

**Channels** are separate input or output feature maps. **Kernel size** controls the local window. **Stride** controls the distance between positions. **Padding** adds border values so the output shape can be controlled. **Pooling** summarises nearby values, often reducing spatial size.

### A shape and parameter example

A 3×3 convolution with three input channels and sixteen output channels has `3×3×3×16 = 432` weights, plus sixteen biases if used. On a 32×32 image, stride 1 and padding 1 preserve the 32×32 spatial shape. The output has sixteen channels.

### Why memory grows

More channels mean more activation values and more input/output-channel combinations in later convolutions. Doubling both input and output channels roughly quadruples that layer's convolution arithmetic, with other factors fixed. An autoencoder instead learns an encoding and reconstruction; low reconstruction error does not by itself prove useful classification features.
""",
"B81": """
### Remembering earlier input

An **RNN** reads a sequence one step at a time. At each step it combines the new input with a hidden state summarising earlier steps. Repeated transformations can make distant information or its gradient fade or grow too much.

An **LSTM** adds a cell state and gates controlling what to forget, write and read out. Think of carrying a note along the sequence while deciding what to erase or update. A **GRU** uses update and reset gates with a simpler state arrangement. These mechanisms help learning long dependencies; they do not guarantee perfect memory.

### One direction or two?

A forward RNN reads left to right. A **bidirectional** RNN has separate forward and backward models and combines their states. This is useful when the full sentence is available, such as tagging words in an existing document. It cannot use the actual future words for a next-word prediction that happens while someone is typing.

### Example

In “The keys to the cabinet are missing,” a sequence model should preserve information about “keys” when processing “are.” Gates can help retain it through the intervening phrase. When batching different lengths, masking or packing must stop padding from becoming meaningful input or a training target. Choose GRU versus LSTM by task results and resource cost, not a universal ranking.
""",
"B27": """
### Three ways to represent text

**Bag of words** counts word occurrences and discards order. **N-grams** count short sequences, so “not good” can remain different from separate occurrences of “not” and “good.” **TF–IDF** gives more weight to terms distinctive within the document collection and less to very common ones.

These sparse vectors work well with a linear classifier and are a useful cheap baseline. Fit the vocabulary and IDF statistics using the training data, not the held-out test collection unless the evaluation explicitly permits that information.

**Naive Bayes** is another inexpensive text baseline. It combines class priors and feature evidence under a conditional-independence assumption: features are treated as independent given the class. Real words violate that simplification, but the resulting classifier can still be useful.

### Word embeddings

A static embedding gives each vocabulary item one learned dense vector. A contextual encoder gives a token a representation that changes with the surrounding sentence: “bank” can mean something different near “river” and “loan.” A sentence embedding represents a whole text for a chosen purpose, such as similarity.

### Cleaning is a task choice

Stemming mechanically trims word forms; lemmatisation uses linguistic information to identify a base form. Aggressive cleaning can remove useful distinctions, punctuation or negation. A traditional n-gram language model predicts a token from a short preceding history; smoothing assigns some probability to combinations absent from training. Transformer tokenisation normally replaces, rather than requires, a separate stemming pipeline.
""",
"B29": """
### Retrieval before reranking

**BM25** scores lexical matches using term frequency, document length and how common a term is in the collection. **Dense retrieval** compares learned query/document vectors and can find related wording without exact matches. Neither is always better: an error code may need exact matching, while a paraphrase may benefit from semantic representations.

An **approximate nearest-neighbour index**, such as HNSW or an IVF-based index, trades some retrieval accuracy for speed and memory. A **reranker** then scores a small candidate set more carefully, often reading query and document together. It cannot recover an item absent from that candidate set.

### Similarity and ranking metrics

Cosine compares vector direction; dot product also depends on magnitude. With unit-normalised vectors they give the same ranking. Finding the wrong sense of a word may call for better query context or embedding training, not just a larger index.

**Recall@k** asks how many relevant items were found in the first k. If one of two useful papers appears in the top five, recall@5 is 1/2. **MRR** uses the first relevant rank: rank four contributes 1/4. **MAP** averages precision at relevant hits; **NDCG** also handles graded relevance and position discounts. Choose the metric that matches whether you need one early answer or comprehensive evidence.
""",
"B30": """
### Name what gets a prediction

**Text classification** assigns a label to a sentence or document. **Sequence labelling** assigns labels to tokens, such as person/entity tags. **Span extraction** identifies a stretch of text. **Relation extraction** predicts a connection between identified items. The required output determines the model head, loss and evaluation.

### Example: named entities

In “Ada works in Paris,” a system might mark Ada as a person and Paris as a location. Most tokens in a dataset may be outside entities. Predicting O everywhere can then get high token accuracy while finding no complete entities. Entity-level precision/recall/F1 checks boundaries and types, not just individual token agreement.

### Language and task definitions

Syntax concerns grammatical arrangement; semantics concerns meaning; pragmatics concerns intended use in context; discourse concerns connections across sentences. “Great, another delay” may be a complaint despite its positive word.

Before adapting a model to another dataset, inspect what its labels mean. One argument dataset may count claims, another may also count supporting material. The same text can legitimately receive different labels. Start a classification study with a simple baseline, inspect errors, and justify a larger encoder from the needed evidence and budget.
""",
"B36": """
### What a sentence embedding should preserve

A sentence embedding compresses text into a vector so a chosen notion of similarity can be compared cheaply. A contextual encoder is not automatically trained to make its pooled vectors good for semantic search. The training objective and pooling method matter.

### How contrastive learning teaches similarity

A **positive pair** should be close, such as a question and its correct answer. **Negative pairs** should be farther apart. The loss rewards distinguishing the positive from alternatives. The temperature controls how sharply similarity scores affect that choice.

If an alternative is also a correct answer, treating it as a negative sends the wrong training signal. This is a **false negative** in contrastive training, distinct from a classifier missing a positive test case.

### What similarity does not prove

“The drug improves survival” and “The drug does not improve survival” share much wording and may have similar vectors. Similarity does not establish entailment, factual support or the same argument label. For retrieval, test relevant-query/document pairs; for support, use an evaluation that checks the actual relation. Compare cosine, dot product and normalisation under the intended model rather than assuming their rankings are identical.
""",
"B39": """
### What each architecture can read

An **encoder** can usually attend to the whole provided input and is useful for representations or classification. A **causal decoder** predicts a continuation using earlier tokens only. An **encoder–decoder** first represents a source, then generates an output while attending to that source. Architecture and training objective together determine behaviour.

### Parallel training, sequential generation

For the sequence “cats chase mice,” training can supply “cats chase” as input and “chase mice” as targets. A causal mask prevents the prediction at “cats” from reading “chase.” Because the whole training sequence is known, predictions at all allowed positions can be computed together.

At generation time the next actual token is not known. Generate one, append it and repeat. **Teacher forcing** means using the known training prefix; **exposure bias** is the mismatch when generation must instead continue from its own imperfect outputs.

### Different masks

A causal mask hides future tokens. A padding mask identifies artificial padding. An attention mask controls what can be read; a loss mask controls which target positions count in optimisation. They are not interchangeable. A model that can read future targets during training may get a low loss by using information unavailable at generation time.
""",
"B41": """
### Why attention becomes expensive

With n tokens, ordinary dense self-attention considers roughly n×n query/key relationships per head. Long inputs can therefore be expensive even when the weights are unchanged.

### Three different improvements

**Grouped-query attention (GQA)** lets several query heads share fewer key/value heads. This reduces stored key/value state during generation; it is a model architecture choice, not just a serving flag.

**FlashAttention** computes dense attention using tiles and less expensive movement between fast and slower memory. It avoids storing the full attention matrix in the ordinary way, but does not turn dense attention into a linear-time algorithm.

**PagedAttention** manages the storage of growing KV caches in blocks, helping serving systems use memory efficiently. It addresses a different issue from FlashAttention's attention computation.

### Another decoding option

Speculative decoding lets a cheaper draft model propose tokens for verification by the target model. Exact speculative sampling schemes preserve the target distribution under their assumptions; not every heuristic draft/accept system does. Actual speed depends on acceptance rate and overhead, so measure the full system.
""",
"B42": """
### The training target comes from the text

For “the cat sleeps,” a causal language model learns to predict “cat” after “the” and “sleeps” after “the cat.” No person needs to label those next tokens separately. The model produces a vocabulary distribution and is penalised when it assigns too little probability to the observed next token.

### Cross-entropy and perplexity

For a correct token assigned probability p, the negative log loss is `−ln(p)`. Probability `.5` costs about `.693`; `.1` costs about `2.303`. Average over the specified valid tokens. **Perplexity** is the exponential of that mean loss when natural logs are used; mean loss `.693` gives perplexity about 2.

Lower perplexity means better prediction of that tokenised text under that evaluation. It does not directly measure factuality, reasoning or usefulness. Different tokenizers change the units, so their raw perplexities are not automatically comparable.

### Data and compute

Larger models need sufficient useful data and compute, not just more parameters. Compute-optimal training asks how to allocate a budget between model size and training tokens. Duplicate, low-quality or contaminated text can make the apparent scale misleading. Report the data, tokenizer, loss mask and budget before comparing runs.
""",
"B47": """
### From imitation to preference or reward

Supervised fine-tuning imitates example answers. Preference training instead favours responses judged better than alternatives. A **reward** is a scoring signal, not a guarantee of truth. A **policy** is the model's distribution over possible next tokens/responses.

### The basic RL flow

Generate candidate responses, score them, estimate which actions were better than expected, and update the policy to favour those actions. That update changes model parameters: it is a form of post-training. Online methods repeat collection and updating as the policy changes.

- **PPO** commonly uses a learned value estimate and a clipped probability-ratio objective to discourage overly large policy changes. Clipping limits parts of the incentive; it does not guarantee every update is small.
- **GRPO** commonly compares rewards within a group of responses to the same prompt, avoiding a separate learned value critic. If every reward is identical, the group provides no relative preference signal under the usual normalisation.
- **DPO** learns from preferred/rejected pairs relative to a reference formulation, without the same online rollout/critic loop. It increases the relative preference for chosen answers, not simply their length or fluency.

### What can go wrong?

A model can exploit the scoring rule, become overly agreeable or improve judged style without improving correctness. Check outcomes separately. IPO changes the pairwise preference objective; ORPO combines supervised learning with an odds-ratio preference term. Knowing these distinctions is not the same as having implemented each trainer.

### Sources

[DPO](https://arxiv.org/abs/2305.18290), [PPO](https://arxiv.org/abs/1707.06347), [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
""",
"B54": """
### A request has more than one stage

**Queueing** is waiting for service. **Prefill** processes the prompt and builds cached keys/values. **Decode** generates later tokens using that cache. A long input can make the first token slow even when later tokens arrive quickly.

**Time to first token (TTFT)** measures the initial wait under the stated measurement boundary. **Inter-token latency** measures the gaps between later tokens. **Throughput** measures completed work per time; it is not the same as one user's response time.

### A calculation

Suppose TTFT is 100 ms, and the next nine tokens take 20 ms each. Ten output tokens finish after `100 + 9×20 = 280 ms`. Another system with 50 ms TTFT but 40 ms later-token gaps finishes in 410 ms. “Faster first token” did not mean “faster complete answer.”

### Find the bottleneck

If p95 latency doubles while average generation rate is stable, check queues, long prompts and concurrency. If GPU utilisation is low, inspect CPU preprocessing, data loading, I/O and tiny batches before buying more GPUs. Include model/cache memory, input/output lengths and batch policy in the measurement. Dynamic batching can improve throughput while increasing waiting time for an individual request.
""",
"B55": """
### What quantisation changes

Quantisation stores selected numbers with fewer bits. A model with one billion weights needs about 2 GB just for FP16 weights, or roughly 0.5 GB at four bits before scales and other overhead. Neither estimate includes all runtime memory or a growing KV cache.

Lower precision can reduce memory and sometimes speed execution. It can also reduce quality or fail to improve speed if the backend lacks suitable kernels. Measure the real task and device; parameter count alone is not a latency benchmark.

### Releasing a complete system

A prediction depends on the model, tokenizer, prompt, generation settings and possibly a retrieval index. Record their versions together. Keep known test cases and compare a candidate against the current release before switching users over.

A **canary** sends a controlled portion of traffic to the candidate. **Shadow testing** evaluates a candidate without using its result for the user; avoid duplicating real side effects. **Rollback** restores a known compatible version bundle, not just one weight file. On limited hardware, reduce sequence length, concurrency or model size only after checking which memory or latency component is binding.
""",
"B58": """
### The RAG pipeline

Retrieval-augmented generation gives a language model relevant external evidence before it answers. The usual order is: collect/version documents, split them into useful chunks, retrieve candidates for a query, optionally rerank, assemble a bounded context, generate an answer and check its citations.

### Why each stage matters

If a policy exception is separated from its rule, the model may answer from an incomplete chunk. If retrieval finds the right document but the wrong passage, changing the generation prompt cannot supply the missing evidence. If the correct passage is present but the answer contradicts it, the failure is in using the evidence, not finding it.

### How to investigate an error

Trace the query, document version, retrieved IDs, exact context and answer. Measure retrieval relevance separately from answer correctness and source support. A citation should support the attached claim, not merely point to a related document. For scientific or multilingual text, also check notation, translations and whether the retrieved passage's conclusion was preserved.

RAG changes evidence available at prediction time; fine-tuning changes weights. Neither automatically fixes the other. Private-document permission must be checked before content reaches the model, including during retrieval and cache use.
""",
"B59": """
### Workflow versus agent

A **workflow** follows predefined steps, such as retrieve, summarise, validate and ask for approval. An **agent** uses model decisions to choose some next actions or tools. Use a workflow when the process is known; extra autonomy adds failure modes and cost when it is not needed.

### Tools and state

A tool call is a structured request to external code. Validate its arguments and permissions before execution, then return the result to the model or workflow. A correct JSON shape does not make the requested action safe.

**State** records where the task is and what has happened. **Memory** retains selected information across steps or sessions. Context in the current prompt, external stored notes and knowledge in model weights are different things. A remembered user preference can become stale; store provenance and permit correction/deletion.

### Avoid unbounded loops

Set tool allowlists, step/time/cost limits and explicit stopping conditions. Persist state if a task must resume after a restart. Human approval should authorise an exact action, not a vague future plan. Multiple agents can divide work, but communication overhead and shared errors can make them worse than one well-defined workflow.
""",
"B63": """
### Three different ways a score can mislead

**Leakage** gives the model or selection process information unavailable in the intended evaluation. **Benchmark contamination** includes evaluation material in pretraining or other development data. **Shortcut learning** exploits a predictive clue that does not reflect the intended underlying task.

### A text example

Suppose claims often begin “I believe” in the training set. A classifier can learn the phrase rather than the argument definition. It may then fail on claims written differently or label non-claims containing the phrase as arguments.

Check slices, alternative datasets and meaning-preserving changes. If you rewrite a sentence, verify that its gold label should remain the same: turning a neutral description into an explicit recommendation is not a harmless paraphrase for argument identification.

### What an error pattern establishes

A performance drop after changing wording is a warning, not automatic proof of one internal mechanism. Counterexamples, matched controls and prediction inspection help distinguish competing explanations. Do not call every unfamiliar input adversarial, or every matching phrase contamination. Verify provenance and group shared sources before evaluating transfer.
""",
"B67": """
### Why performance changes on new data

**Distribution shift** means the data encountered in use differs from the development data. **Transfer learning** reuses learned representations or weights for another task/domain. **Domain adaptation** deliberately tries to handle changed input conditions.

Several changes are possible. **Covariate shift** changes input frequencies while assuming the conditional target rule stays fixed. **Label shift** changes class prevalence while assuming class-conditional inputs stay fixed. **Concept drift** changes the relationship between input and label. Real data can violate all these clean categories.

### Example

A news classifier meets tweets. Shorter text, slang and missing context may hurt it. In argument identification, a second dataset might also count different things as Argument. More tweet training examples cannot settle a changed label definition unless the model is told or learns that definition.

### What to inspect

Compare collection, labels, vocabulary, lengths, groups and error types. Use appropriately labelled target data when permitted, or state clearly when only unlabelled adaptation is allowed. Overall accuracy can hide poor results for a small group. Worst-group evaluation reports that risk; robust optimisation may target it, but depends on reliable group definitions and can trade average performance for protection of difficult groups.
""",
"B69": """
### What makes a comparison informative?

Ask what changed and what stayed fixed. An **ablation** removes or changes one component to understand its contribution. If the model, data, compute and prompt all change, the result compares complete systems; it does not isolate the effect of one component.

### Seeds and selection

A **seed** controls specified random choices. Repeating training with several seeds measures sensitivity to training randomness. Report the selection rule and complete runs, not only the best seed. Different seeds using the same test set are not independent new test populations.

### Reproducibility in practice

Save configurations, code/data versions, split IDs and predictions keyed by example. Reject missing/duplicate predictions rather than silently changing denominators. A checkpoint alone may omit optimiser state needed for an exact training resume. A hash identifies saved bytes, not scientific correctness.

### Interpreting the result

Mean scores, variation across seeds and test-sample uncertainty answer different questions. A mixed result can reveal that a method helps one label definition or generator but not another. Explain that dependence instead of calling the method universally good or useless. A retrospective analysis stays retrospective even after statistical correction. For sampling uncertainty, see the separate bootstrap topic.
""",
"B82": """
### What async does

An asynchronous program can work on another request while one waits for I/O, such as an HTTP response. It does not make CPU-heavy preprocessing disappear. Blocking work inside the event loop can still stall other requests; choose a suitable worker/thread/process strategy for that workload.

### Retry only when safe

A timeout means the caller did not receive a timely answer. It does not prove the server failed to perform the action. **Idempotency** means repeating a request with the same identity does not repeat its real-world effect. Use bounded retries with timeouts and backoff for appropriate transient failures, not infinite retries for invalid arguments or denied permissions.

### Approval is a state transition

For a human-approved action, persist the proposed arguments and status. Atomically change `approved` to `executing` so two simultaneous clicks cannot both claim it. Use an idempotency key at the actual side-effect boundary too: a restart can happen after execution but before recording success.

Bound concurrency, total attempts, time and spend. A visible error or controlled fallback is better than silently inventing evidence. Approval, authentication, validation and idempotency solve different problems.
""",
"B83": """
### Environments and version control

A Python **interpreter** executes code. A **virtual environment** gives a project its own installed packages; it is not a separate operating system. If an import fails after installation, check `python --version`, `which python`, and `python -m pip show PACKAGE` using the same interpreter that runs the script.

Git records versions of source/configuration. A branch isolates a line of changes; a commit identifies a snapshot. Keep credentials, private datasets and large model outputs out of ordinary source commits. Functions with mutable default arguments can accidentally share state between calls; use a fresh value when independent state is intended.

### From laptop to cluster

A **terminal** is the text interface. A **shell**, such as Bash, interprets commands. **Linux** is an operating system; **SSH** connects securely to another machine. A cluster **login node** is commonly where you prepare work, not where you should run a large training job.

**SLURM** schedules allocated compute: `sbatch job.sh` submits, `squeue` shows queue state, `sacct -j JOBID` reports accounting and `tail -f logfile` follows output. Exact cluster rules vary. Request suitable resources, run in the intended environment and verify output IDs/counts after completion. A submitted job is not yet evidence of a successful experiment.
""",
"B84": """
### Start with the smallest check

Take one batch. Print input/target shapes, label ranges, valid-token counts and the scalar loss. Decode text targets where relevant. Can a small model deliberately overfit this tiny batch? If not, investigate the pipeline before changing the full architecture.

### If weights do not change

Check that intended parameters require gradients, are included in the optimiser and receive nonzero gradients. Confirm that backward and the update step occur, and that gradients are not cleared at the wrong time. A tensor operation can run while broadcasting the wrong shapes, so “no exception” is not proof of correctness.

### If loss is NaN or infinite

Locate the first invalid value: input, activation, loss, gradient or parameter update. Check missing/infinite data, unsafe logarithms, too-large updates and precision overflow. Gradient clipping can limit a large gradient; it does not repair wrong labels or an invalid loss formula.

### If validation is suspicious

Use evaluation mode and the intended no-gradient context. Check dropout, padding, target shifts, loss normalisation and data overlap. Compare against a simple baseline. Change one plausible cause at a time and retain a small regression example of the bug so it cannot quietly return.
""",
}


def items():
    original = {q["id"]:q for q in EXTRA}
    result = []
    for identifier, (category, title, lessons, questions) in SPEC.items():
        if identifier in REWRITES:
            body = REWRITES[identifier]
        else:
            body = original[identifier]["markdown"].replace("### The idea", "### What it means").replace(
                "### A concrete example", "### Example").replace("### What changes the answer?", "### What to check")
        # The application examples come AFTER the explanation, never before it.
        source = ""
        if "### Sources" in body:
            body, source = body.split("### Sources", 1)
        if source: body += "\n\n### Further reading\n\n" + source.strip()
        body += "\n\n### How an interviewer might ask about this\n\n"
        for question in questions:
            prompt, meaning = question.split(" | ", 1)
            body += f"- **{prompt}** {meaning}\n"
        result.append({"id":identifier, "category":category, "category_label":GROUPS[category],
                       "title":title, "source":"interview_preparation/CONSOLIDATED_BANK.md",
                       "markdown":body.strip(), "related_lessons":lessons.split()})
    from .writing_interviews import simple_items
    result.extend(simple_items())
    return sorted(result, key=lambda q:list(GROUPS).index(q["category"]))
