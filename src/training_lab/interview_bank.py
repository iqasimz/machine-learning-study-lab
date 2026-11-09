"""Authored concept answers and explicit routing for the supplied 335 prompts.

Question wording is a checklist supplied by the user, not scientific evidence.
Legacy scenario answers keep their IDs. Coding tasks reuse the Python course.
"""
import re
from .artifacts import ROOT

EXTRA = []
ROUTES = {}
LABELS = {"concept-ml": "ML concepts", "statistics": "Statistics and metrics",
          "concept-dl": "Neural networks and optimisation", "concept-nlp": "NLP concepts",
          "architecture": "Transformers and training", "reasoning": "Reasoning and post-training",
          "serving": "Inference and infrastructure", "research": "Research judgement",
          "writing": "Writing interviews", "writing-course": "Writing course", "systems": "Applications and safety", "design": "System-design conversations"}


def numbers(spec):
    result = []
    for part in spec.split(","):
        if not part.strip(): continue
        bounds = part.split("-")
        result.extend(range(int(bounds[0]), int(bounds[-1]) + 1))
    return result


def route(spec, target, kind="question"):
    for number in numbers(spec):
        if number in ROUTES: raise ValueError("Prompt mapped twice: " + str(number))
        ROUTES[number] = {"kind": kind, "target": target}


def add(identifier, category, spec, title, meaning, example, boundary, lessons="", sources=""):
    body = f"### The idea\n\n{meaning}\n\n### A concrete example\n\n{example}\n\n### What changes the answer?\n\n{boundary}"
    if sources: body += "\n\n### Sources\n\n" + sources
    EXTRA.append({"id": identifier, "title": title, "category": category,
                  "category_label": LABELS[category], "source": "interview_bank.py",
                  "markdown": body, "related_lessons": lessons.split(), "incoming": numbers(spec)})
    route(spec, identifier)


add("B01", "concept-ml", "1-3,9,11", "Bias, variance and learning curves",
    "Bias is systematic error from the assumptions a learner makes; variance is sensitivity to the particular training sample. Imagine repeating the whole training study on new samples. A model can systematically miss a curved pattern and still change a lot between samples. Overfitting means training success fails to carry over to new cases; underfitting means the fitted rule misses useful signal even on training data.",
    "A straight line fitted to a curved relationship can underfit. A very flexible curve through ten noisy points may fit perfectly but move wildly if one point changes. Plot training and validation performance as data or capacity changes to distinguish those situations.",
    "More complexity does not always increase variance, and more data helps only when its quality, coverage and target definition are useful. Check leakage and label consistency before prescribing a larger model.", "ML01 ML05")
add("B02", "concept-ml", "13,89", "Zero training error and double descent",
    "Fitting every training example is not automatically memorisation without generalisation. A sufficiently large model may find a smooth or otherwise useful solution among many interpolating solutions. The learning algorithm and data structure matter, not just parameter count.",
    "As capacity rises, test error can first fall, rise near the capacity needed to fit the sample, then fall again beyond that threshold. That empirical shape is called double descent. It is not a rule that every network or dataset must follow.",
    "Judge held-out behaviour under controlled budgets and repeated runs. Zero training loss alone proves neither good generalisation nor failure.", "DL01 ML05")
add("B03", "concept-ml", "4,5,7", "Learning tasks, parameters and hyperparameters",
    "Supervised learning learns from supplied targets. Unsupervised learning looks for structure without those targets. Self-supervision constructs a target from the data, such as the next word. Semi-supervision combines labelled and unlabelled examples. A parameter is learned, whereas a hyperparameter is a training or model choice.",
    "Predict a price: regression. Predict spam/not-spam: classification. Group similar customers: clustering. Order candidate documents: ranking. A logistic-regression weight is a parameter; its regularisation strength and the decision threshold are choices selected without looking at final test labels.",
    "These labels describe different axes. Next-token prediction is self-supervised by target construction but still optimises a supervised-style prediction loss.", "ML01 T1")
add("B04", "concept-ml", "6,8,10,14", "Different ways to describe a learner",
    "Parametric means a fixed-size parameterisation for the chosen model; non-parametric methods can grow their effective representation with data. Generative models represent how data could be produced, often p(x) or p(x,y); discriminative models focus on predicting y from x. These are not synonyms for neural versus non-neural.",
    "A nearest-neighbour learner stores examples and performs much of its work at query time: lazy learning. Logistic regression fits a compact rule before prediction: eager learning. Inductive evaluation asks about new cases; transductive methods can use the particular unlabelled target set when fitting.",
    "State which information is permitted. An LLM can generate text and also serve as a classifier; one system can be used in both generative and discriminative roles.", "ML01 N04")
add("B05", "concept-ml", "19,20", "Missing values and the missingness mechanism",
    "A missing value is not necessarily a zero, and the reason it is missing may itself be informative. MCAR means missingness is unrelated to observed and unobserved values; MAR allows dependence on observed information; MNAR can depend on the missing value itself after conditioning on what you observed.",
    "Income omitted more often by very high earners can bias a simple median imputation. A lab test ordered only for very sick patients has a clinical selection process behind its missingness. Compare an appropriate imputer plus a missingness indicator with methods that handle missing values directly, fitting preprocessing on training data.",
    "You cannot usually establish MAR versus MNAR from the observed table alone. For a target label, inventing a pseudo-label through routine feature imputation is not an innocent repair.", "ML02")
add("B06", "concept-ml", "24,26-28", "Sampling, imbalance and distribution shift",
    "Class imbalance is a difference in class counts; sampling bias is a mismatch between how the sample was collected and the population of interest. Covariate shift changes input frequencies while keeping the conditional target rule fixed; label shift changes class prevalence under a fixed class-conditional input distribution. Concept drift changes the conditional relationship between input and label.",
    "More short messages is an input change. More fraud is a prevalence change. A new policy that counts different transactions as fraud changes the target rule. Stratified splitting preserves specified class proportions; it cannot stop the same patient or document appearing in both splits.",
    "Unlabelled drift or distance alarms can flag unfamiliar inputs but cannot certify that error increased or identify which shift assumption holds. Obtain delayed labels and inspect task definitions.", "ML15 N22")
add("B07", "statistics", "31-34", "Probability, likelihood, Bayes, MLE and MAP",
    "Probability fixes a model and asks how plausible an outcome is. Likelihood fixes observed outcomes and compares possible model parameters. Bayes combines prior plausibility with likelihood to obtain a posterior. MLE chooses the parameters with highest likelihood; MAP chooses the highest posterior density under a specified parameterisation.",
    "After eight heads in ten flips, a fair-coin model assigns that observation one probability; a different coin bias assigns another. MLE estimates the bias as 0.8. A prior favouring moderate biases pulls the MAP estimate toward the middle. Frequentist inference treats the parameter as fixed; Bayesian inference assigns a posterior over it.",
    "A likelihood over parameter values is not automatically a normalised probability distribution. Neither approach removes assumptions, guarantees good priors or proves that the sample represents deployment.", "P3")
add("B08", "statistics", "35,36,40", "Mean, spread, correlation and the central limit theorem",
    "Mean is the arithmetic average; median splits ordered observations in half; mode is a most frequent value. Variance averages squared deviations and standard deviation returns to the original units. Covariance measures joint signed variation; correlation rescales it by standard deviations.",
    "For latencies 1, 1, 1 and 97 ms, the mean is 25 ms but the median is 1 ms. A few slow requests dominate the mean. Under suitable conditions, averages of independent or appropriately weakly dependent observations become approximately normal as sample size grows: the central limit theorem concerns those averages, not the original observations.",
    "Correlation is not causation, zero correlation does not rule out nonlinear dependence, and heavy tails or strong dependence can invalidate a casual normal approximation.", "P3")
add("B09", "statistics", "37-39,41,42", "What confidence intervals and p-values actually say",
    "Imagine evaluating the same two fixed models on a different sample of documents. Their score difference would move because the sample contains different easy and difficult cases. A confidence interval expresses how precisely the sample lets us estimate a stated population quantity, such as the score difference. A 95% confidence procedure is designed to include that fixed quantity in about 95 of 100 repetitions under its assumptions; it is not a 95% probability assigned to the parameter after seeing this one interval. A prediction interval concerns a future observation itself, rather than only uncertainty in an average.",
    "For a tiny accuracy example, model B beats A on two documents, loses on one and ties on one: its average advantage is (1 + 1 − 1 + 0)/4 = 0.25. A bootstrap draws four of these document IDs with replacement, so an ID can repeat. Drawing the two wins and two ties gives an advantage of 0.5. Other draws give other values. Repeat many times to see sampling variation, always carrying both systems' predictions for the same drawn IDs. For macro-F1, recompute F1 from the sampled predictions rather than average per-document F1s. If documents share a patient or passage, draw the appropriate groups together.\n\nA permutation test asks a different question: would a gap this extreme be unusual under a specified no-difference model? It constructs that reference by swaps allowed by the null, such as swapping system assignments within valid paired units. A p-value is the resulting tail probability under that testing model, not the chance that the no-difference claim is true.\n\nHolm handles a family of comparisons. For raw p-values .010, .030 and .040, compare .010 with .05/3: it passes. Compare .030 with .05/2: it fails, so stop; the last comparison is not declared significant. More testing otherwise creates more chances to mistake noise for a discovery.",
    "Statistical significance is not practical value: a tiny reliable gain may not justify doubling cost. An interval crossing zero does not establish equivalence. Test resampling does not cover all training, new-domain or provider-change uncertainty. Bootstrap resamples approximate sampling from the observed cases; they do not create genuinely new data.", "P3 ML16")
add("B10", "statistics", "43,44", "Entropy, cross-entropy, KL and Jensen–Shannon divergence",
    "Entropy measures uncertainty within a distribution. Cross-entropy scores probabilities supplied by q when outcomes come from p. Their difference is KL(p||q): the extra expected log cost of using q instead of p. Direction matters because outcomes are weighted by p.",
    "For a known correct binary label, predicting probability 0.9 costs −ln(0.9) ≈ 0.105; predicting 0.1 costs about 2.303. Jensen–Shannon divergence compares each distribution with their mixture, making it symmetric and bounded under a specified logarithm base.",
    "KL is not a symmetric geometric distance and can be infinite if q assigns zero probability to an event possible under p. In classification, cross-entropy assesses probabilities; F1 assesses thresholded labels.", "P3 PT06")
add("B11", "statistics", "47,49,56,57", "Precision, recall and the different F1 averages",
    "Precision asks how many alerts were right; recall asks how many real positives were found. Sensitivity is positive-class recall; specificity is negative-class recall. F1 combines precision and recall through their harmonic mean. State the positive label before calculating anything.",
    "Eight true alerts, four false alerts and two missed positives give precision 8/12, recall 8/10 and F1 16/22 ≈ 0.727. Macro averages class scores equally; weighted macro weights them by support; micro pools the counts first. Sample averaging instead computes a score per example, useful for multilabel tasks. Averaging corpus F1s differs from pooling their predictions.",
    "Accuracy can improve by fixing many negatives while the positive F1 falls. Specify zero-division handling and whether an F1 concerns one class, all classes or several datasets.", "P8 DS09")
add("B12", "statistics", "50,51", "ROC-AUC versus precision–recall evaluation",
    "ROC plots true-positive rate against false-positive rate across thresholds. A PR curve plots precision against recall. Neither curve chooses a deployment threshold for you; ranking quality and the number of alerts a team can inspect are different decisions.",
    "With 10 positives and 9,990 negatives, a 1% false-positive rate still creates about 100 false alerts. Finding eight positives then yields only about 8/108 precision despite an apparently small false-positive rate. PR makes that denominator more visible. Average precision and trapezoidal PR-AUC are related summaries but not identical calculations.",
    "PR depends strongly on prevalence. Compare methods on the same population and report a useful operating point, not only a single area score.", "P8 ML10")
add("B13", "statistics", "52-54,249", "Probability calibration, loss and abstention",
    "Accuracy asks whether labels are right. Calibration asks whether stated probabilities match observed frequencies. Cross-entropy punishes confident mistakes logarithmically; Brier score uses squared probability errors. Selective prediction lets the system abstain on some cases and evaluates error together with the proportion it answers.",
    "If 100 comparable cases are scored near 0.8, about 80 should be positive for that probability to be calibrated. Two systems can predict the same labels while one says 0.6 and the other 0.999; their accuracy matches but probability losses differ. Tune an F1 threshold or abstention rule on validation data.",
    "A next-token probability is not a calibrated probability that a whole answer is factually correct. Calibration on yesterday's population need not survive a prevalence or task change.", "SK10 P8")
add("B14", "statistics", "55,219", "Retrieval metrics: recall@k, MRR, MAP and NDCG",
    "Ranking metrics ask where useful items appear. Recall@k counts how many relevant items were found in the first k. Reciprocal rank uses the first relevant position; MRR averages it across queries. Average precision rewards the precision at each relevant hit; MAP averages that per-query measure. NDCG rewards graded relevance with lower weight at later ranks, normalised by an ideal ranking.",
    "If the first relevant paper is at rank 4, reciprocal rank is 1/4. Finding one of two relevant papers in the top five gives recall@5 = 1/2. These scores answer different questions: one early hit versus comprehensive evidence retrieval.",
    "Specify the relevance pool, cutoff and treatment of queries with no relevant documents. Top-k classification accuracy is a different metric: whether the correct class appears among k predicted classes.", "P8 NC06")
add("B15", "concept-ml", "61-63", "Linear regression, logistic regression and correlated features",
    "Linear regression predicts a numeric outcome with a weighted sum. Logistic regression passes a weighted sum through a sigmoid to model a binary-class probability; its name describes the regression of log odds, not a continuous-output task. Correlated features can make individual coefficients unstable even when predictions remain useful.",
    "Two nearly identical income columns can trade large positive and negative weights while leaving their combined prediction almost unchanged. Removing one or regularising can stabilise the fit. Classical regression inference also needs assumptions about the errors and design; normal errors are not required simply to compute an ordinary least-squares fit.",
    "Separate assumptions used for prediction from those used for standard errors or causal interpretation. A regression coefficient is not automatically an intervention effect.", "SK03 ML04")
add("B16", "concept-ml", "66-69", "Tree splits, bagging, boosting and stacking",
    "A tree chooses a feature/threshold that improves its split objective, such as reducing Gini impurity or entropy for class labels. Bagging averages varied fits; a random forest also randomises feature choices. Boosting adds learners sequentially to improve the current objective. Stacking trains a separate combiner on appropriately out-of-fold predictions.",
    "For squared-error boosting, a prediction of 7 against target 10 leaves residual 3; a small new tree learns a correction. XGBoost and LightGBM are efficient boosting systems with different implementation/sampling/growth options; CatBoost specifically addresses categorical-feature handling and prediction shift. Compare an actual configuration, not brand names.",
    "Boosting can still overfit through too many, too deep or poorly regularised rounds. Never fit a stacker on in-sample expert predictions and treat that as clean out-of-sample evidence.", "ML08 SK06 SK07")
add("B17", "concept-ml", "70,71", "Margins and the kernel trick",
    "An SVM looks for a decision boundary with a large margin. A hard-margin formulation permits no violations and requires separability; a soft margin trades violations against margin width. A kernel evaluates inner products in a feature representation without explicitly constructing all its coordinates.",
    "A linear boundary in a transformed feature space can become a curved boundary in the original two-dimensional input. A linear SVM can also be a strong sparse-text baseline without any nonlinear kernel.",
    "A kernel is not free computation: forming or using pairwise relationships can be expensive in memory and time as the training set grows. Tune the violation penalty and kernel settings on training/validation splits.", "ML06")
add("B18", "concept-ml", "72,73", "Clustering versus reducing dimensions",
    "Clustering assigns groups; dimensionality reduction constructs a smaller representation. K-means favours compact squared-distance groups; DBSCAN connects sufficiently dense regions and marks noise; hierarchical clustering builds nested groupings. PCA preserves high-variance linear directions, whereas t-SNE and UMAP emphasise particular neighbourhood relationships for a low-dimensional embedding.",
    "A two-dimensional t-SNE plot can show apparent islands even when the original data do not contain clean categories. Conversely, a customer-clustering algorithm can output labels without producing a useful two-dimensional plot.",
    "Do not read global distance, cluster size or predictive importance directly from a visualisation. Scaling, distance, random seed and hyperparameters change what a plot or clustering emphasises.", "ML11 ML12")
add("B19", "concept-dl", "76-79", "What a neuron and an activation function do",
    "A basic neuron computes a weighted sum plus bias, then applies a function: y = f(w·x + b). The weights decide which input directions matter. A nonlinearity lets a network express more than a single affine transformation.",
    "Without nonlinearities, W2(W1x + b1) + b2 equals one new matrix times x plus one bias. ReLU keeps positive values and zeros negative ones; Leaky ReLU keeps a small negative slope; GELU smoothly gates values. Sigmoid is useful for binary probabilities; tanh is bounded around zero.",
    "Activation choice affects gradients and ranges, not just expressiveness. Sigmoid can saturate, and an inactive ReLU can have zero gradient. A deeper stack of linear layers is not a nonlinear classifier.", "DL01 PT03")
add("B20", "concept-dl", "80-82", "Backpropagation without the mystery",
    "The forward pass computes predictions and loss. Backpropagation applies the chain rule to calculate how each parameter locally affects that loss. The optimiser then uses those gradients to update parameters; computing a gradient does not itself change a weight.",
    "For y = wx and L = (y − t)², dL/dw = 2(y − t)x. With x = 2, w = 1 and t = 3, the gradient is −4; a small gradient-descent step increases w. Across many layers, repeated derivative products can shrink or grow, producing vanishing or exploding gradients.",
    "Check loss scaling, masks, initialisation and graph connectivity before blaming the optimiser. Gradient clipping limits update signals; it does not establish that the data, objective or model is correct.", "DL02 PT02")
add("B21", "concept-dl", "83,149", "BatchNorm, LayerNorm, RMSNorm and GroupNorm",
    "Normalisation rescales activations over specified axes. BatchNorm uses batch/spatial statistics and typically running statistics at inference. LayerNorm normalises a token/example's feature dimensions. RMSNorm rescales using root mean square without subtracting the mean. GroupNorm normalises groups of channels within an example.",
    "For a language-model tensor [batch, tokens, hidden], token-wise LayerNorm works across hidden features rather than making one token depend on other examples in the batch. Pre-norm transformer blocks normalise before attention/MLP residual branches, which can make optimisation easier than some post-norm configurations.",
    "Always name the axes. These operations do not normalise raw dataset columns, and train/eval behaviour differs between BatchNorm and ordinary LayerNorm.", "DL03 T3")
add("B22", "concept-dl", "84,87,88", "Initialisation, residual paths and capacity",
    "Initialisation controls the scale of early activations and gradients; Xavier and He schemes choose variances suited to particular layer/activation assumptions. A residual block computes x + f(x), preserving a direct information and gradient route. Depth counts layers; width counts units/features per layer; parameter count is one capacity-related measure, not a complete description.",
    "A residual branch can initially learn a small correction rather than reconstruct the whole representation from scratch. Doubling the width of a dense hidden-to-hidden matrix roughly quadruples that matrix's weights, while adding another same-width block adds another block's weights.",
    "Skip connections do not guarantee learning or remove all vanishing-gradient problems. Compare actual dimensions and operations rather than assuming equal parameter counts imply equal function or cost.", "DL03 DL06")
add("B23", "concept-dl", "85,86", "Why dropout changes between training and evaluation",
    "Dropout randomly removes selected activations during training so the model cannot depend on every particular co-adaptation. In ordinary inverted dropout, surviving values are scaled during training; dropout is disabled at evaluation.",
    "With drop probability 0.5, a surviving activation of 2 becomes 4 during training, preserving its expected value. The evaluation network uses the full unmasked activations. This resembles training many related submodels, but it is not generally exactly equal to averaging all independently trained networks.",
    "model.eval() changes dropout and some normalisation behaviour; no_grad() prevents gradient recording. They solve different problems. Repeated stochastic dropout at test time is a deliberate uncertainty technique, not the default inference setting.", "DL04 PT03")
add("B24", "concept-dl", "91-93", "SGD, momentum, Adam and AdamW",
    "Gradient descent uses all training examples per update; stochastic and minibatch variants estimate an update from fewer examples. Momentum smooths directions over time. RMSProp tracks squared gradients; Adam combines first- and second-moment estimates. AdamW applies weight decay separately from the adaptive gradient transformation.",
    "A parameter with a repeatedly large raw gradient gets a different effective step under Adam's per-coordinate rescaling. Adding an L2 term to the loss then passing it through that rescaling is not generally the same update as separately shrinking the weight in AdamW.",
    "No optimiser wins everywhere. State learning rate, decay and schedule together; an apparent optimiser improvement may just reflect a better-tuned configuration.", "DL03 PT05")
add("B25", "concept-dl", "94-96,98-100", "Learning-rate schedules, batch size and stopping",
    "The learning rate controls update size. Too large can overshoot; too small can waste the budget. Warmup increases it gradually, then schedules such as cosine decay reduce it. Early stopping selects a checkpoint using validation behaviour, not the final test.",
    "A lower training loss with worsening validation loss suggests overfitting or a mismatch worth investigating; noisy validation can also mislead a very short patience setting. Increasing batch size changes gradient noise and often the number of updates per data pass, so an unchanged learning-rate recipe is not automatically a fair comparison.",
    "Large batches do not inherently generalise badly. Check total examples/tokens, update count, schedule, validation cadence and clipping. Patience is a chosen rule, not evidence that the chosen checkpoint generalises to every domain.", "T5 DL04")
add("B26", "concept-dl", "101-103", "Precision and activation checkpointing",
    "FP32, FP16, BF16 and FP8 allocate bits differently to range and precision. BF16 has a larger exponent range than FP16, often reducing overflow risk; support and numerical behaviour still depend on hardware and kernels. Mixed precision uses different precisions for different parts of training.",
    "A finite FP32 value may overflow when cast to FP16. Loss scaling can help avoid small-gradient underflow in FP16, but cannot repair every overflow or invalid operation. Activation checkpointing saves fewer forward intermediates and recomputes them during backward, trading compute for memory.",
    "Do not estimate training memory from stored weights alone. Include activations, gradients, optimiser state, temporary buffers and distributed communication. FP8 is not a universal safe switch for all tensors.", "T6 PT05")
add("B27", "concept-nlp", "106,110", "Stemming, lemmatisation and n-gram models",
    "Stemming applies rough word-form reductions; lemmatisation uses lexical/morphological analysis to recover a dictionary form. An n-gram language model estimates the next unit from a short fixed history, whereas a neural language model learns a distributed representation of context.",
    "A stemmer may map 'studies' to a nonword stem, while a suitable lemmatiser maps it to 'study'. A bigram estimates the next word after one preceding word using counts and smoothing; it cannot directly distinguish two longer contexts that end in that same word.",
    "Do not blindly normalise spelling or morphology before a pretrained tokenizer: its expected input can differ, and distinctions such as named entities may matter. Simple count models remain useful baselines.", "N02 N05")
add("B28", "concept-nlp", "107,111,155", "Tokens, vocabulary and context length",
    "A tokenizer maps text to units and IDs. Word tokenizers use whole-word units; subword systems split frequent pieces; character and byte approaches use smaller inventories. Vocabulary size is the number of token types; context length is how many token positions the model can process at once.",
    "One rare name can occupy several subword tokens. Byte-based coverage avoids an unknown-character problem, but does not make every word one token or every language equally efficient. A longer tokenised document raises attention and serving costs even when its character count looks modest.",
    "Offsets, special tokens, normalisation and chat formatting affect alignment and model behaviour. A model's tokenizer and embedding weights must agree on ID meanings.", "P6 NC01")
add("B29", "concept-nlp", "108,109,213", "Why lexical retrieval still matters",
    "Bag-of-words counts terms. TF–IDF downweights globally common terms; BM25 is a lexical ranking function with term-frequency saturation and document-length normalisation. These methods preserve exact identifiers and rare vocabulary instead of asking an embedding model to infer what matches.",
    "A query for error E-1042 may be best answered by the exact manual paragraph containing E-1042, even if a dense retriever prefers a semantically similar paragraph about another error. Hybrid retrieval combines complementary candidates; reranking then compares the query with a smaller candidate set.",
    "Lexical success is not a reason to discard dense retrieval everywhere. Test query types, domain vocabulary, tokenisation, relevance labels, recall and operating cost.", "N03 NC02")
add("B30", "concept-nlp", "112-114", "Syntax, meaning and the labelled unit",
    "POS tagging assigns grammatical categories; NER marks entity spans; dependency parsing predicts grammatical relations. Syntax concerns arrangement, semantics meaning, pragmatics use in context and discourse connections between utterances. Sentence classification returns one label per sentence; sequence labelling returns coordinated labels over units such as tokens.",
    "'Great, another delay' can be criticism despite positive-looking vocabulary. Identifying a person in it is a different task from identifying its sentiment or argumentative role. A sentence-level score does not evaluate exact entity boundaries.",
    "Start an NLP interview answer by identifying the unit, label definition and available context. Do not claim a discourse interpretation has been verified merely because a classifier's score improved.", "N01 N13 N17")
add("B31", "concept-nlp", "115,116", "HMMs and CRFs as sequence models",
    "An HMM models hidden-label transitions together with how observations are emitted. A linear-chain CRF models a conditional score for a whole label sequence given the input. Both make neighbouring labels part of the decision instead of classifying each token independently.",
    "In BIO entity tagging, an I-PER immediately after O can be illegal under the chosen scheme. A CRF can score or constrain transitions while using a neural encoder's token scores. Viterbi efficiently finds the highest-scoring sequence by reusing best partial-path scores.",
    "A CRF does not automatically impose every desired rule unless the scoring/constraints implement it. It cannot rescue missing entities or an unsuitable annotation scheme by itself.", "N12 NC05")
add("B32", "concept-nlp", "117", "Exposure bias in generation",
    "During ordinary teacher-forced training, the model predicts the next token after the correct preceding tokens. At generation time it sees its own earlier choices. An early mistake can therefore create a context unlike the training contexts and influence later predictions.",
    "A summariser trained after the correct entity name may generate a wrong name first, then continue consistently about the wrong person. The next-token training loss can look reasonable while the full generated summary is unreliable.",
    "Changing the training/generation mismatch has trade-offs; more beam search is not a universal correction. Evaluate complete outputs and error propagation, not only teacher-forced loss.", "T1 N11")
add("B33", "concept-nlp", "118,119", "What text-generation metrics miss",
    "BLEU emphasises modified n-gram precision with a brevity penalty; ROUGE variants measure reference overlap, often recall-oriented. METEOR includes alignment variants such as stemming/synonyms; chrF compares character n-grams. BERTScore compares contextual representations. These are proxies for particular similarities, not factuality certificates.",
    "'Treatment reduced mortality' and 'Treatment did not reduce mortality' share most words but reverse the result. Two valid paraphrases can instead have little word overlap. A summary needs checks for claims, numbers and attribution in addition to overlap.",
    "Name the variant, implementation and aggregation, and validate its relevance to human judgement for the task. A higher similarity score does not prove source faithfulness.", "N20 NC07")
add("B34", "concept-nlp", "121-124", "Static embeddings and contextual representations",
    "One-hot vectors identify tokens without encoding graded similarity. Learned embeddings place them in a numerical space. Word2Vec learns through local prediction (CBOW predicts a word from context; Skip-gram predicts context from a word); GloVe uses co-occurrence statistics; FastText shares subword information. Contextual encoders compute a token representation using the actual input.",
    "A static vector for 'bank' must serve both a river bank and a financial bank. A contextual representation can differ between those sentences. A sentence vector also needs an appropriate pooling/training objective; a CLS vector is not automatically a strong semantic-search representation.",
    "Cosine proximity reflects the learned objective and corpus. It need not mean entailment, factual agreement or the same class label.", "N06 N14 N15 NC03")
add("B35", "concept-nlp", "12,125,126,131,132", "Similarity measures and embedding geometry",
    "A dot product combines alignment and vector magnitude. Cosine divides by the vector lengths; Euclidean distance measures coordinate separation. On unit-normalised vectors, squared Euclidean distance equals 2 − 2 cosine. Semantic similarity concerns likeness of meaning; relatedness also includes associated but different concepts.",
    "'Doctor' and 'hospital' are related, not interchangeable. If most embeddings point in a common direction (anisotropy), cosine scores may occupy a narrow range. High dimensions can also make distance contrasts less informative under particular distributions.",
    "The curse of dimensionality means that covering a space can require rapidly more data as relevant dimensions grow; irrelevant coordinates can also make nearest-neighbour distances less useful. Inspect score distributions, normalisation and task-labelled neighbours. High dimensionality alone does not make cosine search useless.", "P7 N15 ML06")
add("B36", "concept-nlp", "129,130", "Contrastive learning and negative examples",
    "Contrastive training makes a query more compatible with a positive candidate than with selected negatives. InfoNCE uses a normalised classification-style objective over that candidate set. The temperature changes how sharply score differences affect the loss.",
    "A hard negative looks relevant but is wrong, such as a manual for a similar device model. A random negative may be obviously unrelated. If a supposedly negative passage is another correct answer, pushing it away teaches the wrong distinction.",
    "Negative mining changes the training distribution. Validate labels, avoid false negatives and keep held-out queries out of mining/tuning decisions.", "DL09 N15")
add("B37", "concept-nlp", "133", "Approximate nearest-neighbour indexes",
    "Exact vector search compares against every candidate. Approximate indexes reduce the search work at a possible recall cost. HNSW navigates a graph of neighbours; IVF first narrows the search to selected coarse clusters; product quantisation compresses vector subparts into codebook indices.",
    "For a million document embeddings, searching only a few IVF clusters can be cheaper than scoring all vectors, but can miss a relevant document assigned elsewhere. Increasing the search effort often raises recall and latency together.",
    "Compression and approximate search are different choices and can be combined. Measure recall against exact search on a representative subset, plus update, filter, memory and latency behaviour.", "R4 NC06")
add("B38", "architecture", "136-141,148", "Attention as a concrete calculation",
    "Queries ask what to retrieve, keys describe possible matches and values carry the information to mix. For one head, Attention(Q,K,V) = softmax(QKᵀ/√d_k + mask)V. The scale limits score growth as key dimension increases. Self-attention reads the same sequence; cross-attention reads another sequence; a causal mask blocks future positions.",
    "If a query gives two allowed keys equal scores, their softmax weights are 0.5 and 0.5, so the output is the average of their value vectors. A padding mask excludes fake padding positions, whereas a causal mask excludes future positions. Multi-head attention uses several learned projections before recombining outputs.",
    "Attention weights are not automatically faithful explanations. State shapes and mask conventions; an additive blocked mask usually uses −∞, not zero. Causal masking does not imply there is no parallel training over positions.", "T3 PT09")
add("B39", "architecture", "143,151-154", "Encoder, decoder and encoder–decoder models",
    "An encoder builds contextual representations of an available input; a causal decoder generates by predicting the next token from its prefix. An encoder–decoder reads an input with an encoder and conditions a decoder through cross-attention. Training objective and attention pattern, not a marketing label, define the distinction.",
    "BERT commonly uses masked-token pretraining and an encoder; GPT-style models commonly use causal next-token training; T5 uses an encoder–decoder denoising/text-to-text design. A conventional classifier adds a task-specific prediction head; an LLM can generate that label as text instead.",
    "Next-token training can encourage rich representations and algorithms because predicting text is difficult; it does not guarantee factuality or every downstream capability. Compare the actual objective and permitted information.", "T1 T4")
add("B40", "architecture", "144-146", "Where a transformer gets position information",
    "Attention alone treats a set of vectors without an inherent left-to-right order. Position mechanisms tell the model which positions or relative offsets are involved. Learned/sinusoidal embeddings add positional signals; RoPE rotates query/key coordinates to encode relative-position relationships; ALiBi adds distance-related attention biases.",
    "The words 'dog bites man' and 'man bites dog' use the same vocabulary but need different representations. A causal mask already supplies some asymmetric positional structure, so 'no explicit position embedding' does not always mean 'no position information whatsoever'.",
    "A position scheme supporting a long window does not establish reliable use of every location. Test extrapolation, truncation and task quality rather than assuming a context-length setting proves comprehension.", "T3 T4")
add("B41", "architecture", "142,147,150", "Attention cost and shared key/value heads",
    "Standard dense attention forms query–key interactions for token pairs, giving an O(T²) interaction term as sequence length T grows. Multi-query and grouped-query attention share key/value heads across query heads, reducing cached key/value storage while retaining multiple query projections.",
    "Doubling T roughly quadruples the pairwise interaction count, but not every model operation or measured latency. MLP work is usually linear in T for a fixed model width. FlashAttention changes memory traffic and avoids materialising the full attention-score matrix; it does not make dense attention's arithmetic generally linear.",
    "Training, prefill and cached single-token decode have different cost profiles. Include batch size, sequence lengths, cache policy and hardware when predicting memory or latency.", "T6 R6")
add("B42", "architecture", "156,157", "Perplexity is not a reasoning score",
    "Perplexity is exp(mean next-token negative log-likelihood) when natural logarithms are used. It measures how well a model predicts the evaluated token sequence under a specified tokenizer and context protocol.",
    "A mean loss of ln(4) gives perplexity 4. That does not mean four equally likely answers to a user's question. A model can predict routine language very well while failing a logical puzzle, because most test tokens may not require that puzzle-solving skill.",
    "Do not compare token perplexities across incompatible tokenizers as if the unit were unchanged. Match data, boundaries and masking; use task tests for reasoning and factuality.", "T1 P8")
add("B43", "architecture", "158,159", "Scaling and compute-optimal training",
    "Scaling laws are empirical relationships between resource choices and measured loss in a stated regime. Compute-optimal training asks how to divide a finite budget between model size and training tokens, not how to make the largest possible parameter count.",
    "A very large model trained on too few tokens may lose to a smaller model trained longer under the same training compute. But a model served billions of times may justify extra training to reduce lifetime inference cost.",
    "Do not extrapolate one fitted law indefinitely or treat benchmark quality as identical to pretraining loss. Data quality, architecture, tokenisation and deployment economics change the decision.", "T6")
add("B44", "architecture", "160,161,189", "Continued pretraining, SFT and forgetting",
    "Continued pretraining keeps learning the broad language objective on new text. Supervised fine-tuning learns from task demonstrations or labelled targets. Catastrophic forgetting means adaptation damages capabilities learned earlier; low new-domain training loss does not reveal that damage.",
    "Technical manuals may help a base model learn terminology through continued pretraining. Example question/answer pairs teach a response behaviour through SFT. A small encoder classifier may be the simpler choice when the only output is a label.",
    "Keep a held-out task set and a relevant retention set. Diagnose prompt formatting, overfitting and objective mismatch before concluding that more finetuning is needed. Retrieval supplies evidence without changing weights.", "R1 T1")
add("B45", "reasoning", "162", "Distillation: learning from a teacher",
    "Distillation trains a student to match information supplied by a teacher: distributions, labels, responses or solution traces. The student still needs a training objective; merely calling a teacher API is not distilling a model. Self-distillation uses a model or related checkpoint as its own teacher in a specified procedure.",
    "A larger model can label training texts, then a smaller encoder learns those labels. If the teacher consistently mistakes premises for claims, the student may inherit that bias. High agreement demonstrates imitation, not independent semantic correctness.",
    "State the supervision, data rights, filtering and independent evaluation. A smaller distilled model's success on selected tasks does not establish that it copied all of the teacher's abilities.", "R7 R1")
add("B46", "architecture", "163,164,296,297", "Ensembles, routers and sparse experts",
    "A static ensemble combines predictions using a fixed rule. A stacker learns from expert predictions. A router chooses or weights experts for an input, and a jointly trained mixture can learn experts and gate together. Sparse LLM MoE layers activate only selected experts per token, unlike simply running several complete chatbots.",
    "A model can have many total expert parameters but use only two experts per token, reducing active arithmetic relative to activating them all. Loading, communication, routing and shared layers still cost resources. In a classifier ensemble, correlated errors or miscalibrated probabilities can make an average worse than the strongest expert.",
    "Name what is selected—prompt, classifier or internal expert—and at what granularity. A dataset-level route using labelled development data is not an oracle for an unseen individual sentence.", "ML08 T4")
add("B47", "reasoning", "166-168,171-174", "Preferences, rewards and what alignment changes",
    "Pretraining learns a general predictive objective; instruction tuning imitates demonstrations; preference optimisation changes which responses are favoured. A reward model can learn to rank preferred over rejected responses. Outcome supervision scores an answer; process supervision assesses intermediate steps.",
    "If reviewers prefer confident, fluent answers, a system can become more pleasant without becoming more factual. Reward hacking exploits gaps in the scoring rule. A KL penalty can discourage drifting too far from a reference policy, but does not by itself define truth or prevent every exploit.",
    "Evaluate correctness, helpfulness, refusal and distribution shift separately. Better human preference under a chosen rubric is a specific result, not a universal quality improvement.", "R2", "[Preference optimisation](https://arxiv.org/abs/2305.18290)")
add("B48", "reasoning", "170", "DPO, IPO, ORPO and GRPO are not interchangeable",
    "DPO uses preferred/rejected pairs to optimise a policy relative to a reference formulation without the usual separate online PPO rollout loop. IPO modifies the preference objective and its treatment of confidence. ORPO combines supervised learning with an odds-ratio preference term. GRPO uses relative rewards within groups of sampled responses, typically avoiding a separately trained value critic.",
    "With several generated solutions to one verifiable maths problem, GRPO can compare each reward with the group's reward statistics. A static table of chosen/rejected answers instead supports a pair-based method. PPO is another policy-gradient approach with a clipped surrogate and commonly a learned value estimate.",
    "Implementations and objectives differ. Explain the data and update signal before listing acronyms; normalising rewards in a toy calculation is not evidence that an RL trainer has been built.", "R2 LC05", "[IPO](https://arxiv.org/abs/2310.12036), [ORPO](https://arxiv.org/abs/2403.07691), [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)")
add("B49", "reasoning", "175,176,178,179", "Reasoning steps, verification and test-time compute",
    "A direct prompt asks for an answer; chain-of-thought prompting also elicits intermediate text. Self-consistency aggregates answers from several samples. Best-of-N selects a candidate using a score or verifier, while rejection sampling retains candidates passing a criterion. These spend compute in different ways.",
    "Ten long but wrong derivations are not better than one correct short answer. A calculator can validate arithmetic, while a learned verifier can itself make systematic mistakes. Compare the same question set under explicit sampling, tool and token budgets.",
    "Reasoning-token count is expenditure, not proof of reasoning quality. More steps can amplify errors; a helpful external scaffold does not mean the underlying model has no useful reasoning ability.", "R3 R5")
add("B50", "reasoning", "177", "Can LLMs reason, or is it all the harness?",
    "Reasoning is an ability to carry out useful multi-step inference, not a synonym for human consciousness. Learned next-token computation can implement parts of that ability. Models can solve some novel multi-step tasks without tools, and specialised post-training can improve such performance. That does not establish that their process is human-like, generally reliable or understood.",
    "Separate three experiments: the base model answering alone; the same model with a scratchpad or extra sampling; and the complete system with search, memory, tools or verifiers. A calculator may explain an arithmetic gain in the last condition, but cannot explain every success of the model alone. Perturb the task and evaluate under matched budgets to test generalisation.",
    "A strong interview answer is: 'I would separate model capability from scaffold effects and rationale faithfulness. I would not infer human-like thought from fluent explanations, but I also would not claim models cannot reason at all.' A printed rationale may omit or invent influences; test it rather than treating it as a transcript of internal computation.", "R3 T8", "[DeepSeek-R1](https://arxiv.org/abs/2501.12948), [CoT interventions](https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning), [Circuit tracing](https://www.anthropic.com/research/tracing-thoughts-language-model)")
add("B51", "reasoning", "181-188,193", "LoRA and QLoRA with a parameter calculation",
    "Full fine-tuning updates the chosen base weights. LoRA freezes a matrix W and learns a low-rank update BA, often scaled by alpha/r. For a d_out by d_in matrix, the adapter has r(d_in + d_out) weights rather than d_in·d_out. Prompt tuning learns input-side embeddings instead of those weight updates.",
    "For a 1,024 by 1,024 matrix and rank 8, the adapter uses 8(1024 + 1024) = 16,384 weights versus 1,048,576 in the matrix. QLoRA stores the frozen base in a low-bit representation and trains adapters in suitable compute precision. NF4 is a nonuniform 4-bit codebook; double quantisation also compresses quantisation constants.",
    "Activations still use memory. Rank, target modules, scaling, data and objective affect quality; low rank may miss a useful full-rank change. Attention-only and MLP adaptation are different interventions. QLoRA does not train every base weight in four-bit arithmetic.", "R1 LC04")
add("B52", "architecture", "190-192", "Loss masks, packing and chat templates",
    "An attention mask controls what information a token can read; a loss mask controls which targets contribute to optimisation. Packing combines sequences to reduce wasted padding; dynamic padding pads only as much as the current batch requires. Their boundary rules must match the intended objective.",
    "For assistant-only SFT, user text can be visible as context while its tokens are excluded from the target loss. A bad EOS or chat template can teach the model to continue user turns or fail to stop. Packing unrelated examples without the intended document boundaries can leak context between them.",
    "Inspect actual IDs, masks and decoded target tokens on a tiny batch. 'The loss decreased' does not certify the formatting or target shift is correct.", "PT04 PT05 LC02")
add("B53", "serving", "196-199", "Greedy, beam and sampling decoders",
    "Greedy decoding takes the highest-scoring next token. Beam search retains several partial sequences under a scoring rule. Top-k sampling samples from k candidates; top-p samples from a probability-mass-selected set. Temperature rescales logits before sampling; penalties modify scores to discourage repeats or change length preference.",
    "A model assigning probabilities 0.6 and 0.4 can always choose the first token greedily or sometimes choose the second under sampling. A low-temperature answer can still confidently be false. Beam search optimises its sequence scoring rule, not factual correctness.",
    "Temperature zero does not guarantee identical outputs across hardware, kernels, backend versions or ties. Record the complete decoding and serving setup, not just one parameter.", "R3")
add("B54", "serving", "200-202,209,210", "Prefill, decode and KV-cache memory",
    "Prefill processes the prompt and builds key/value state. Autoregressive decode then produces new tokens using that cached state instead of recomputing every previous key/value. TTFT measures waiting for the first token; inter-token latency measures later gaps; throughput measures work completed per time.",
    "A long prompt can delay the first token even when later tokens are fast. Several concurrent long conversations can exhaust cache memory although the model weights alone fit. More admitted requests may raise throughput while worsening queue time for each user.",
    "Measure prompt/output lengths, queue time, cache use and latency percentiles under realistic load. The right optimisation depends on whether prefill, decode, memory or queuing is the bottleneck.", "R6 LC09")
add("B55", "serving", "203,204", "Quantisation is a system trade-off",
    "Quantisation represents values with fewer bits or a restricted codebook. Post-training quantisation converts an already trained model; quantisation-aware training simulates relevant effects during training. INT8, four-bit formats and FP8 have different arithmetic and hardware requirements.",
    "Changing stored weights from 16 bits to 4 bits reduces that idealised weight payload by 75%, not necessarily total GPU use. KV cache, activations, metadata, scales, unquantised layers and temporary buffers remain. A smaller payload can even run slower if the hardware lacks suitable kernels.",
    "Compare measured memory, latency and held-out quality. A nominal bit width alone does not establish a deployment improvement.", "R7")
add("B56", "serving", "205-207", "FlashAttention, PagedAttention and speculative decoding",
    "FlashAttention targets attention memory traffic and intermediate materialisation. PagedAttention organises key/value-cache storage to reduce fragmentation and improve serving utilisation. Continuous batching admits/removes requests as they progress instead of making every request wait for one fixed batch to finish.",
    "Speculative decoding uses a cheaper draft model to propose tokens that the target model verifies. Appropriate acceptance/correction rules can preserve the target distribution; ordinary unverified draft output cannot. It helps when drafting plus verification is cheaper than serial target decoding with a useful acceptance rate.",
    "These optimise different bottlenecks. Do not claim all reduce attention arithmetic complexity or always improve latency. Measure the actual implementation and workload.", "R6 T7")
add("B57", "serving", "208,277,278", "Parallelism and why more GPUs can be slower",
    "Data parallelism replicates a model across example batches; tensor parallelism splits operations within layers; pipeline parallelism splits stages/layers; expert parallelism distributes MoE experts. DDP synchronises gradients across replicas. FSDP/ZeRO-style methods shard selected model/gradient/optimiser states to reduce replication memory.",
    "A small training step can spend more time exchanging gradients than computing them, so adding GPUs need not halve elapsed time. Pipeline stages can sit idle during bubbles; uneven sequence lengths or expert routing can create load imbalance.",
    "Name which state is sharded and which communication is required. Compare tokens per second and time-to-quality at matched effective batch and schedule, not GPU count alone.", "T7")
add("B58", "systems", "217,218,220-224", "When retrieval succeeds but answering still fails",
    "Retrieval recall asks whether useful evidence was found. Context construction decides which passages, order and metadata reach the model. Generation must then interpret and support the answer. These stages can fail independently. More retrieved material can also add distractors, conflicts and token cost.",
    "A correct policy paragraph is retrieved, but its exception is truncated; that is a context failure. If both are present and the answer ignores the exception, investigate generation. Query expansion can improve recall yet retrieve the wrong sense of a word. Resolve conflicting sources using authority, scope and version—not whichever sentence the model prefers.",
    "Groundedness is support by supplied evidence; factual correctness concerns the world. A response can faithfully repeat a stale source and still be wrong today. Long-context prompting and continued pretraining do not replace access control or provenance.", "R4 N19")
add("B59", "systems", "226-231,235-238", "Workflows, tool calls and agents",
    "A workflow follows an explicit control path; an agent gives a model some discretion over next actions. Function calling requests a structured tool invocation; schema validation checks shape, while authorisation checks whether this actor may perform this action. ReAct interleaves model-generated reasoning/action steps with observations.",
    "A syntactically valid SQL update can target the wrong customer. Use allowlisted operations, scoped identities, validation, dry runs and approval for sensitive writes. Multiple agents can divide work, but also add cost, coordination failures and new injection paths.",
    "More reflection or more agents is not automatic reliability. Measure task success, verified tool outcomes, constraint violations and cost. Prefer a fixed workflow when its boundaries are known.", "R5 LC07")
add("B60", "systems", "232", "Conversation state, external memory and model knowledge",
    "Short-term state is information kept for the current conversation or task. External memory is stored information retrieved later under a policy. Parametric knowledge is information represented in learned weights. Remembering a message in a database does not update those weights.",
    "A user's preferred report format can be saved with provenance and expiry. A current legal policy should come from an authoritative versioned source, not an old conversational summary. Compressing a conversation can omit a crucial constraint.",
    "Define ownership, retention, correction, deletion and access controls. Memory retrieval is evidence to inspect, not an automatically authoritative instruction.", "R5 LC08")
add("B61", "research", "241,242,250", "Factuality, faithfulness and helpfulness",
    "Factuality asks whether a claim is correct; source faithfulness asks whether it is supported by the specified source; rationale faithfulness asks whether an explanation reflects relevant influences on a decision. Relevance and helpfulness concern the user's task. A fluent answer can fail any of these independently.",
    "An assistant may accurately quote a superseded handbook, giving a source-faithful but currently wrong answer. A next-token model may strongly predict a familiar false claim because probability of text is not probability of real-world truth.",
    "Use claim-level evidence checks, appropriate references and abstention when evidence is insufficient. Retrieval can reduce unsupported output but cannot make every generated statement true.", "N20 R8")
add("B62", "research", "243-247,257", "How to use an LLM judge without trusting it blindly",
    "When there is no single reference string, define a rubric and collect representative judgements. Pairwise evaluation asks which answer is better; absolute scoring assigns a score; a rubric identifies dimensions such as support and completeness. A model judge is a measurement instrument that needs validation.",
    "Swap answer order to test position bias, control for length to inspect verbosity bias, and compare against independent human-labelled cases. The same judge may favour its own style or model family. Separate quality dimensions rather than hide factual errors in an overall eloquence score.",
    "Report disagreement, blinding and judge versions. High judge agreement on easy examples does not establish reliability on adversarial or domain-critical cases.", "R8 N24")
add("B63", "research", "248,254-256", "Contamination, memorisation and robustness",
    "Contamination means evaluation information entered learning or selection in a way that compromises the intended test. Memorisation is retention of particular examples; verbatim extraction is eliciting stored text; generalisation is useful performance on appropriately new cases. These need different tests.",
    "A model that recalls benchmark solutions can score well without solving a new version. Use controlled paraphrases, changed quantities, source/time separation and held-out task families to probe brittleness. Keep the meaning and label intact when perturbing text.",
    "No exact duplicate found is not proof that a proprietary model never saw related material. State the checks and uncertainty. A leaderboard improvement can miss the deployment population, error costs or human-useful outcome.", "T8 N22")
add("B64", "research", "251-253,258,259", "Safety, privacy and excessive refusal",
    "Prompt injection places adversarial instructions in untrusted inputs; a jailbreak seeks to bypass intended behavioural constraints. Safety evaluation also includes privacy, harmful assistance, subgroup performance and over-refusal. A system message is not a substitute for enforced tool permissions.",
    "Evaluate both harmful requests that should be refused and harmless requests that should be answered. An assistant that refuses everything can have few harmful completions but zero usefulness. In a document system, permission checks must occur before sensitive content is retrieved or actions executed.",
    "Define harm categories and the affected population, inspect severity as well as rates, and protect sensitive evaluation data. A higher preference score can coexist with lower factuality or fairness.", "R8")
add("B65", "serving", "261-263,268,269,274", "Model deployment, versioning and rollback",
    "An ML deployment includes more than executable code: weights, preprocessing/tokenizer, label mapping, thresholds, index/data versions and resource behaviour. Batch, online and streaming inference differ in how inputs arrive and when outputs are needed. Version the components that actually determine a prediction.",
    "A canary exposes a limited live population; a shadow runs without affecting the user's outcome; blue–green keeps two environments for switching. If an index changes and answers fail, reverting only model weights may not restore the previous system.",
    "A reproducible training experiment is not the same as reproducing a deployed service with its permissions, dependencies and upstream state. Define a tested rollback unit and verify behaviour after switching.", "R8 LC10")
add("B66", "serving", "270-273", "SLOs, latency tails and delayed feedback",
    "An SLI is a measured service indicator; an SLO is its target; an SLA is a service commitment with specified terms. An error budget records the allowed shortfall against an objective. A percentile describes a distribution, not a typical request's guarantee.",
    "An average latency of 200 ms can hide p99 latency of 10 seconds during traffic bursts. Inspect queue time and request-length groups. When quality labels arrive weeks later, monitor safe input/output indicators and audit sampled cases while keeping eventual labelled outcome checks.",
    "A drift alert is not a measured quality loss. User feedback is selective and can form a loop: users see only recommended items, whose clicks then reinforce the same recommendations.", "R6 R8")
add("B67", "research", "281-283,285,286", "Transfer, domain adaptation and changed label definitions",
    "Transfer learning reuses knowledge across tasks or domains. Domain adaptation uses specified target-domain information; domain generalisation aims to work on unseen domains without that adaptation. A change in annotation definition can alter the correct answer even when the input text stays fixed.",
    "Training on news and testing on tweets mixes style, topic, context and possibly label changes. Reannotating the same text under several guidelines helps isolate definition dependence. A multi-domain model may suffer negative transfer when its domains reward conflicting cues or definitions.",
    "Do not harmonise labels by name alone. Read the annotation schemes, preserve provenance, document projections and consider guideline-conditioned or multi-task outputs when one universal label is inappropriate.", "N22 ML15")
add("B68", "research", "284,287-290", "Shortcuts, disagreement and meaning-changing augmentation",
    "A shortcut is a predictive cue that does not implement the intended task and may fail when conditions change. Label noise, genuine ambiguity and principled annotator disagreement are different explanations for inconsistent labels; majority voting is not automatically the right truth model.",
    "Rewriting a neutral sentence into 'therefore we should act' can create a claim, changing the label instead of merely clarifying the wording. Hard negatives can similarly teach artefacts if all negatives follow one template. Inspect examples and definitions before treating a disagreement as an annotator mistake.",
    "Use adjudication or disagreement-aware evaluation when appropriate, and test invariance only for transformations meant to preserve the label. A useful prediction channel is not necessarily a faithful semantic analysis.", "N01 N21 N22")
add("B69", "research", "291-293,295,298", "Ablations, seeds, test selection and causal wording",
    "An ablation changes a defined component; a sensitivity analysis varies a setting. To attribute an improvement to one factor, hold relevant alternatives fixed and rule out competing explanations. Repeated seeds probe training randomness; repeated datasets probe another axis. They are not exchangeable independent observations by default.",
    "If a new graph model adds edges, layers and a different readout, its gain supports the complete system, not edges alone. If a prompt was chosen after reading test scores, that set has influenced selection even if no test label appeared in the prompt.",
    "Report per-domain results, aggregation, budgets and uncertainty with their actual unit. A best mean can hide a disastrous subgroup. Preregistered and exploratory comparisons can both be informative, but must not be relabelled after the fact.", "ML16 T8")
add("B70", "research", "294", "Worst-group risk and robust optimisation",
    "Average risk weights errors according to an evaluation distribution. Worst-group risk examines the hardest specified group; minimax optimisation chooses parameters to minimise an explicit worst-case objective. Distributionally robust optimisation optimises against a defined family of possible distributions.",
    "A classifier can improve its overall score while worsening a small-language group's error. A robust objective may trade some average performance for better protection of that group, but only for groups or distribution changes represented by its assumptions.",
    "A group-free robustness claim is underspecified. Tiny groups have uncertain estimates, and an uncertainty set is a modelling choice—not protection against every imaginable future dataset.", "ML15")
add("B71", "research", "299,300,319", "How to interpret a negative or mixed research result",
    "Start by checking implementation and comparison validity, then distinguish 'the idea did not help here' from 'the idea cannot work'. A negative result can be useful if it rules out a plausible explanation under a strong, clearly scoped test.",
    "If a new method wins on three domains and loses on seven, inspect effect sizes, label definitions, variance, compute and meaningful subgroup patterns. Compare a strong tuned baseline and matched controls before telling a mechanistic story. Report the unsuccessful comparisons rather than selecting the three wins as the whole evaluation.",
    "A weak baseline or a missing held-out domain may require another experiment, not more persuasive prose. A publishable consequence must be supported; failure after months of work is not itself a contribution.", "T8 N24")

add("B72", "serving", "105", "A GPU is allocated but utilisation is low",
    "Allocation means the scheduler gave you a device; utilisation measures its activity over a sampling window. The GPU may wait for data, CPU tokenisation, transfers, synchronisation or tiny kernel launches. Low utilisation does not identify the bottleneck by itself.",
    "Time a few steady-state batches and profile data loading, transfer, forward, backward and optimiser work. Compare a cached/synthetic-input diagnostic with the real data path. If the GPU is waiting for the loader, larger model layers are not the immediate fix.",
    "Check batch/sequence distributions, CPU workers, pinned-memory/transfer choices, logging synchronisation and precision support. Optimise a measured bottleneck without changing the experimental objective unnoticed.", "T6 T7")
add("B73", "concept-nlp", "134,135", "Embedding retrieval finds the wrong meaning",
    "An embedding index can be mechanically correct but encode the wrong similarity for the application. Separate representation mismatch from preprocessing, index approximation and relevance-label problems.",
    "For 'Apple released a phone' versus fruit-related apple passages, inspect actual top neighbours, tokenisation and score distributions. Compare exact vector search with the approximate index to see whether approximation caused the miss. Then compare lexical/hybrid retrieval and a query–candidate reranker under reviewed relevance labels.",
    "A general embedding model may not represent technical identifiers or the new domain well. Fine-tune only with suitable positive/negative examples and a clean held-out query set; do not tune until one demonstration query looks good.", "N15 R4")
add("B74", "concept-nlp", "195,304", "Choose a text classifier under a fixed data and GPU budget",
    "Start from the required label, language/domain coverage, latency and available examples, not from the largest model that fits. Use a lexical linear baseline, a suitable pretrained encoder and a prompted model as meaningful alternatives; adapt a larger generator only if the task and evidence justify it.",
    "With 500 labels, cross-validation and a pretrained representation may be more useful than a large fine-tuning sweep. With 50,000 labels and a 24 GB GPU, encoder fine-tuning is a credible candidate; QLoRA is an alternative, not an automatic winner. Active learning can target informative cases but needs a real annotation budget.",
    "Weak labels and synthetic examples require validation and provenance. Measure per-class error, calibration if used, serving cost and robustness. A larger model's availability is not a reason to skip a strong small baseline.", "SK02 NC08 R1")
add("B75", "systems", "240", "Design a source-checked literature-review workflow",
    "Separate finding a paper, extracting its claims and checking those claims. Store structured records with identity/version, task, data, method, metric, result, limitation and exact evidence locations. A model-produced citation string is not a verified publication.",
    "Read the authorised document, extract a result with its table/page reference, validate the record schema, and check that the cited passage really supports the result. Deduplicate paper versions and distinguish a preprint from a later publication. Route uncertain identity or interpretation to a human rather than fabricating missing metadata.",
    "Evaluate field correctness and evidence alignment on reviewed records. Avoid uploading confidential manuscripts without permission, and do not let text inside a paper issue tool instructions. This is a system-design answer, not a claim that the local lab already implements a production literature agent.", "R5 N20")
add("B76", "design", "320", "Design a multilingual document-analysis service",
    "Break the job into document ingestion, classification, extraction, retrieval/grounded answering and escalation. Share versioned document identities and access permissions across stages, but use a metric suited to each output rather than one average 'AI accuracy'.",
    "Preserve layout and source offsets when parsing. Evaluate class F1, extraction span/field accuracy, retrieval recall, evidence support and human-review coverage per language. Reuse small specialised models where adequate; send difficult authorised cases to a larger model or human under explicit budgets.",
    "Check OCR and tokenisation failures, low-resource language imbalance, stale indexes, unsupported answers and unsafe writes. Version the parser, model, tokenizer and index together. Deployment requires monitoring delayed labels, tail latency and rollback—not just a successful demonstration.", "N19 N24 R8")
add("B77", "research", "333", "Detect duplicates without confusing identity and similarity",
    "Exact duplicate checks compare a defined normalised representation or its hash across splits. Near-duplicate checks seek highly similar content and require a threshold or grouping policy. Duplicate IDs and duplicate texts are different issues.",
    "Two records can have different IDs but identical passage text. First verify ID uniqueness, then compare cross-split text hashes after a documented normalisation. For near duplicates, candidate retrieval with shingles/MinHash or embeddings can reduce work; review boundary cases and group related documents before splitting.",
    "Aggressive normalisation can erase meaningful differences, and a hash detects byte/content equality under its input convention, not semantic equivalence. Check provenance, recurring templates and passage/patient groups rather than treating every shared phrase as leakage.", "SK08 NC08")
add("B78", "research", "334", "Build a multi-seed evaluation report you can trust",
    "Each run needs a configuration, seed, code/data identity, complete prediction IDs and a declared selection rule. Aggregate only comparable runs. Mean and standard deviation across seeds describe training variation; an interval over resampled test units asks a different question.",
    "For each source/condition/seed, save predictions keyed by dataset and example ID. Reject missing or duplicate rows, compute the same per-dataset metric, then form the specified equal-dataset or pooled summary. Keep per-seed and per-domain results instead of reporting only the best seed.",
    "A failed run is not silently a zero or an omitted inconvenience. Report failures and reused fits. Use paired or clustered resampling where the design supports it; do not pretend seeds sharing a test set are independent new test populations.", "NC08 LC10")

# Incoming questions already answered by a concrete scenario in the original bank.
for spec, target in [
    ("15", "IML11"), ("16,21,23,25,30", "IML01"), ("17,18", "IML06"),
    ("22", "IML10"), ("29", "IML12"), ("45", "IML07"), ("46,48,59", "IML03"),
    ("58,60,276", "IDL12"), ("64,65", "IML02"), ("74", "IML08"), ("75", "INLP01"),
    ("90", "IML02"), ("97", "IDL06"), ("104", "M06"),
    ("120", "INLP11"), ("127,128", "INLP07"), ("165,194", "M09"),
    ("169", "M12"), ("180", "M16"), ("211,212,216", "S02"), ("214,215", "S03"), ("225", "D01"),
    ("233,234", "S07"), ("239", "P11"), ("260", "M16"),
    ("264-266", "IML12"), ("267", "IDL12"), ("275", "F09"), ("279", "S11"),
    ("280", "D03"), ("301", "IML03"), ("302", "M16"), ("303", "D03"),
    ("305", "INLP11"), ("306", "S01"), ("307", "M07"),
    ("308", "IDL07"), ("309", "INLP11"), ("310", "S11"), ("311", "B68"),
    ("312", "F15"), ("313", "IML07"), ("314", "B46"), ("315", "IML03"),
    ("316", "B64"), ("317", "INLP07"), ("318", "S08"),
    ("323", "P03"), ("327", "P04"), ("335", "S01")
]: route(spec, target)

# These are implementation exercises, not a request for another training runner.
for spec, target in [("321,331", "DS09"), ("322", "SK03"), ("324", "PT06"),
                     ("325,330", "PT05"), ("326,328", "PT09"), ("329", "LC04"), ("332", "NC06")]:
    route(spec, target, "lesson")


def source_prompts():
    source = ROOT / "data/ML_Interview_335_Questions.md"
    if not source.exists():
        # Release packages use the reviewed snapshot, not an arbitrary file path.
        source = ROOT / "learning_materials/interview_preparation/IMPORTED_335_PROMPTS.md"
    if not source.exists():
        from .materials import MATERIALS
        source = MATERIALS / "interview_preparation/IMPORTED_335_PROMPTS.md"
    text = source.read_text(encoding="utf-8").split("## Suggested answer structure")[0]
    prompts = {}
    for num, title in re.findall(r"^(\d+)\. (.+)$", text, re.M):
        title = re.sub(r"[★◆▣*]", "", title).strip()
        prompts[int(num)] = title
    if set(prompts) != set(range(1, 336)):
        raise ValueError("Imported checklist must contain exactly questions 1–335.")
    if set(ROUTES) != set(prompts):
        raise ValueError("Unrouted or extra imported prompts: " + str(set(prompts) ^ set(ROUTES)))
    return [{"number": number, "title": title, **ROUTES[number]} for number, title in sorted(prompts.items())]


def extension_items():
    return EXTRA
