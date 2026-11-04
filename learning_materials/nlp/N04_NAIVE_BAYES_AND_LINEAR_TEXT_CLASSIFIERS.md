# N04. Naive Bayes and linear text classifiers

A text classifier turns a feature representation into a task label. Multinomial Naive Bayes combines class-specific word probabilities; logistic regression combines learned feature weights into a conditional class score. Both can use simple lexical features, but they estimate different quantities and should be evaluated rather than ranked by their complexity.

The phrase 'good good bad' becomes the count vector [2,1]. We compare supplied positive/negative token counts with a fixed logistic weight vector. Switching models demonstrates their arithmetic; it does not train them or establish which is more accurate.

The previous lesson constructed sparse document features. This lesson attaches a class-prediction rule and shows why a baseline includes fitted parameters, preprocessing and independent evaluation—not simply a small architecture.

## 1. Class probabilities, linear weights and margins

Multinomial Naive Bayes estimates class priors and class-conditional token probabilities from counts. Additive smoothing gives unseen vocabulary items nonzero probability when its strength is positive. The conditional independence assumption is about features given the class; it does not assert that words are independent in real language. Compute products as sums of log probabilities.

Logistic regression fits a conditional class probability from a weighted feature vector. A linear SVM optimizes a margin objective; its raw decision values are not probabilities without a separate calibration procedure. NB-SVM-style features and character n-grams are useful comparison branches, not guaranteed winners.

The browser uses authored counts and a fixed illustrative logistic coefficient vector. It does not fit these models or claim an accuracy comparison. Compare prediction mechanisms first; measured performance requires fitting and held-out cases.

## 2. Reproducible lexical baselines

Save the feature pipeline, class mapping, regularization choice and development selection rule. Use macro-F1 or another justified metric from the shared metrics lesson, not training accuracy alone. A majority baseline and a lexical model answer different questions.

For imbalanced labels, class weights or resampling alter fitting; decision-threshold changes alter actions after scoring. Neither makes the underlying probabilities automatically calibrated. Read the biggest false positives and false negatives, including boilerplate and author markers that may be shortcuts. A fast, well-audited linear model is a meaningful baseline for a sentence encoder.

## Sources

[scikit-learn sparse text classification](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html) · [Speech and Language Processing — author-hosted draft](https://web.stanford.edu/~jurafsky/slp3/)
