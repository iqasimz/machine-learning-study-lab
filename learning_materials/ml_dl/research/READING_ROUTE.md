# Companion readings and visual resources

These supplement the original lesson drafts; a learner should not need to purchase a book or hold a Stanford login to understand our hand calculations. Do not redistribute author books or copy third-party site code/figures without permission. The [source registry](../metadata/sources.json) records access and caveats.

## Statistical learning

[ISLP, author site](https://www.statlearning.com/): use the Python edition. Read the statistical-learning, regression/classification, resampling, regularization, nonlinear, trees, SVM and unsupervised chapters alongside ML01–ML12. Attempt one chapter lab after reproducing our smaller hand example. The book's survival-analysis and multiple-testing chapters are useful later branches, not falsely marked fully taught here.

[CS229 overview](https://cs229.stanford.edu/): use it to understand prerequisite expectations and the breadth of supervised/unsupervised/theoretical learning. The currently linked detailed materials require Stanford access. Do not get stuck trying to access them; the author books and official implementation guides provide the public path.

[Understanding Machine Learning, author site](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/): use the PAC learning, uniform convergence and VC-dimension sections after ML15. The accessed author PDF explicitly permits personal use, not redistribution. Derive why a union bound makes a finite-class capacity term appear rather than memorizing the bound's final shape.

[GPML, author site](https://gaussianprocess.org/gpml/): read the regression/covariance-function treatment after the one-observation calculation in ML15. Separate latent-function variance from noisy-observation variance.

[Forecasting: Principles and Practice](https://otexts.com/fpp3/): use the forecast toolbox, time-series regression and evaluation sections alongside ML13. Examples use R; translate the information-time logic before translating syntax.

## Neural methods

[Deep Learning, author site](https://www.deeplearningbook.org/): chapters on feed-forward networks, regularization, optimization, convolution, sequence models and practical methodology support DL01–DL07/DL13. Autoencoder/representation/generative chapters extend DL09/DL10. Read the equation's assumptions, then calculate a small instance; do not replace explanation with a chapter link.

[D2L](https://d2l.ai/): use linear regression, softmax, MLP/autograd, convolution, recurrent networks and attention chapters as executable companions. Match tensor conventions and framework versions to our environment. Running a provided notebook does not demonstrate that you can explain its loss or fix a wrong mask.

[CS230 syllabus](https://cs230.stanford.edu/syllabus/) and [CS231n](https://cs231n.stanford.edu/): inspect the foundations → optimization → project strategy → architectures progression. Use the public [CS231n training notes](https://cs231n.github.io/neural-networks-3/) for gradient checks and failure diagnosis; do not imply access to restricted current lecture videos.

[PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html): trace data, Dataset/DataLoader, model, autograd, optimizer and saved state. Compare the order of its training loop with the code shown in the local DL lab.

## Optional visual companions

[TensorFlow Playground](https://playground.tensorflow.org/): try a nonlinear two-class pattern, compare no hidden layer with a small hidden representation, then change one setting. Predict the boundary/fit trade-off before running. This is an optional external learning tool, not this lab's training backend.

[CNN Explainer](https://poloclub.github.io/cnn-explainer/): candidate visual companion for convolution/activation/pooling. Its landing URL was reachable but not text-readable in this review; manually verify interaction/accessibility before assigning it as required work. No claim that its UI was tested here.

[Receptive-field derivation](https://distill.pub/2019/computing-receptive-fields/): trace the single-path receptive-field example after DL05, then compare a residual branch. Calculate which input pixels can influence one output location before considering the whole network.

## Three connected stories

1. **Fixed features to learned features:** a line fits a simple relationship; polynomial/kernel/tree methods represent nonlinear patterns in different ways; an MLP jointly learns intermediate features. Compare assumptions and evidence, not age.
2. **Pixels to spatial structure:** a linear classifier sees flattened pixels; an MLP learns unrestricted combinations; a CNN shares local filters; residual paths help deeper optimization; attention permits other interactions. Every transition changes costs and biases, not an automatic ranking.
3. **History to context:** lags summarize chosen past offsets; recurrence learns a state; gates control paths; attention connects permitted positions; the existing LLM course adds language objectives and a full decoder. The information available at prediction time remains the common test.

End each reading with one reconstructed calculation, one implementation/inspection task and one counterexample. A reference list without these outputs is not a completed lesson.
