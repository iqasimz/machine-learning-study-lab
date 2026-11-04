# How the methods relate

These are problem-driven branches, not a claim that each row historically replaced the preceding row.

| Question or limitation | Method family | What changes | What remains difficult |
| --- | --- | --- | --- |
| Need a reference before modelling | Dummy/seasonal baseline | Constant or repeated past pattern | Cannot use richer features |
| Need an interpretable numeric relationship | Linear/GLM models | Weighted feature combination, declared link/loss | Feature design, outliers, nonlinearity, causal interpretation |
| Need nonlinear local behaviour | k-NN / kernels | Neighbour geometry or implicit feature space | Scale, high dimensions, sample-size cost |
| Need class probabilities from simple distributions | Naive Bayes / LDA / QDA | Model class-conditional data and priors | Misspecification and calibration |
| Need feature interactions without global scaling | Trees | Piecewise feature-space partitions | Instability and tiny leaves |
| Need more stable tree predictions | Random forests | Randomized tree averaging | Correlated bias, storage, leakage |
| Need sequential correction of fitting errors | Boosting | Additive learners following the objective | Tuning, overfitting, cost |
| Need unlabeled structure or compression | Clustering / PCA | Grouping or feature projection | Structure is not target truth |
| Need learned nonlinear features | MLPs | Jointly fitted representations and task head | Data, optimization, inductive bias |
| Need spatial sharing | CNNs | Local reused filters, receptive fields | Long-range structure, invariance limits |
| Need deeper compositional fitting | Residual networks | Additive skip paths | More depth does not automatically help |
| Need reuse of prior learned features | Transfer | Pretrained starting information | Exposure confounds, negative transfer |
| Need ordered state | RNN / GRU / LSTM | Shared temporal updates and gated memory | Long derivative paths and state handling |
| Need flexible content interactions | Attention / Transformers | Pairwise matching and value mixing | Context cost, positions, faithful evidence use |
| Need training targets without many external labels | Autoencoding / contrastive learning | Reconstruction or view-consistency objectives | Task utility and augmentation assumptions |
| Need relational information | GNNs | Neighbourhood message passing | Graph shortcuts, transductive access, oversmoothing |
| Need a distribution/sample model | VAE / GAN / diffusion | Different likelihood/adversarial/denoising signals | Quality, diversity, memorization and compute |
| Need sequential consequences | Bandits / RL | Feedback depends on selected actions and transitions | Exploration, confounding, reward design and safety |

## Controlled comparisons instead of slogans

- **Tabular:** dummy → ridge/logistic → tree/forest/boosting → MLP. Keep the strongest valid classical competitor.
- **Image:** same pixels and split → linear head → MLP → small CNN. Report when reshaping, augmentation, parameter count or training exposure changes.
- **Sequence:** seasonal/lag baseline → recurrent model → attention where justified. Keep the forecast horizon and available history fixed.
- **Representation:** raw features → PCA → autoencoder/contrastive features with the same downstream probe.
- **Graph:** node-only MLP → permitted-edge aggregation → edge-ablation control. Do not grant target labels to one branch.
- **Pretraining:** show scratch and pretrained conditions separately. Extra pretraining data is not just a better head.

The point is not to display many bars. One well-designed contrast answers why a component helped or failed. The [experiment contracts](metadata/experiments.json) define the first planned comparisons.
