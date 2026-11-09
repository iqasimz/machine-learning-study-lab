# Coverage and limits

## Current authored course and local lab

Thirty integrated lessons: 16 ML and 14 DL, with definitions, process explanation, worked calculations, alternatives, scenarios, question/answers and official references. Twenty lessons add a deeper derivation or implementation walkthrough. The reader imports all relevant definitions from 98 teaching parameter cards. There are 24 practical ML/DL interview questions and six static method figures. Ten research data cards include held/conditional candidates; they are not all executable datasets.

The training labs offer classical methods, neural image and sequence models, a small vision transformer, a scratch decoder and pretrained LoRA adaptation on compatible datasets. The lab menus list current executable options. These are real training, evaluation and persistence paths in a local installation, not evidence that every course topic has a trainer.

This is an independently authored first edition, not a Stanford-equivalent qualification or a certificate of proficiency. It is not exhaustive mathematical ML/DL coverage. Deeper proofs, additional coding assessments and independent human teaching review remain useful extensions.

## Coverage benchmark

| Coverage family | Main material | Depth in this edition |
| --- | --- | --- |
| Problem definition, leakage, baselines | ML01/ML02/ML16 | Core procedural explanation and worked cases |
| Linear models, GLMs, regularization, selection | ML03–ML05 | Core mechanics; full normal-equation/convex analysis derivations remain extension work |
| Neighbours, kernels, margins | ML06 | Core calculation; full SVM dual/KKT derivation not yet authored |
| Probabilistic classification | ML07 | Core factorization, posterior example, LDA/QDA contrast |
| Trees, forests, boosting, stacking | ML08/ML09 | Core mechanics and numerical examples; implementation-specific boosting derivations need the chosen backend |
| Metrics, uncertainty, generalization | ML10/ML15 | Core distinctions, bounded examples, finite-class/GP/conformal orientation; not a full mathematical statistics course |
| Clustering, EM, PCA, anomalies | ML11/ML12 | Core k-means/PCA mechanics; full EM likelihood derivation and manifold methods are deeper branches |
| Forecasting | ML13 | Causal evaluation and baseline foundation; full ARIMA/state-space course is not included |
| Recommendation, bandits/RL | ML14 | Introductory decision/feedback bridge; not a complete RL/control course |
| Neural approximation/autograd/optimization | DL01–DL04 | Core numerical mechanics; deeper initialization/convergence proofs remain optional |
| CNNs, residuals, transfer | DL05/DL06 | Core shapes, parameters, data/exposure distinctions |
| RNN/GRU/LSTM and attention | DL07/DL08 | Core sequence flow and bridge to existing full LLM lessons |
| Autoencoding/self-supervision/generation | DL09/DL10 | Objective distinctions and toy calculations; no promised SOTA image-generation training |
| Graphs and structured vision tasks | DL11/DL12 | Bounded mechanism/shape lessons; not a full graph-learning or detection/segmentation programme |
| Systems, reproducibility, comparison | DL13/DL14 | Working local fit/train/evaluate/persist/compare workflow; production/distributed serving remains outside the current lab |

## Explicit specialist backlog

Survival analysis/censoring; causal identification and treatment estimation; econometric inference; advanced convex duality and optimization proofs; Bayesian computation/MCMC; probabilistic graphical models; full Gaussian-process approximation methods; complete EM derivation; manifold-learning mathematics; large-scale ranking/recommenders; offline policy evaluation; deep RL/control; speech/audio encoders and CTC; object detection/segmentation training; advanced diffusion/flow models; distributed DL implementation; federated/privacy training; formal safety and certified robustness.

Relevant names may be introduced in the branch map, but introduction is not complete teaching or implementation. Add a specialist only when it serves a real question with prerequisites, derivation, data, controls and checks—not to extend a list of fashionable terms.

## Runtime boundaries

The runtime registry in training_lab/model_lab.py is authoritative for executable models and validated choices; the JSON teaching catalogue contains broader options. Forecasting, recommendation/RL, GP fitting, recurrent/GNN trainers, pretrained image transfer, contrastive and stochastic image-generation training remain course-only. Dataset uploads and untrusted model loading are not implemented. Downloads and training are explicit user actions.
