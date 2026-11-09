# The Ultimate ML, Deep Learning, NLP & LLM Interview Question Bank

**335 interview and coding questions | Fundamentals to research-level | Theory, comparisons, debugging, coding and real-world scenarios**

Designed for **Machine Learning Engineer, AI Engineer, NLP Engineer, LLM Engineer, Applied Scientist, Research Engineer, and PhD interviews**.

From deceptively simple questions like *bias vs. variance* to advanced problems involving transformer internals, fine-tuning, hallucinations, evaluation, distribution shift, and production AI systems.

**Legend:**
- ★ = High priority / commonly asked
- ◆ = Tricky / advanced
- ▣ = Practical scenario

*The priority markers are study recommendations, not measured interview frequencies.*

---

# Part I — Machine Learning Fundamentals

## 1. Core concepts and tricky comparisons

1. ★ What is the difference between **bias and variance**? Can a model have both high bias and high variance?
2. ★ What is the difference between **overfitting and underfitting**? How do you identify them from learning curves?
3. ★ What is the bias–variance trade-off? Is it always true that increasing model complexity increases variance?
4. ★ What is the difference between supervised, unsupervised, semi-supervised, and self-supervised learning?
5. ★ What is the difference between classification, regression, clustering, and ranking?
6. What is the difference between parametric and non-parametric models?
7. ★ What is the difference between model parameters and hyperparameters?
8. What is the difference between generative and discriminative models? Can a model exhibit characteristics of both?
9. ◆ Why can a simpler model outperform a more complex model with greater representational capacity?
10. What is the difference between lazy learning and eager learning?
11. ◆ Does having more training data always improve generalization?
12. What is the curse of dimensionality, and why does it affect nearest-neighbor methods?
13. ◆ Can a model achieve zero training error and still generalize well? Explain double descent.
14. What is the difference between inductive learning and transductive learning?
15. ◆ What is the difference between correlation, causation, and predictive usefulness?

## 2. Data preprocessing, splitting, and leakage

16. ★ What is data leakage? Explain target leakage, preprocessing leakage, and train–test contamination.
17. ★ Why should normalization or standardization be fitted only on training data?
18. ★ What is the difference between normalization, standardization, and robust scaling?
19. How do you handle missing values? When is imputation inappropriate?
20. What is the difference between MCAR, MAR, and MNAR missingness?
21. ★ Why do we need separate training, validation, and test datasets?
22. ◆ Why is random splitting unsuitable for time-series data?
23. ◆ How can duplicate or near-duplicate samples inflate evaluation scores?
24. What is the difference between stratified sampling and ordinary random sampling?
25. What is group-aware splitting, and when is it essential?
26. ★ What is the difference between covariate shift, label shift, and concept drift?
27. ◆ How do you identify out-of-distribution inputs without having labels for them?
28. What is the difference between class imbalance and sampling bias?
29. ▣ Your model achieves 97% test accuracy but only 68% after deployment. What would you investigate first?
30. ▣ You discover that documents from the same authors appear in both training and test sets. How would you redesign the evaluation?

## 3. Probability, statistics, and mathematical intuition

31. ★ What is the difference between probability and likelihood?
32. ★ What is the difference between maximum likelihood estimation (MLE) and maximum a posteriori estimation (MAP)?
33. What is Bayes' theorem, and how is it used in classification?
34. ◆ What is the difference between Bayesian and frequentist approaches to learning?
35. What is the difference between mean, median, mode, variance, and standard deviation?
36. What is covariance, and how is it different from correlation?
37. ★ What is the difference between statistical significance and practical significance?
38. ◆ Why does a p-value not represent the probability that the null hypothesis is true?
39. What are confidence intervals, and how do they differ from prediction intervals?
40. What is the central limit theorem, and when does it apply?
41. What is the difference between bootstrap resampling and permutation testing?
42. ◆ How do multiple comparisons affect the reliability of experimental conclusions?
43. What are entropy, cross-entropy, and KL divergence? How are they related?
44. Why is KL divergence asymmetric, and when is Jensen–Shannon divergence more useful?
45. ▣ Two models differ by 0.8 percentage points in F1 on a small test set. How would you determine whether the difference is meaningful?

## 4. Evaluation metrics and their traps

46. ★ Why can accuracy be misleading for imbalanced datasets?
47. ★ Explain precision, recall, specificity, sensitivity, and F1-score.
48. ★ When should you prioritize precision over recall, and vice versa?
49. ★ What is the difference between micro, macro, weighted, and sample-averaged F1?
50. What is the difference between ROC-AUC and PR-AUC? When is PR-AUC more informative?
51. ◆ Why can ROC-AUC remain high even when precision is poor?
52. What is the difference between log loss, cross-entropy loss, and Brier score?
53. ★ What is probability calibration? Can a highly accurate model be poorly calibrated?
54. Why might optimizing F1 lead you to select a decision threshold other than 0.5?
55. What are top-k accuracy, mean average precision, and NDCG?
56. ◆ Why is averaging F1 across datasets different from computing pooled F1?
57. ◆ How can accuracy improve while F1 decreases?
58. What is the difference between offline evaluation and online A/B testing?
59. ▣ Model A has 92% accuracy, Model B has 88%, but Model B produces fewer expensive business mistakes. Which would you deploy?
60. ▣ A new model improves macro-F1 from 0.81 to 0.83 but doubles inference cost. How would you decide whether to ship it?

## 5. Classical ML algorithms

61. ★ What is the difference between linear regression and logistic regression? Why is logistic regression considered a classification algorithm?
62. ★ What are the assumptions of linear regression, and what happens when they are violated?
63. What is multicollinearity, and how does it affect regression coefficients?
64. ★ What is the difference between L1 (Lasso), L2 (Ridge), and Elastic Net regularization?
65. ◆ Why does L1 encourage sparsity while L2 typically shrinks weights without making them exactly zero?
66. ★ How does a decision tree choose its splitting features? Explain Gini impurity and information gain.
67. ★ What is the difference between bagging, boosting, and stacking?
68. ★ Compare Random Forest, XGBoost, LightGBM, and CatBoost.
69. ◆ Why can gradient boosting overfit, despite combining weak learners?
70. What is the difference between a hard-margin and soft-margin SVM?
71. Explain the kernel trick. Why can kernel methods be computationally expensive?
72. ★ What is the difference between K-means, DBSCAN, and hierarchical clustering?
73. What is the difference between PCA and t-SNE or UMAP?
74. ◆ Why is PCA not guaranteed to preserve the most predictive information?
75. ▣ You have 10,000 samples and 50,000 sparse features. Which models would you try first, and why?

---

# Part II — Deep Learning

## 6. Neural network fundamentals

76. ★ What is a neuron in a neural network, mathematically?
77. ★ Why do neural networks need nonlinear activation functions?
78. ★ Compare ReLU, Leaky ReLU, GELU, sigmoid, and tanh.
79. ◆ Why can stacking linear layers without activation functions be collapsed into a single linear transformation?
80. ★ How does backpropagation work? Explain it using the chain rule.
81. What is the difference between forward propagation and backward propagation?
82. ★ What are vanishing and exploding gradients, and why do they occur?
83. ★ What is the difference between Batch Normalization, Layer Normalization, RMSNorm, and Group Normalization?
84. Why does weight initialization matter? Compare Xavier and He initialization.
85. ★ What is dropout, and why is it applied differently during training and inference?
86. ◆ Is dropout equivalent to using an ensemble of neural networks?
87. What are residual connections, and how do they help deep networks train?
88. What is the difference between model depth, width, capacity, and parameter count?
89. ◆ Why can an overparameterized neural network generalize well?
90. ▣ Your neural network's training loss decreases, but validation performance stagnates. How would you diagnose it?

## 7. Optimization, training, and debugging

91. ★ Compare batch gradient descent, stochastic gradient descent, and mini-batch gradient descent.
92. ★ What is the difference between SGD, SGD with momentum, Adam, AdamW, and RMSProp?
93. ★ Why is AdamW different from Adam with naive L2 regularization?
94. ★ What happens when the learning rate is too high or too low?
95. What are learning-rate warmup, cosine decay, and learning-rate scheduling?
96. ◆ Why can a larger batch size sometimes hurt generalization?
97. What is gradient accumulation, and is it always equivalent to using a larger physical batch?
98. ★ What is gradient clipping, and when is it useful?
99. What is early stopping, and how do you choose its patience?
100. ★ What is the difference between training loss and validation loss?
101. What is mixed-precision training? Compare FP32, FP16, BF16, and FP8.
102. ◆ Why can FP16 training produce NaNs while BF16 training remains stable?
103. What is gradient checkpointing, and what is its computational trade-off?
104. ▣ Your loss becomes NaN after 2,000 optimization steps. How would you isolate the cause?
105. ▣ GPU utilization is only 35% despite having a large model and dataset. What would you investigate?

---

# Part III — NLP Foundations

## 8. Classical NLP and linguistic concepts

106. ★ What is the difference between stemming and lemmatization?
107. ★ What is tokenization? Compare word, subword, character, and byte-level tokenization.
108. ★ What is the difference between Bag-of-Words and TF-IDF?
109. Why does TF-IDF sometimes outperform deep embeddings for retrieval?
110. What is the difference between n-gram language models and neural language models?
111. What are out-of-vocabulary tokens, and how do modern tokenizers reduce this problem?
112. ★ What is the difference between POS tagging, named entity recognition, and dependency parsing?
113. What is the difference between syntactic and semantic information in NLP?
114. What is the difference between sentence classification, token classification, and sequence labeling?
115. What are Hidden Markov Models and Conditional Random Fields?
116. ◆ Why might a CRF layer improve a neural sequence-tagging system?
117. What is exposure bias in sequence generation?
118. What are BLEU, ROUGE, METEOR, BERTScore, and chrF? What are their limitations?
119. ◆ Why can two semantically equivalent sentences have low lexical-overlap scores?
120. ▣ A sentiment model performs well on product reviews but poorly on sarcastic social-media comments. How would you improve it?

## 9. Word embeddings, sentence representations, and semantic search

121. ★ What is the difference between one-hot encoding and learned embeddings?
122. ★ Compare Word2Vec, GloVe, FastText, and contextual embeddings.
123. What is the difference between CBOW and Skip-gram?
124. ★ What is the difference between static and contextual word embeddings?
125. What is cosine similarity, and how is it different from dot product and Euclidean distance?
126. ◆ Why can cosine similarity become less informative in high-dimensional embedding spaces?
127. ★ What is the difference between a bi-encoder and a cross-encoder?
128. Why are cross-encoders generally more expensive for large-scale retrieval?
129. What is contrastive learning, and how does InfoNCE work?
130. What is the difference between positive pairs, random negatives, and hard negatives?
131. ◆ What is embedding anisotropy, and why might it matter?
132. What is the difference between semantic similarity and semantic relatedness?
133. What is approximate nearest-neighbor search? Compare HNSW, IVF, and product quantization.
134. ◆ Why can embedding models trained on one domain fail to retrieve relevant documents in another?
135. ▣ Your embedding search considers "Apple released a new phone" and "I bought fresh apples" highly similar. How would you investigate the problem?

---

# Part IV — Transformers and LLM Architecture

## 10. Transformers from first principles

136. ★ What is self-attention, and why did it become central to modern language models?
137. ★ Explain queries, keys, and values in attention.
138. ★ Derive the scaled dot-product attention equation:

    $$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

139. ◆ Why are attention scores divided by $\sqrt{d_k}$?
140. ★ What is the difference between self-attention, cross-attention, and masked causal attention?
141. ★ What is multi-head attention, and why not just use one larger attention head?
142. What is the difference between Multi-Head Attention, Multi-Query Attention, and Grouped-Query Attention?
143. ★ Compare encoder-only, decoder-only, and encoder–decoder transformer architectures.
144. Why do transformers need positional information?
145. Compare sinusoidal positional embeddings, learned position embeddings, RoPE, and ALiBi.
146. ◆ Can a transformer represent word order without explicit positional encodings?
147. What is the computational complexity of standard attention with respect to sequence length?
148. ★ What is the difference between padding masks and causal masks?
149. Why do modern transformers often use pre-normalization instead of post-normalization?
150. ▣ You double the context length of a transformer from 4,096 to 8,192 tokens. What happens to attention computation, activation memory, and inference costs, and what optimizations could change those relationships?

## 11. LLM fundamentals, pretraining, and scaling

151. ★ What is a large language model, and how does it differ from a conventional NLP classifier?
152. ★ What is the difference between BERT, GPT-style models, and T5?
153. ★ What is next-token prediction, and why can it lead to useful representations and general capabilities?
154. What is the difference between masked language modeling, causal language modeling, and denoising objectives?
155. ★ What are tokens, vocabulary size, and context length? How do they affect computational cost?
156. What is perplexity? Why can lower perplexity fail to translate into better downstream-task performance?
157. ◆ Can an LLM have excellent perplexity but perform poorly on reasoning tasks?
158. What are scaling laws, and how do model size, data quantity, and compute interact?
159. What is compute-optimal training, and why might a smaller model trained on more data outperform a larger model?
160. ★ What is catastrophic forgetting?
161. What is continued pretraining, and how does it differ from supervised fine-tuning?
162. What is knowledge distillation? Compare teacher–student training and self-distillation.
163. What is a Mixture-of-Experts model, and how does sparse activation work?
164. ◆ Why might a model with more total parameters require fewer active FLOPs per token than a dense model?
165. ▣ You have domain-specific technical documents and an open-weight LLM. How would you determine whether continued pretraining, supervised fine-tuning, RAG, or no additional training is necessary?

## 12. LLM alignment, reasoning, and post-training

166. ★ What is the difference between pretraining, instruction tuning, and preference optimization?
167. ★ What is RLHF, and what problems is it intended to solve?
168. What is a reward model, and how is it trained from preference comparisons?
169. ★ What is the difference between RLHF using PPO and Direct Preference Optimization (DPO)?
170. What is the difference between DPO, IPO, ORPO, and GRPO?
171. ◆ Why can preference optimization improve human-rated quality without necessarily improving factual accuracy?
172. What is reward hacking, and how can it appear during LLM training?
173. What is KL regularization in preference optimization, and why might it be useful?
174. What is the difference between outcome rewards and process rewards?
175. What are rejection sampling, best-of-N sampling, and test-time compute scaling?
176. ★ What is the difference between chain-of-thought prompting and ordinary direct-answer prompting?
177. ◆ Does a model's generated reasoning trace necessarily explain how it reached its answer?
178. What are self-consistency decoding, verifier models, and reasoning-model distillation?
179. What is the difference between reasoning capability, reasoning-token expenditure, and task accuracy?
180. ▣ You fine-tune a model to provide better explanations, but its factual accuracy drops. How would you test whether alignment, training data, or decoding caused the regression?

## 13. Fine-tuning, LoRA, QLoRA, and efficient adaptation

181. ★ What is the difference between full fine-tuning, LoRA, QLoRA, and prompt tuning?
182. ★ How does LoRA work mathematically? Why represent a weight update as a low-rank product?
183. What is the relationship between LoRA rank, scaling factor, trainable parameters, and adaptation capacity?
184. ★ What is QLoRA, and how can a quantized base model support adapter training?
185. What are NF4 quantization and double quantization?
186. ◆ Does QLoRA mean that every parameter of a 4-bit model is trained in 4-bit precision?
187. Why might LoRA underperform full fine-tuning?
188. What is the difference between applying LoRA to attention projections versus feed-forward layers?
189. ★ What are the differences between instruction tuning, classification fine-tuning, and continued pretraining?
190. What is loss masking? Why might you exclude user and system tokens from the SFT loss?
191. What are sequence packing and dynamic padding, and why can they influence training efficiency or correctness?
192. ◆ How can incorrectly configured chat templates, EOS tokens, or padding cause a fine-tuned model to behave badly?
193. How would you select hyperparameters for a LoRA experiment under a fixed GPU budget?
194. ▣ A fine-tuned 7B model performs worse than the original instruction model on your test set. How would you debug the result?
195. ▣ You have one 24 GB GPU and 50,000 labeled text examples. How would you build a competitive domain classifier, and which alternatives to LLM fine-tuning would you benchmark?

## 14. LLM inference, decoding, quantization, and serving

196. ★ What is the difference between greedy decoding, beam search, top-k sampling, and nucleus (top-p) sampling?
197. ★ How do temperature and sampling parameters affect output diversity and reproducibility?
198. ◆ Does setting temperature to zero guarantee identical outputs across all runs and serving systems?
199. What are repetition penalties and length penalties?
200. ★ What is a KV cache, and why does autoregressive generation benefit from it?
201. ★ What is the difference between prefill and decode phases in LLM inference?
202. What is the difference between time to first token (TTFT), inter-token latency, throughput, and end-to-end latency?
203. ★ What is quantization? Compare PTQ, QAT, INT8, 4-bit quantization, and FP8.
204. ◆ Why doesn't 4-bit quantization necessarily reduce total GPU memory usage by exactly 75%?
205. What is speculative decoding, and under what conditions can it improve performance?
206. What are FlashAttention and PagedAttention? Which different bottlenecks do they address?
207. What is continuous batching, and how does it differ from static batching?
208. What are tensor, pipeline, data, and expert parallelism?
209. ▣ A model fits comfortably in GPU memory but runs out of memory under concurrent requests. Why could that happen?
210. ▣ Your model serves 10 requests per second, but the business requires 100 with acceptable latency. What would you measure and optimize first?

---

# Part V — RAG, Agents, and Applied LLM Systems

## 15. Retrieval-Augmented Generation (RAG)

211. ★ What is RAG, and how does it differ from fine-tuning?
212. ★ What is the difference between sparse retrieval, dense retrieval, and hybrid retrieval?
213. What is BM25, and why does it remain competitive for retrieval?
214. ★ What is document chunking? Compare fixed-size, recursive, semantic, and structure-aware chunking.
215. ◆ How do chunk size and overlap affect retrieval precision, context coverage, and token costs?
216. ★ What is the difference between a retriever and a reranker?
217. What is query rewriting, query expansion, and multi-query retrieval?
218. What are metadata filtering, parent-document retrieval, and hierarchical retrieval?
219. How do you measure retrieval quality using recall@k, MRR, MAP, and NDCG?
220. ◆ How can increasing retrieval recall make final answer quality worse?
221. What are context precision, context recall, groundedness, and faithfulness?
222. How do you detect and mitigate conflicting information in retrieved documents?
223. What is the difference between RAG, long-context prompting, and a model with domain-specific continued pretraining?
224. ▣ Your RAG system retrieves the correct document but gives the wrong answer. How would you separate retrieval, context construction, and generation failures?
225. ▣ Design a RAG system for 500,000 technical PDFs that supports document updates, citations, access permissions, and low-latency queries.

## 16. AI agents, function calling, and workflows

226. ★ What is the difference between a chatbot, an LLM-powered workflow, and an autonomous agent?
227. ★ What is function calling, and how is it different from generating ordinary natural-language output?
228. What is the ReAct pattern, and how does it combine reasoning and tool use?
229. What is the difference between planning, execution, and reflection in an agent architecture?
230. What are structured outputs, JSON schema validation, and constrained decoding?
231. ◆ Why can a syntactically valid tool call still be semantically incorrect or unsafe?
232. What is the difference between short-term conversational state, external memory, and parametric knowledge?
233. How would you handle tool failures, retries, timeouts, and idempotency?
234. How do you prevent an agent from repeatedly calling the same tool without making progress?
235. What are single-agent and multi-agent architectures, and when does multi-agent design add unnecessary complexity?
236. ◆ Does adding more agent reasoning steps necessarily improve performance?
237. How would you measure agent success, tool-call correctness, and completion cost?
238. What are the differences between deterministic workflow orchestration and open-ended agent planning?
239. ▣ Your agent keeps modifying the wrong database records despite producing valid SQL. How would you make the workflow safer?
240. ▣ Design an agent that reads research papers, extracts methodology and results, validates citations, and creates structured literature-review records.

---

# Part VI — Evaluation, Safety, and Production AI

## 17. LLM evaluation, hallucinations, and reliability

241. ★ What is hallucination in an LLM, and why does it occur?
242. ★ What is the difference between factual accuracy, faithfulness, relevance, and helpfulness?
243. ★ How would you evaluate an LLM when there is no single correct reference answer?
244. What is LLM-as-a-judge, and when is it reliable or unreliable?
245. ◆ What are position bias, verbosity bias, and self-preference bias in LLM evaluation?
246. What is the difference between exact-match evaluation, semantic evaluation, and human evaluation?
247. How would you build an evaluation dataset for a domain-specific assistant?
248. ★ What is benchmark contamination, and how does it distort model comparisons?
249. What is the difference between model calibration, uncertainty estimation, and selective prediction?
250. ◆ Why can a model confidently produce an incorrect answer even when its next-token probabilities are well estimated?
251. What are adversarial examples, jailbreaks, and prompt injection attacks?
252. ★ What is the difference between a system instruction and untrusted retrieved content?
253. How do you evaluate toxicity, harmfulness, fairness, and privacy risks?
254. What is the difference between memorization, verbatim extraction, and generalization?
255. How do you evaluate an LLM's robustness to paraphrasing, formatting changes, and misleading context?
256. ◆ Why can leaderboard improvements fail to translate into real-world usefulness?
257. What is the difference between pairwise evaluation, rubric-based evaluation, and absolute scoring?
258. How do you evaluate a refusal mechanism without rewarding excessive refusal?
259. ▣ An LLM's human preference score increases while its factual correctness decreases. How would you interpret this?
260. ▣ You need to compare five open-weight LLMs for a sensitive document-analysis application. Design a rigorous evaluation protocol.

## 18. MLOps, deployment, and production engineering

261. ★ What is the difference between deploying a machine learning model and deploying a conventional software application?
262. ★ What is the difference between batch inference, online inference, and streaming inference?
263. What are model registries, experiment tracking, and artifact versioning?
264. ★ What is the difference between data drift, concept drift, model degradation, and infrastructure failure?
265. How would you detect distribution drift without immediate access to ground-truth labels?
266. What is training–serving skew, and how can you prevent it?
267. ★ What is the difference between a model's offline accuracy and its online business performance?
268. What are canary deployments, shadow deployments, and blue–green deployments?
269. How would you implement rollback for a failing ML service?
270. What are SLAs, SLOs, latency percentiles, and error budgets?
271. ◆ Why might average latency look acceptable while users still complain about slow responses?
272. How would you monitor an ML system when labels arrive weeks after predictions?
273. How can feedback loops cause a deployed model to reinforce its own errors?
274. What is the difference between a reproducible experiment and a reproducible deployed system?
275. What are deterministic operations, random seeds, and sources of nondeterminism in GPU training?
276. How would you optimize model quality subject to constraints on latency, throughput, cost, and memory?
277. What are data parallelism and distributed training strategies such as DDP, FSDP, and ZeRO?
278. ◆ Why might distributed training with more GPUs fail to produce proportional speedups?
279. ▣ A classification API suddenly becomes slower after a model update. How would you isolate the root cause?
280. ▣ You must deploy a domain-specific LLM serving thousands of requests daily under a limited cloud budget. Describe your architecture, scaling strategy, monitoring, and fallback behavior.

---

# Part VII — Advanced Research and PhD-Level Questions

## 19. Generalization, robustness, experiments, and scientific reasoning

281. ★ What is the difference between in-domain and out-of-domain generalization?
282. ★ What is the difference between domain adaptation, domain generalization, and transfer learning?
283. ◆ How can you distinguish a domain-shift problem from a change in annotation guidelines?
284. What is dataset bias, and how can models exploit superficial correlations instead of learning the intended task?
285. ◆ Why might a model trained on several datasets perform worse on one dataset than a specialist trained only on that dataset?
286. How would you harmonize inconsistent label definitions across multiple corpora?
287. What is the difference between label noise, ambiguous examples, and annotator disagreement?
288. ◆ Is majority voting always appropriate when annotations are subjective?
289. What are hard-negative mining and adversarial data augmentation, and when can they introduce unwanted bias?
290. How can data augmentation or text simplification change the original labels or meaning?
291. ★ What is an ablation study, and how does it differ from a hyperparameter sensitivity analysis?
292. How should you report performance across multiple seeds, datasets, and metrics?
293. ◆ Why might the model with the highest average score not be the most robust model?
294. What are worst-group risk, minimax optimization, and distributionally robust optimization?
295. How do you avoid selecting a model indirectly using test-set information?
296. What is the difference between a static ensemble, stacking, and a learned mixture-of-experts router?
297. ◆ Why can a learned ensemble or router perform worse than a simple average of model predictions?
298. How would you establish that a new modeling technique actually causes an improvement rather than merely correlating with one?
299. ▣ Your proposed research method improves performance on 3 out of 10 datasets but loses on the other 7. How would you analyze the result?
300. ▣ A reviewer says your NLP paper lacks evidence of generalization and relies on a weak baseline. What experiments would you add?

---

# Part VIII — Real-World Interview Scenarios

These are deliberately open-ended. A good answer requires diagnosis, assumptions, experiments, trade-offs, and validation.

## 20. Twenty difficult real-world case studies

301. **▣ The 99% accuracy trap.** You train a fraud detector achieving 99% accuracy, but it misses almost all fraudulent transactions. Diagnose the problem and propose a new training and evaluation strategy.
302. **▣ The impressive benchmark.** Your new model beats the published state of the art by 5 F1 points. How do you rule out leakage, duplication, preprocessing differences, test-set tuning, and invalid baselines?
303. **▣ The 7B vs. 70B decision.** Your company wants a customer-support assistant. A 70B model is more accurate but costly and slow. A 7B model is cheaper but makes more mistakes. How would you decide, including possible RAG, routing, and escalation strategies?
304. **▣ The limited-data problem.** You have only 500 labeled examples for a specialized NLP task. Compare the expected merits of traditional ML, encoder fine-tuning, prompting, weak supervision, synthetic data, and active learning.
305. **▣ The dataset shift.** A model trained on news articles is deployed on Reddit discussions and performance collapses. How would you diagnose linguistic, semantic, labeling, and distributional differences?
306. **▣ The hallucinating legal assistant.** Your RAG assistant invents legal citations. How would you modify retrieval, generation, citation verification, abstention, and evaluation to reduce unsupported answers?
307. **▣ The GPU bottleneck.** You must fine-tune an 8B model, but the available GPU cannot hold the model, gradients, and optimizer states. Explain your practical memory-saving options and their trade-offs.
308. **▣ The apparently successful fine-tuning.** Training loss declines substantially, but validation performance deteriorates. Design a systematic troubleshooting plan covering data, labels, regularization, learning rates, evaluation, and catastrophic forgetting.
309. **▣ The multilingual deployment.** You must build a classifier for English, German, Urdu, and Arabic with heavily imbalanced amounts of labeled data. How would you choose the base model, handle tokenization and language imbalance, and evaluate each language?
310. **▣ The latency incident.** An LLM API has acceptable average latency but severe slowdowns at peak traffic. How would you investigate queuing, prefill, decode, batching, KV-cache memory, and request scheduling?
311. **▣ The conflicting annotators.** Three annotators disagree over whether a sentence contains an argument. How would you determine whether the problem is annotation quality, ambiguity, or incompatible guidelines?
312. **▣ The cheaper retrieval system.** BM25 outperforms your expensive dense retriever on internal documents. What hypotheses would you test before replacing it?
313. **▣ The unstable experiment.** Five random seeds give substantially different F1 scores. How would you investigate instability, quantify uncertainty, and choose a fair model-selection protocol?
314. **▣ The ensemble failure.** You combine five independently strong NLP classifiers, but their ensemble performs worse than the best individual classifier. Explain possible causes and methods for diagnosing them.
315. **▣ The impossible threshold.** The business simultaneously demands 99% precision and 99% recall. Your best model cannot meet either requirement at the same threshold. How would you explain the trade-off and propose alternatives?
316. **▣ The model that says no.** A safety-trained assistant refuses many harmless questions. How would you evaluate and improve the balance between helpfulness and appropriate refusals?
317. **▣ The million-document search engine.** Build a semantic search system for one million research papers. Explain indexing, embeddings, hybrid retrieval, reranking, freshness, metadata, evaluation, and computational cost.
318. **▣ The adversarial document.** An attacker inserts instructions into a retrieved PDF that tell your assistant to reveal confidential information. How should the system architecture prevent the document from overriding authorized instructions?
319. **▣ The negative research result.** You spend three months developing a new model, but a simple RoBERTa baseline performs better. How would you analyze the result, communicate it, and determine whether the work is publishable?
320. **▣ The full system-design interview.** Design an end-to-end multilingual document-analysis service that performs classification, information extraction, grounded question answering, and confidence-based escalation. Explain your architecture, datasets, evaluation, deployment, monitoring, and expected failure modes.

---

# Part IX — Coding and Whiteboard Challenges

## 21. Practical Python, PyTorch, and ML implementation

321. ★ Implement precision, recall, and F1-score from raw predicted and true labels without using scikit-learn.
322. ★ Implement gradient descent for linear regression using NumPy.
323. ★ Implement a numerically stable softmax function.
324. Implement binary cross-entropy loss and explain why naive implementations can suffer numerical instability.
325. ★ Implement a simple multilayer perceptron and training loop in PyTorch.
326. Implement scaled dot-product attention using tensor operations.
327. Implement causal masking and explain the required tensor dimensions.
328. Implement multi-head attention and verify that the tensor reshaping is correct.
329. Implement a basic LoRA-enabled linear layer and calculate its number of trainable parameters.
330. ★ Write a PyTorch training and validation loop with gradient accumulation, mixed precision, and gradient clipping.
331. Implement macro-F1 and micro-F1 for a multiclass classifier, including appropriate zero-division handling.
332. Implement a simple cosine-similarity search function for document embeddings.
333. Write code to detect exact train–test duplicates and propose an approach for near-duplicate detection.
334. Build an evaluation pipeline that runs several random seeds and reports mean performance, dispersion, and confidence intervals.
335. Implement a small RAG pipeline using a document corpus, embeddings, retrieval, and generation, then describe how you would test each component independently.

---

# How to Prioritize Your Preparation

| Study track | Questions | Main focus |
| --- | --- | --- |
| ML fundamentals | 1–75 | Generalization, data leakage, statistics, evaluation, classical algorithms |
| Deep learning | 76–105, 136–150 | Backpropagation, optimization, transformer architecture |
| NLP foundations | 106–135 | Tokenization, text representations, retrieval, sequence modeling |
| LLM engineering | 151–225 | Pretraining, alignment, LoRA/QLoRA, quantization, inference, RAG |
| Applied AI | 226–280 | Agents, evaluation, safety, MLOps, production systems |
| Research | 281–300 | Domain shift, reproducibility, robust evaluation, methodology |
| Interview simulation | 301–335 | Case studies, system design, coding |

## Suggested answer structure

For each conceptual question, practice answering in four parts:

1. **Definition:** Explain the concept clearly and concisely.
2. **Mechanism:** Explain the technical intuition or mathematics.
3. **Example:** Give a concrete application or failure scenario.
4. **Trade-off:** Explain when it works, when it fails, and alternatives.

For scenario questions, start with diagnostic hypotheses, then explain the experiments, measurement strategy, and decision criteria.

Try a **30-second answer**, followed by a **3–5-minute technical explanation**. For NLP/LLM research and PhD interviews, emphasize questions **136–225** and **281–320**, while keeping the fundamentals (1–105) strong.
