# DL11. Graph neural networks

A graph represents objects as nodes and relationships as edges. A graph neural network, or GNN, updates node features by collecting messages from connected neighbours. Its calculation therefore depends on both the feature values and which connections are present; merely listing nodes in a table would omit important information.

The graph is A—B—C, with scalar features [1,2,4]. We calculate B's update using mean or sum neighbour messages and its own feature. A separate readout section applies the same rule to all nodes and shows a whole-graph sum.

CNNs share local filters on a regular image grid. Message-passing GNNs generalize local information exchange to an explicit graph; neighbour order should not arbitrarily change an invariant aggregation.

[course map](../COURSE_MAP.md). Required background: DL02, P2.

## 1. Topology, features and neighbour exchange

**Key terms.** A **graph** contains nodes and edges; a **message** carries neighbouring features; **aggregation** combines them; a **readout** produces node/edge/graph outputs. **Permutation invariance** means reordering neighbours does not change the aggregate.

**Process.** Build a task-appropriate graph with allowed features/edges, aggregate neighbourhood information, apply a learned transformation and repeat for a bounded number of layers. A task head then predicts only from information permitted by the split.

An MLP treats feature rows independently. A graph network introduces relational structure, which can help only if that structure is useful and available. Sum and mean aggregation encode different size information. The canonical normalized GCN rule uses degree-normalized adjacency with self loops; a simple mean-neighbour example is not numerically identical to that rule.

Node classification on one known graph can be transductive, with unlabelled test-node features/edges visible. Inductive evaluation instead asks about unseen nodes/graphs under stated access. Using test labels in messages is leakage in either case unless the task explicitly grants those labels. Too many layers can oversmooth representations; graph topology shortcuts can dominate genuine feature reasoning.

## 2. A worked calculation

**Symbols and assumptions.** h_v is node v's feature vector; N(v) is its neighbour set; AGG is a stated order-independent aggregation. A generic update applies a learned map to h_v and AGG({h_u:u∈N(v)}).

For a toy mean aggregator including self, node features [2,0] with two neighbours [0,2] and [4,2] give aggregate ([2,0]+[0,2]+[4,2])/3=[2,4/3]. Identity weights and ReLU leave it unchanged. Reordering the neighbours gives the same result; deleting an edge can change the result.

The numbers above are illustrative calculations, not measured benchmark results.

## 3. Choosing a graph aggregation rule

Compare an MLP on identical node features, a graph-only reference and message passing. This separates useful attributes from useful edges. Graph attention learns neighbour weights under a different rule; it does not prove that its largest edge caused a correct prediction.

The related settings are defined below this lesson in Model Lab, including their mathematical meaning, expected effects and interactions. The training lab exposes only controls that change an implemented operation; the broader [parameter catalogue](../metadata/parameters.json) also describes theory-only options.

## 4. Example: legitimate graph access

Use an authored graph with known edges and planted label leakage first. Public graph datasets need a later dataset/rights/split review; do not invent a default downloadable graph benchmark.

**Check.** Why is a node label appended to its input embedding usually a fatal evaluation error?

**Answer.** The target has been supplied as a feature. A sophisticated graph model can exploit it perfectly without learning the intended prediction problem.

## 5. Degree-normalized graph convolution

**Key terms.** A is adjacency, I adds self loops, and D̃ records degrees after adding them. Symmetric normalization weights an edge by both endpoint degrees. H contains node features.

**Process.** Form the permitted graph, add self loops, normalize adjacency, mix neighbouring node features and apply a learned feature transform and activation.

Read the formula from right to left as a feature transformation combined with graph-based mixing, followed by an activation. A+I supplies edges including self loops. The degree factors on both sides rescale contributions using each endpoint's connection count; this is a different layer from the walkthrough's simple self-plus-neighbour mean/sum update.

\[
H^{(\ell+1)}=\sigma\!\left(\widetilde D^{-1/2}(A+I)
\widetilde D^{-1/2}H^{(\ell)}W^{(\ell)}\right).
\]

In a two-node graph with one undirected edge, both self-looped degrees are two. Every normalized coefficient is 1/2; each node initially receives half its own features plus half its neighbour's. For a three-node chain, the endpoint self-weight is 1/2, the middle self-weight 1/3 and an endpoint-middle edge weight 1/√6. This is not simply a mean over each node's own neighbourhood.

### Evaluate the graph access policy

A train/test split over nodes can expose test-node features and edges during fitting in a transductive task while withholding labels. An inductive task with new graphs has a different information boundary. Neither can use test labels as features.

Compare an MLP using the same permitted attributes before attributing improvement to graph relations. No graph trainer or public graph-data download is silently enabled by reading this lesson.

## 6. GraphSAGE: learn an aggregation function, not just node identities

**Read the degree-normalized GCN section first.** A GCN mixes neighbors under its specified normalization. GraphSAGE instead emphasizes learning reusable neighborhood-aggregation functions, often with bounded neighbor sampling, so new nodes can be represented from permitted features and neighborhoods. “Inductive” does not mean the model works on every unseen graph without compatible features or topology.

A mean-aggregation variant calculates a neighborhood summary, concatenates it with the node's own representation, then applies a learned transformation:

\[
m_v=\operatorname{mean}_{u\in N(v)}h_u,\qquad
h'_v=\sigma\left(W[h_v\Vert m_v]\right).
\]

N(v) is the specified neighbor set and the parallel symbol denotes concatenation, not addition. Normalization, nonlinearities and whether self-features enter the aggregation vary by implementation. Sampling is a computational choice and also changes which information a particular forward pass sees.

For own scalar feature 2 and neighbor features 1 and 3, the mean is 2. Concatenation gives [2,2]. With row-compatible projection weights [0.5,1] and ReLU, the output is max(0,0.5×2+1×2)=3. A neighborhood [1,1,3,3] has the same mean despite twice as many entries. This illustrates lost multiplicity; sum aggregation would distinguish their totals, but changing aggregation also changes scale and model assumptions.

With ten sampled neighbors at each of two layers, one seed can require up to 1+10+100 sampled positions before overlap/deduplication. This is not necessarily 111 distinct nodes. Sampling prevents full-neighborhood expansion but can miss important rare neighbors. State train/evaluation sampling and graph access rules.

**Practice.** Can a learned identity embedding for an unknown node be looked up automatically? **Answer.** Not if it has no trained row. A feature-based aggregator can compute a representation when suitable new-node features and allowed neighbors are supplied. Compare it with a feature-only baseline to establish whether edges help.

[GraphSAGE](https://arxiv.org/abs/1706.02216) · [Aggregation expressivity](https://arxiv.org/abs/1810.00826).

## 7. Graph attention: learn weights within the allowed neighborhood

A graph attention network (GAT) calculates different mixing weights for a node's permitted neighbors. Unlike averaging all neighbors equally, a learned scoring function uses node features before normalizing the scores over that neighborhood. Connectivity still determines which candidates can contribute.

One original-style attention head forms projected features, concatenates a target/neighbor pair, scores it with a learned vector and a LeakyReLU, then applies neighbor softmax:

\[
e_{vu}=\operatorname{LeakyReLU}(a^\top[Wh_v\Vert Wh_u]),\quad
\alpha_{vu}=\frac{\exp(e_{vu})}{\sum_{j\in N(v)}\exp(e_{vj})},\quad
h'_v=\sigma\left(\sum_u\alpha_{vu}Wh_u\right).
\]

If supplied scores are log 3 and 0, weights are 0.75 and 0.25. Supplied projected scalar values 2 and 6 then mix to 3. These numbers illustrate normalization and mixing; they are not the result of fitting a GAT. Add a self-loop only under the declared graph convention. Multiple heads may concatenate or average outputs, affecting shapes and cost.

Follow the dependencies: permitted edges/features → projected pairs → scores → neighbor softmax → weighted values. These are learned neighbor weights, not a guarantee that the largest weight identifies a causal explanation of the final prediction. Degree, alternate paths and later layers also matter.

**Practice.** Does adding attention let a one-layer local GAT directly read an unconnected distant node? **Answer.** No. It reweights the allowed neighborhood. Additional layers or changed connectivity expand access, with different compute and leakage implications. Too many mixing layers can make node features indistinguishable (oversmoothing); narrow paths can also bottleneck distant information (oversquashing). These are distinct issues.

[GAT primary paper](https://arxiv.org/abs/1710.10903).

## 8. Node, link and whole-graph predictions

The same message-passing backbone can feed different tasks. A node head predicts one output per selected node. A link decoder scores a pair. A graph readout converts a variable number of node representations into one whole-graph representation before its prediction head. The data split and metric must match the task, not just the tensor shape.

For a simple link decoder, logit h_u dot h_v becomes a score after sigmoid. If h_u=[1,0] and h_v=[0.5,1], the logit is 0.5 and sigmoid is 0.6225. This formula does not establish calibration or prove that an edge exists. Negative sampling defines which nonedges are compared, and filtered candidate sets can make evaluation easier than deployment.

For whole-graph prediction, sum/mean/max pooling are invariant to node order, but retain different information. Scalar nodes [1,3] and [1,1,3,3] both have mean 2; their sums are 4 and 8. Mean loses multiplicity, while sum scales with graph size. Choose a readout compatible with whether size itself carries legitimate signal.

Keep held-out link targets outside the message-passing graph unless the benchmark explicitly permits that information. Handle reversed undirected edges consistently. Transductive access to unlabeled nodes/topology is not the same as permission to read held-out labels. Temporal tasks must not receive future edges or target-derived node features.

**Practice.** A link predictor sees its positive test edge during neighborhood aggregation. Is this harmless because the test label was removed? **Answer.** Not necessarily: the edge itself can disclose the target. Rebuild the allowed graph from the intended protocol, then repeat evaluation. Test graph-level splits on entire graphs or source groups where required.

[OGB link protocols](https://ogb.stanford.edu/docs/linkprop/) · [OGB graph protocols](https://ogb.stanford.edu/docs/graphprop/).

## 9. Graph Transformers: specify structure and connectivity

A graph Transformer uses attention-based transformations with a stated way to represent graph structure. A token sequence has order; an unordered graph needs its own structural/positional information and prediction invariances. Merely feeding node vectors to a language Transformer can discard topology or introduce an arbitrary order dependence.

Possible structural inputs include permitted adjacency, edge features, graph-distance biases or Laplacian positional encodings. A local graph-attention architecture restricts keys to neighbors; a global variant can compare many or all nodes. These are different connectivity choices, not a universal definition that all graph Transformers attend globally.

For global attention over 100 nodes, one head has 10,000 ordered score entries before masking. A local design where each node reads five keys including itself uses 500 valid query/key interactions in that illustrative fixed-degree case. A dense implementation may still allocate the full matrix; theoretical interaction count is not measured memory or runtime. Structural preprocessing also has a cost and can leak information if computed on prohibited target edges.

Node permutation should reorder node-level outputs consistently (**equivariance**); a whole-graph output commonly should remain unchanged (**invariance**). Positional encodings need care with sign/basis ambiguity and transfer between graphs. No architecture label substitutes for testing these properties and comparing a feature-only model and a simpler message-passing baseline.

**Practice.** Can a global Transformer use a held-out edge-distance matrix safely just because it has no label column? **Answer.** Only if that structural information is allowed by the task protocol. Structural features can encode the target. Choose and document the graph access boundary before training.

[A graph Transformer architecture](https://arxiv.org/abs/2012.09699). These are advanced explanations, not newly implemented graph benchmark trainers.

## Sources

- [Graph Convolutional Networks](https://arxiv.org/abs/1609.02907).
- [PyTorch Geometric message-passing API](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.MessagePassing.html), for comparing aggregation/update conventions before implementing a graph layer.
