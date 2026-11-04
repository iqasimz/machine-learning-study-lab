# R4. Retrieval-augmented generation and vector indexes

Retrieval-augmented generation, or RAG, supplies external evidence to a model at use time. Ingestion builds a versioned, permission-aware source index; a query-time pipeline retrieves candidates, assembles context and generates an answer. Finding similar text and checking support are different tasks, and retrieval does not ordinarily update the generator's weights.

Allowed passages A and B have supplied scores 0.8 and 0.3. Forbidden C scores 0.95 but must remain excluded. Select one or two eligible passages, then inspect a separate 100-token budget that reserves 20 for fixed input and 30 for output.

Prompting determines what context the model receives. RAG adds a source-selection system whose permissions, coverage and freshness must be evaluated separately from how the model uses the selected evidence.

## 1. Retrieval-augmented generation

**Key terms.** **RAG** means retrieval-augmented generation; a **retriever** finds evidence candidates; a **generator** writes using supplied context.

**Process.** A question enters retrieval, permitted evidence is added to its prompt and the generator answers. Evidence changes input, not normally learned weights.

Retrieval-augmented generation supplies external evidence at inference time. A basic system retrieves relevant content, constructs a context and asks a model to answer using it. The original RAG research also studies learned retrieval/generation combinations; the term now covers a wider family of pipelines. RAG does not automatically fine-tune the model or eliminate hallucinations.

Ingestion parses, versions, chunks and indexes approved documents. At query time, user identity constrains retrieval; reranking selects candidates; a bounded prompt supplies evidence; generation produces an answer whose support and permissions must be checked.

Two pipelines exist: **ingestion** builds the knowledge store; **query-time execution** retrieves and uses it. Debug them separately.

## 2. Document chunking

**Key terms.** A **chunk** is an indexed text segment; **overlap** repeats boundary text; **metadata** records source, version and permissions.

**Process.** Extract text, preserve useful boundaries and split it into chunks. Carry provenance into the index so retrieved pieces can support citations and access checks.

Chunk by meaningful units where possible: sections, paragraphs, tables or source-linked spans. Fixed token windows are simple but can split a claim from its caveat. Overlap reduces boundary loss but duplicates evidence and increases index/context cost. Parent–child retrieval indexes smaller chunks and returns a larger parent for interpretation.

Preserve document ID, version, source location, date, access controls, language, parser version and content hash. A scientific statement without its population, comparison or uncertainty can become misleading even if every copied word is accurate.

There is no universal best chunk size. Evaluate several sensible sizes on representative questions, holding retrieval budget and generator access explicit. Tables may need structural parsing rather than plain text concatenation. OCR mistakes require confidence checks or human review where stakes justify it.

## 3. Retrieval methods and advanced branches

**Key terms.** **Sparse retrieval** uses terms; **dense retrieval** compares embeddings; a **reranker** re-scores candidates with more context; **hybrid** combines signals.

**Process.** Generate affordable candidates, merge them under a declared rule and rerank a bounded set. Only selected permitted chunks proceed to prompt construction.

**Lexical/BM25** handles exact terms; **dense** retrieval finds semantic neighbours; **learned sparse** combines sparse structure with learned expansion; **multi-vector/late interaction** retains token-level matches. Hybrid retrieval can fuse ranked lists. Cross-encoders rerank candidate pairs.

**Query expansion/rewrite** adds alternate formulations but can alter intent. **Multi-query retrieval** broadens candidate coverage at additional cost. **Hypothetical-answer retrieval** embeds a generated answer-like text, risking invented assumptions. **Multi-hop retrieval** follows evidence across documents. **GraphRAG** organizes entities/relations or communities to support relational/global questions; graph extraction adds its own errors and costs. **Corrective/self-reflective RAG** checks evidence sufficiency and retrieves again under a bounded policy.

These are optional mechanisms, not a ladder where every extra component improves quality. A small corpus may perform best with simple lexical search and a clear citation policy.

## 4. Vector records and metadata

**Key terms.** A **vector database** stores vectors and records; an **index** accelerates lookup; an **embedding revision** fixes their generating model.

**Process.** Encode chunks with a fixed model and store vectors alongside text/metadata. Encode queries compatibly; similarity search then returns records for filtering and evidence inspection.

A record typically contains ID, vector(s) and metadata/payload; it may store original text or a reference to it. The database needs insertion, update, deletion, indexing, filtering and operational durability. An approximate-nearest-neighbour library is not automatically a complete database with permissions, backups and replication.

For `n=1,000,000` vectors of width 768 at FP32, raw vectors alone use `n×768×4≈3.072 GB` decimal. Index graph/list structures, metadata, replicas and text add more. A smaller embedding can reduce storage while changing retrieval quality.

## 5. Search index families

**Key terms.** **Exact search** checks all candidates; **ANN** approximates nearest neighbours; **HNSW** uses graph links; **IVF** searches clusters; **PQ** compresses vectors.

**Process.** Build an index for the corpus and tune search effort. Compare recall with exact retrieval on a sample before accepting speed at the cost of missed evidence.

| Index | Mechanism | Trade-off |
| --- | --- | --- |
| Flat/exact | Compare query with all stored vectors | Exact under chosen metric; cost grows with corpus size |
| HNSW | Navigate a multi-level neighbour graph | Fast approximate search; memory/build cost and tuning |
| IVF | Cluster vectors; search selected inverted lists | Fewer comparisons; can miss relevant clusters |
| PQ | Compress vectors into subvector codebooks | Smaller memory; approximate distances and training needs |
| IVF-PQ | Combine list pruning and compression | Strong savings; two sources of approximation |
| Disk-oriented ANN | Place much state on storage with caching | Large collections; storage latency/access-pattern trade-offs |

For HNSW, graph connectivity, construction effort and query search effort influence recall/memory/latency. For IVF, `nlist` controls partitions and `nprobe` how many are searched. PQ splits a vector into subvectors and represents each with a codebook ID; it is not simply rounding every float to an integer.

**ANN recall** measures how closely approximate search recovers an exact top-k under the embedding metric. **Task relevance recall** measures whether the returned passages are truly useful. A perfect ANN implementation can retrieve semantically wrong evidence if the embedding space is poor.

## 6. Filters and permissions

**Key terms.** **Authorization** checks permitted access; a **tenant** is an isolated customer/group; a **filter** restricts candidates by metadata.

**Process.** Apply permissions during retrieval and preserve them through reranking and generation. A vector match must not expose a forbidden source.

Apply tenant/user/document access controls as part of retrieval design—not only after showing candidate text to a model. Pre-filtering and post-filtering have different recall/performance effects. If filtering removes all top-k candidates, retrieving only k then discarding denied items can leave an empty answer despite accessible relevant material farther down.

Filters are not a replacement for database/service authorization. Keep permissions in an authoritative policy system and test cross-tenant queries. A malicious document can contain instructions; treat retrieved content as untrusted data, not executable policy.

## 7. Freshness and deletion

**Key terms.** **Freshness** reflects current sources; **deletion propagation** removes derived copies; a **tombstone** marks an invalidated record.

**Process.** Detect changed/removed sources, update affected chunks and vectors, and invalidate caches. Retest retrieval so old answers do not survive in another store.

Updating a source requires parsing, re-chunking and re-embedding affected content, then replacing or versioning index entries. If the embedding model changes, rebuild/migrate the document space compatibly. Use an atomic collection alias/snapshot switch where supported to avoid mixed generations.

Deletion must propagate to text stores, embeddings, caches, backups under the relevant retention policy and logs where applicable. Removing a vector does not undo prior model training. Keep provenance so you can identify affected artifacts.

## 8. Context construction and generation

**Key terms.** A **context budget** limits input tokens; a **citation** points to evidence; **abstention** declines unsupported answers.

**Process.** Deduplicate permitted ranked chunks and fit evidence into the budget. Generate, then verify that each citation supports its associated claim.

Choose a token budget, remove duplicates, preserve caveats and order evidence clearly. Supply source IDs the system can resolve. Ask for claims tied to evidence and an abstention when support is missing. Verify that citations actually support the nearby claim; link existence alone is insufficient.

A useful diagnosis table:

| Observation | Likely layer to inspect |
| --- | --- |
| Relevant document absent from store | Ingestion/rights/parser |
| Present but not retrieved | Query, embedding, index, filters |
| Retrieved but removed | Reranking/context budget/deduplication |
| In context but ignored | Position, prompt, model capability |
| Quoted correctly but misrepresented | Scope, negation, interpretation |
| Correct answer leaks denied source | Authorization/cache isolation |

## 9. Evaluate RAG as a system

**Key terms.** **Retrieval recall** measures evidence recovered; **faithfulness** checks use of supplied evidence; **answer correctness** checks the final claim.

**Process.** Evaluate retrieval and generation separately, then trace end-to-end failures. Fix missing evidence or misuse of available support according to the earliest failing stage.

Build questions with answerable, unanswerable, conflicting, outdated and permission-restricted cases. Measure retrieval recall, source support, answer correctness, completeness, refusal quality, latency and cost. Compare to direct prompting, lexical-only retrieval and an oracle-evidence condition. Oracle evidence helps separate generation failure from retrieval failure, but is not a deployable result.

For a controlled educational support metric, split the output into checkable claims, label supported/contradicted/unverifiable under instructions and report denominators. Automated judges need validation against reviewed cases. Factual truth, source faithfulness and usefulness remain distinct.

### Diagnose the failure with an evidence funnel

For each evaluation question, record whether the necessary evidence survives each boundary: source store, retrieval candidates, reranking, assembled prompt and final answer.

1. If the source is absent or badly parsed, tuning the generator cannot restore the missing evidence reliably.
2. If the right chunk exists but is not retrieved, compare exact versus approximate search under the same embedding metric. This isolates index approximation from representation/relevance failure.
3. If the chunk is retrieved but excluded from the prompt, inspect reranking, deduplication, access filtering and token budgets.
4. If sufficient evidence reaches the prompt but the answer is wrong, compare an oracle-evidence condition and inspect scope, negation, distractors and citation attachment.
5. Keep answerable and genuinely unanswerable cases. A model that always abstains can have few unsupported claims while being useless; report coverage too.

Let `E` mean sufficient evidence reaches the generator and `C` mean a correct answer. Then:

\[
P(C)=P(E)P(C\mid E)+P(\neg E)P(C\mid\neg E).
\]

In an invented example with 80% evidence availability, 90% correctness when evidence is present and 10% otherwise, total correctness is `0.8×0.9+0.2×0.1=0.74`. This is conditional-probability accounting, not an independence assumption or measured benchmark. It explains why retrieval and answer quality need separate reporting. References: [RAG](https://arxiv.org/abs/2005.11401), [Faiss index choices](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) and [ColBERT's late interaction](https://arxiv.org/abs/2004.12832).

## Exercises and answers

1. Exact search retrieves the nearest vectors perfectly but factual QA is poor. Explain at least three causes: embedding mismatch, bad chunks, missing context, generator misuse or wrong query/permissions.
2. When might BM25 outperform dense retrieval? Exact identifiers, rare terms or a domain poorly represented by the embedding model.
3. Design an update for a retracted paper: version/tombstone source, remove affected chunks, invalidate caches, test stale citations, retain an auditable record under policy.
4. Why not add every retrieved chunk? Context limits, irrelevant distraction, duplicates, privacy and cost; more text is not equivalent to more usable evidence.

Sources: [original RAG](https://arxiv.org/abs/2005.11401), [Faiss index documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes), [Sentence Transformers](https://www.sbert.net/), [Qdrant filtering](https://qdrant.tech/documentation/search/filtering/). Product-specific capabilities must be checked for your deployed version; this lesson is not an endorsement of one vendor.
