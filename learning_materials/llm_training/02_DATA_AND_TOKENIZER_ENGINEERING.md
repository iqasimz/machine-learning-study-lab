# T2. LLM training data, tokenization and packing

A language-model data pipeline converts permitted source material into the exact token inputs and prediction targets seen during training. It includes extraction, duplicate grouping, splits, tokenization, packing and masks. Corpus descriptions alone are insufficient: the resulting batches must preserve the intended information and evaluation boundaries.

Four source records with one exact duplicate illustrate identity and grouping. A separate supplied token sequence [3,7,2,9] shows how window length changes the aligned input and target slices. Neither example downloads or prepares an external corpus.

An objective defines the intended prediction problem. The data pipeline must implement that problem without leaking evaluation examples, changing tokenizer meanings or introducing unintended cross-document targets.

## 1. Data provenance and permissions

**Key terms.** **Provenance** records where data came from; a **license** states permitted use; a **pipeline** converts sources into training inputs.

**Process.** Start with permitted sources and record their origins, then extract, clean and tokenize them. Audit the actual final batches: the model learns what reaches them, not the description of the source corpus.

Start with a data contract: source, rights/consent, intended use, language/domain, sensitive fields, extraction method, retention and deletion policy. Publicly accessible does not automatically mean licensed for every use. Hashing a private identifier does not necessarily anonymise it.

The pipeline is: approved sources; extraction; quality/language checks; sensitive-data handling; deduplication and grouping; tokenization/packing; versioned shards; batch audits.

Treat the pipeline as a chain of transformations that can each change the evidence. Keep rejection reasons and a sample of discarded material under appropriate access controls. Aggressive filtering can improve some benchmarks while removing dialects or lower-resource languages. “Quality” is a policy and measurement choice, not an objective universal scalar.

## 2. Extraction and deduplication

**Key terms.** **Extraction** recovers text from a source format; **deduplication** finds repeated material; a **near-duplicate** differs slightly but carries largely the same content.

**Process.** Extract text, remove corruption and identify duplicate groups before splitting. Preserve links to original sources so filtering decisions and suspicious examples can be traced later.

Web extraction removes navigation/boilerplate while retaining main text. PDFs add columns, tables, OCR errors and page breaks. Preserve headings, source order and identifiers where meaningful. A failed parser can silently train the model on menus or repeated footers.

**Exact deduplication** compares identical canonical content. **Near deduplication** compares overlapping shingles or representations. MinHash approximates Jaccard similarity between sets; locality-sensitive hashing proposes likely similar pairs without comparing everything. Semantic deduplication can remove paraphrases but may also merge genuinely distinct facts.

Canonicalization must be specified. Lowercasing, whitespace changes and punctuation deletion alter what “same” means. Keep the original alongside the normalized representation where justified.

## 3. Records, deterministic IDs and hashes

**Key terms.** A **record** is one structured data entry; a **deterministic ID** repeats for the same defined inputs; a **hash** is a content fingerprint, not a guarantee of truth.

**Process.** Normalize the intended identity fields, compute an ID and content fingerprint, then store both with provenance. Later validation can distinguish the same example, changed content and repeated pipeline output.

A **record** is one structured item plus its metadata. A deterministic ID is recreated from a stable rule: `source_id + version + span boundaries`. A content hash is a fingerprint computed from exact bytes or a documented canonical serialization. A cryptographic hash is not encryption and does not prove a record is true.

Example educational record:

```json
{
  "record_id": "paper17:v2:abstract",
  "source_id": "paper17",
  "source_version": "v2",
  "language": "en",
  "text": "A treatment reduced symptoms in a small trial.",
  "rights_status": "approved_for_this_exercise",
  "parser_version": "exercise-1"
}
```

Changing source text should change its content fingerprint. Changing only record order should not change per-record content fingerprints, though it changes a whole-file hash. Save the algorithm and serialization, not just an unexplained digest. Source links enable tracing; they do not guarantee that generated claims preserve the source.

## 4. Splits, contamination and mixing

**Key terms.** **Contamination** exposes evaluation material during fitting; a **mixture** combines data sources; **sampling weights** determine how often sources appear.

**Process.** Group related sources and reserve evaluation material before fitting or mixing. Choose training-source proportions next, then inspect exposures so a large repeated source does not silently dominate.

Split at the level that prevents leakage. Search for benchmark questions and answer-like variants in training data; exact string checks alone miss paraphrases. A clean contamination audit is evidence about the checks performed, not proof that no hidden overlap exists.

A data mixture specifies sampling weights by source/domain/language. A source with twice the weight is seen more often, not necessarily learned twice as well. Token budgets and document sampling create different exposures. Track repeated epochs and duplicated synthetic content. Continual/domain pretraining can forget earlier capabilities if the mix is too narrow.

Synthetic data can improve coverage and formatting but inherits teacher errors, biases and benchmark leakage. Verify provenance and licensing, sample outputs independently and maintain human/real-data evaluation. Model collapse is a risk under particular repeated synthetic-data conditions, not a theorem that all synthetic data is bad.

## 5. Packing and masks

**Key terms.** **Packing** combines sequences into efficient windows; an **attention mask** controls information flow; a **loss mask** controls which targets are scored.

**Process.** Token sequences enter windows with explicit boundaries and padding. Build attention and loss masks before the forward pass, then verify that neither cross-document leakage nor unintended target scoring has been introduced.

**Calculation.** With two independent documents of 12 and 8 tokens in a 32-position window, 20 positions contain document tokens and 12 are padding. Under shifted-target scoring with no cross-document transition, target counts must be computed per boundary; 20 input tokens do not automatically mean 20 valid next-token targets.

Packing concatenates short samples to use token slots efficiently. EOS boundaries indicate document endings; whether attention crosses a boundary is a design decision. A block-diagonal attention mask can isolate samples. An attention mask controls **what can be seen**; a loss mask controls **which predictions are scored**. They are different.

For SFT, packing must preserve response spans and role boundaries. For causal pretraining, verify no target sees itself/future tokens. Test padding on both short and full-length batches. One off-by-one error can make a misleadingly easy objective.

### Audit one packed sequence before scaling the corpus

Take two independent documents, A and B, packed into one token window. Decide the intended statistical task before choosing an efficient attention kernel.

1. Assign each token a document/group identity and original source offset. Keep a separate marker for padding.
2. If documents must be independent, allow a query to read only keys from its own document at the same or earlier position. A global lower-triangular mask alone lets B read A.
3. Decide whether the artificial transition from the end of A to the beginning of B is scored. If independence is intended, exclude that cross-document target rather than accidentally teaching a continuation that never existed.
4. Inspect position IDs and response-loss spans after packing. Resetting position IDs does not, by itself, prevent cross-document attention.
5. Test a counterfactual: change A while holding B fixed. Under an isolated-document policy, B's permitted predictions should not change in evaluation mode. Also verify that padding never adds targets or produces an all-masked invalid attention row.

Document isolation is a policy choice, not a universal requirement for all pretraining recipes. Some intentionally concatenate documents. The error is claiming isolation while the executed mask permits mixing. Follow the selected implementation's actual packing/attention support: [Transformers causal objectives](https://huggingface.co/docs/transformers/tasks/language_modeling), [TRL packing and loss masks](https://huggingface.co/docs/trl/sft_trainer), and [FineWeb's data-pipeline study](https://arxiv.org/abs/2406.17557).

## 6. Storage and reproducibility

**Key terms.** A **manifest** lists exact files and versions; a **shard** is one partition of a dataset; **checksums** detect changed bytes.

**Process.** Save processed records in bounded shards with a manifest and hashes. The loader uses those artifacts to reconstruct batches; a later run can then check that it received the same data.

Use sharded files suitable for streaming; keep a manifest of source versions, row/token counts, schema, hashes and filtering configuration. A streaming loader avoids loading the whole corpus into RAM but complicates shuffling and exact resumption. Record shard order, sampler state and offsets. Checkpointing only weights is insufficient for exact continuation.

## Exercises and checks

1. A source appears as both HTML and PDF. Where should leakage checks occur? **Before treating splits as independent**, using source identity and near-duplicate checks, not file extensions alone.
2. Can a hash validate factual accuracy? **No:** it detects content identity/change under the chosen rule.
3. Design a five-record audit containing a Unicode example, duplicate, private field, malformed page and empty extraction. Specify expected outcomes.

Case study: [FineWeb](https://arxiv.org/abs/2406.17557) documents data-pipeline decisions; [OLMo 2](https://arxiv.org/abs/2501.00656) connects data and training. Our record examples and checks are original teaching exercises, not claims about those projects' exact schemas.
