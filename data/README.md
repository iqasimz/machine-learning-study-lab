# First corpus

`corpus.jsonl` contains 24 newly authored synthetic documents: 16 train, 4 dev, 4 test. Released as **CC0-1.0** learning fixtures. No thesis, personal records, API output or scraped corpus is used.

These are deliberately small, related teaching statements. Splits share language and templates. Normalized exact deduplication is implemented; semantic/near-duplicate isolation is not established. Never market this as a real-world generalization benchmark.

BPE is fitted only on train. Each document has BOS/EOS; no attention or transition crosses into another document. Long documents become disjoint target chunks with context reset at chunk boundaries. Targets are the text shifted one token, not argument labels. Test evaluation occurs at run completion; choosing more runs after looking at these scores makes the test a development resource.
