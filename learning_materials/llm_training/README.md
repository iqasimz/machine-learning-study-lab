# Stage 2 — LLM training

This stage turns the foundations into a trainable model. “Pretraining” belongs here: it is training before a particular downstream adaptation, not a synonym for the prerequisite maths.

Begin with [the complete training loop](00_THE_COMPLETE_TRAINING_LOOP.md): one plain-language walkthrough from text to predictions, gradients, updates, validation, checkpointing and generation. Then use the detailed lessons below. In the application it is the first item under **LLM training**, with additional interactive diagrams alongside the relevant sections.

1. [Language-model training objectives](01_OBJECTIVES_AND_LIFECYCLE.md)
2. [LLM training data, tokenization and packing](02_DATA_AND_TOKENIZER_ENGINEERING.md)
3. [Decoder Transformer architecture](03_TRANSFORMER_FROM_FIRST_PRINCIPLES.md)
4. [Language-model architecture families](04_ARCHITECTURES_AND_MODALITIES.md)
5. [Gradient accumulation and stable LLM training](05_STABLE_TRAINING_AND_DEBUGGING.md)
6. [LLM parameter counts, memory and compute](06_SCALING_COMPUTE_AND_MEMORY.md)
7. [Distributed LLM training and gradient reduction](07_DISTRIBUTED_TRAINING_AND_KERNELS.md)
8. [LLM evaluation and interpretability evidence](08_EVALUATION_INTERPRETABILITY_AND_RELEASE.md)

The website reader introduces each topic before its calculations and integrates selected deeper notes. The files here retain the full authored source; they are not a second lecture displayed below the diagram. The decoder view keeps all 25 calculation steps. Architecture alternatives and estimates are distinguished from an executed training run.

Do not rent a cluster to “complete” the distributed chapter. First explain the design on paper and profile a small authorised run. The stage gate is in [Assessments](../ASSESSMENTS.md).
