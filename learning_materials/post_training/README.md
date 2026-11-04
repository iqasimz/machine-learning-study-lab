# Stage 3 — post training, systems and deployment

First change model behaviour where necessary; then build the surrounding system. A prompt, retrieved document, tool response and gradient update are four different mechanisms.

1. [Supervised fine-tuning and LoRA](01_SFT_AND_PARAMETER_EFFICIENT_ADAPTATION.md)
2. [Preference learning: DPO, PPO and GRPO](02_PREFERENCES_RLHF_DPO_PPO_AND_GRPO.md)
3. [Prompting, reasoning and output decoding](03_PROMPTING_REASONING_AND_STRUCTURED_OUTPUT.md)
4. [Retrieval-augmented generation and vector indexes](04_RAG_AND_VECTOR_DATABASES.md)
5. [Tool-using agents and MCP boundaries](05_AGENTS_TOOLS_AND_MCP.md)
6. [LLM inference, KV caching and serving](06_INFERENCE_AND_SERVING.md)
7. [LLM quantization, compression and deployment](07_COMPRESSION_EDGE_AND_DEPLOYMENT.md)
8. [LLM system evaluation, security and operations](08_PRODUCTION_EVALUATION_SECURITY_AND_OPERATIONS.md)
9. [LLM research methods and activation steering](09_ADVANCED_BRANCHES_AND_RESEARCH_JUDGMENT.md)

Each website lesson begins with a definition and a concrete example. Method comparisons use reading sections rather than an invented execution chain. Diagrams show actual dependencies: denied tool calls skip execution and cache-byte accounting sits beside generation. Detailed objectives, practical constraints and primary sources remain in the same reader's deeper notes.

Complete the [system capstone and oral defence](../ASSESSMENTS.md). Optional real models must run only on approved hardware and approved public data. No lesson requires paid API calls or uploading private documents to an external service.
