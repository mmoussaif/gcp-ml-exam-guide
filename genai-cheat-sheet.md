# GenAI System Design Cheat Sheet (Complete Interview Prep)

**Purpose:** Essential GenAI system design in a 3-pager style: mental model, system shape, architectures, RAG, agents, serving, image/video/multimodal, metrics, cost, security, scalability, interview. **All gold concepts** from the full guide are covered below or in the **Glossary (Adapted)** at the end. Use the **Coverage** section to verify nothing is missing.

### Mental model (full guide core)

**Three questions:** (1) **Non-determinism** — evaluate and control probabilistic outputs. (2) **Token economics** — cost and latency scale with length. (3) **Orchestration** — combine models, retrieval, and tools.

**GenAI vs traditional ML:** Traditional ML = one input → one output (calculator). GenAI = one prompt → stream of tokens (person typing). So: variable latency, KV cache growth, per-token cost, non-determinism → need streaming, token budgets, evaluation, guardrails.

**Six challenges:** Non-determinism (E.5 eval, E.10 guardrails) | Token economics (E.7 cost, caching, routing) | Memory pressure (E.1 serving, E.8 KV cache) | Hallucinations (E.2 RAG, E.5 eval) | Orchestration (E.4 agents, tools) | Scale unpredictability (E.8 batching, capacity). Each addressed in sections below.

**Request path (full guide B.1):** User/Frontend → API Gateway (auth, rate limit) → Orchestration (RAG + agents, E.2/E.4) → LLM(s) (E.1) → Response. Cross-cutting: E.5 eval, E.6 data, E.7 cost, E.8 scale, E.9 monitoring, E.10 security.

---

## 1. CORE COMPONENTS

| Component          | What It Is                                   | Purpose                              | Key Choice                   |
| ------------------ | -------------------------------------------- | ------------------------------------ | ---------------------------- |
| **LLM**            | Large language model (GPT, Gemini, Claude)   | Text generation, reasoning, tool use | Size vs cost vs latency      |
| **Embeddings**     | Dense vector representation of text/images   | Semantic similarity, retrieval       | Dimension (768-3072), model  |
| **Vector DB**      | Database optimized for similarity search     | Store & retrieve embeddings          | FAISS, Pinecone, Vertex AI   |
| **Tokenizer**      | Splits text into tokens (BPE, SentencePiece) | Model input preprocessing            | Vocab size, subword handling |
| **Prompt**         | Instructions + context sent to LLM           | Control model behavior               | System prompt, few-shot      |
| **Context Window** | Max tokens model can process (4K-1M)         | Limits input + output length         | Cost grows with length       |
| **KV Cache**       | Stores computed attention keys/values        | Avoid recomputation                  | Grows with context length    |

## 2. ARCHITECTURES

```
DECODER-ONLY          ENCODER-ONLY           ENCODER-DECODER         MoE (Mixture of Experts)
(GPT, LLaMA, Gemini)  (BERT, RoBERTa)        (T5, BART)              (Mixtral, Gemini 1.5)
─────────────────     ─────────────────      ─────────────────       ─────────────────────
Generates text        Understands text       Transforms text         Sparse activation
Causal attention      Bidirectional          Cross-attention         Top-K experts per token
→ Chatbots, code      → Classification       → Translation           → High capacity, low cost
```

**MoE Key Insight**: 8×7B model = 56B total params, but only ~14B active per token → capacity of large, cost of small

## 3. TRAINING STAGES

```
PRETRAINING              SFT                      RLHF
(Trillions tokens)       (10K-100K pairs)         (Human preferences)
─────────────────        ─────────────────        ─────────────────
Web text, books, code    (instruction, response)  Preference rankings
Next-token prediction    Same objective           Reward model + PPO
→ Raw capability         → Follows instructions   → Aligned & safe
```

| Stage           | Data                                   | Objective                        | Output                      |
| --------------- | -------------------------------------- | -------------------------------- | --------------------------- |
| **Pretraining** | Trillions of tokens (web, books, code) | Next-token prediction            | Base model (predicts)       |
| **SFT**         | 10K-100K (prompt, response) pairs      | Next-token on instruction format | Instruction-tuned           |
| **RLHF**        | Human preference rankings              | Maximize reward model score      | Aligned (helpful, harmless) |

## 4. KEY TECHNIQUES

| Technique              | What It Does                       | When to Use                  | Trade-off                 |
| ---------------------- | ---------------------------------- | ---------------------------- | ------------------------- |
| **RAG**                | Retrieve docs → inject into prompt | Ground answers in your data  | Latency vs accuracy       |
| **Fine-tuning**        | Train model on your data           | Specialized style/domain     | Cost vs customization     |
| **Prompt Engineering** | Craft instructions for behavior    | Quick iteration, no training | Limited vs flexible       |
| **Function Calling**   | LLM outputs structured tool calls  | Connect to APIs, DBs         | Reliability vs capability |
| **Agents**             | LLM in loop: think → act → observe | Multi-step tasks, tools      | Complexity vs power       |
| **Guardrails**         | Filter input/output for safety     | Block harmful content        | Safety vs helpfulness     |
| **Grounding**          | Verify claims against sources      | Reduce hallucination         | Latency vs accuracy       |

**RAG vs fine-tune (when):** RAG when data changes often or you need citations; fine-tune when you need custom style/domain or RAG isn’t accurate enough. LoRA/PEFT for cheaper fine-tune.

## 5. RAG PIPELINE (Most Common Pattern)

```
Query → Embed → Vector Search → Rerank → Top-K chunks → LLM + context → Answer
         │           │             │                          │
    text-embedding  HNSW/IVF   Cross-encoder            Cite sources
```

| RAG Component    | Options                             | Notes                  |
| ---------------- | ----------------------------------- | ---------------------- |
| **Chunking**     | 500 tokens, 200 overlap             | RecursiveTextSplitter  |
| **Embeddings**   | text-embedding-004, CLIP            | 768-3072 dimensions    |
| **Vector Index** | HNSW (fast), IVF (memory efficient) | HNSW for <100M vectors |
| **Retrieval**    | Dense, BM25, Hybrid (RRF)           | Hybrid often best      |
| **Reranking**    | Cross-encoder (ms-marco)            | Top-20 → Top-5         |

**RAG Evaluation (RAGAS)**:
| Metric | Measures | Target |
|--------|----------|--------|
| **Faithfulness** | % claims supported by context | > 0.9 |
| **Answer Relevancy** | Response addresses question | > 0.85 |
| **Context Precision** | % retrieved chunks relevant | > 0.8 |
| **Context Recall** | % relevant chunks retrieved | > 0.8 |

## 6. AGENT PATTERNS

```
SINGLE AGENT              MULTI-AGENT               ORCHESTRATION
─────────────             ───────────               ─────────────
LLM + Tools               Specialized agents        Sequential / Parallel / Debate
ReAct loop                Handoffs between          DAG of steps
→ Most common             → Complex tasks           → Pipelines
```

**Agent = LLM in a loop**: Reason → Act (call tool) → Observe (get result) → Repeat until done

**Reasoning Frameworks**:
| Framework | How It Works | Best For |
|-----------|--------------|----------|
| **CoT (Chain-of-Thought)** | "Think step by step" in prompt | Math, logic |
| **ReAct** | Reason + Act + Observe loop | Tool use, multi-step |

**Tool Types**:
| Type | Where Executed | Example |
|------|----------------|---------|
| **Function Calling** | Client-side | API calls, DB queries |
| **Code Execution** | Agent sandbox | Python, SQL |
| **MCP Tools** | External servers | Standardized tool protocol |

## 7. SERVING & OPTIMIZATION

| Concept                  | What It Is                            | Why It Matters                          |
| ------------------------ | ------------------------------------- | --------------------------------------- |
| **KV Cache**             | Stores computed attention keys/values | Avoid recomputation; grows with context |
| **Continuous Batching**  | Add new requests to running batch     | Better GPU utilization                  |
| **Quantization**         | Reduce precision (FP16→INT8→INT4)     | Faster, smaller, slight quality loss    |
| **Model Routing**        | Small model for easy, large for hard  | Cost optimization                       |
| **Caching**              | Response/semantic/KV cache            | Reduce redundant LLM calls              |
| **Tensor Parallelism**   | Split layers across GPUs              | Serve models > 1 GPU memory             |
| **Pipeline Parallelism** | Split layers sequentially             | Very large models                       |

**Quantization Levels**:
| Level | Memory | Speed | Quality |
|-------|--------|-------|---------|
| FP32 | 1× | 1× | Best |
| FP16/BF16 | 0.5× | 1.5× | ~Same |
| INT8 | 0.25× | 2× | 1-2% loss |
| INT4 | 0.125× | 3× | 3-5% loss |

## 8. IMAGE GENERATION

### Diffusion Models (DALL-E, Stable Diffusion, Imagen)

```
TRAINING:   Image → Add noise (T steps) → Noisy image
            Model learns to predict/remove noise at each step

INFERENCE:  Pure noise → Denoise (T steps) → Generated image
                              ↑
                    Conditioned on text (CLIP/T5 embeddings)
```

| Component                          | Purpose                                        |
| ---------------------------------- | ---------------------------------------------- |
| **Text Encoder**                   | CLIP or T5 → text embeddings                   |
| **U-Net / DiT**                    | Predicts noise to remove                       |
| **DDIM Sampler**                   | Faster sampling (20-50 steps vs 1000)          |
| **CFG (Classifier-Free Guidance)** | Balance text adherence vs diversity (w=7-15)   |
| **VAE**                            | Latent diffusion: compress 512× for efficiency |
| **Super-Resolution**               | Upscale 64×64 → 256×256 → 1024×1024            |

**Latent vs Pixel Diffusion**:
| Approach | Training | Quality | Examples |
|----------|----------|---------|----------|
| **Latent** | 512× faster | Good (VAE may lose details) | Stable Diffusion |
| **Pixel** | Expensive | Highest | Imagen |

### GANs (StyleGAN for faces)

```
Generator (G): Noise → Fake image
Discriminator (D): Real or fake?

Training: G tries to fool D, D tries to catch G → adversarial
```

| GAN Concept        | What It Is                           |
| ------------------ | ------------------------------------ |
| **Mode Collapse**  | G produces limited variety           |
| **Truncation (ψ)** | Trade diversity for quality          |
| **Latent Space**   | 512-dim noise vector → style control |

## 9. VIDEO GENERATION (Sora, Movie Gen)

```
Text → DiT with Temporal Layers → Low-res video → Spatial SR → Temporal SR → Final
              ↑
    3D patches + temporal attention for consistency
```

| Video Component               | Purpose                                  |
| ----------------------------- | ---------------------------------------- |
| **Temporal Attention**        | Consistency across frames                |
| **Temporal Convolution**      | Local motion patterns                    |
| **VAE Compression**           | 8× temporal + 8×8 spatial = 512× smaller |
| **Spatial Super-Resolution**  | 160×90 → 1280×720                        |
| **Temporal Super-Resolution** | 8fps → 24fps                             |

## 10. MULTIMODAL (Vision-Language)

```
IMAGE CAPTIONING:
Image → ViT/CLIP Encoder → Patch embeddings → Cross-attention → Text Decoder → Caption

VQA (Visual Q&A):
Image + Question → Encoder + Cross-attention → Decoder → Answer
```

| Model             | Architecture                    | Use Case        |
| ----------------- | ------------------------------- | --------------- |
| **BLIP-2**        | Frozen encoder + Q-Former + LLM | Captioning, VQA |
| **LLaVA**         | ViT + LLaMA                     | Open-source VQA |
| **Gemini Vision** | Multimodal encoder + decoder    | Everything      |

## 11. ALL METRICS YOU NEED TO KNOW

### Text Generation

| Metric         | Measures                      | Higher/Lower Better |
| -------------- | ----------------------------- | ------------------- |
| **Perplexity** | Model uncertainty             | Lower               |
| **BLEU**       | N-gram precision vs reference | Higher              |
| **ROUGE**      | Recall vs reference           | Higher              |
| **METEOR**     | Semantic similarity           | Higher              |

### RAG Quality

| Metric                | Measures                    | Target |
| --------------------- | --------------------------- | ------ |
| **Faithfulness**      | Claims supported by context | > 0.9  |
| **Answer Relevancy**  | Addresses the question      | > 0.85 |
| **Context Precision** | Retrieved chunks relevant   | > 0.8  |
| **Context Recall**    | Relevant chunks retrieved   | > 0.8  |

### Retrieval

| Metric          | Measures                              |
| --------------- | ------------------------------------- |
| **Precision@K** | % of top-K that are relevant          |
| **Recall@K**    | % of relevant in top-K                |
| **MRR**         | Rank of first relevant result         |
| **NDCG**        | Ranking quality with graded relevance |
| **Hit Rate**    | % queries with ≥1 relevant in top-K   |

### Image Generation

| Metric                   | Measures                               | Target |
| ------------------------ | -------------------------------------- | ------ |
| **FID**                  | Distribution similarity to real images | < 10   |
| **Inception Score (IS)** | Quality × diversity                    | > 10   |
| **CLIPScore**            | Text-image alignment                   | > 0.3  |
| **LPIPS**                | Perceptual similarity                  | Lower  |

### Video Generation

| Metric                   | Measures                              | Target      |
| ------------------------ | ------------------------------------- | ----------- |
| **FVD**                  | Fréchet Video Distance (I3D features) | < 300       |
| **FID (per-frame)**      | Average frame quality                 | < 15        |
| **Temporal Consistency** | Smoothness across frames              | Human > 4/5 |

### Captioning

| Metric     | Measures                           |
| ---------- | ---------------------------------- |
| **CIDEr**  | Consensus with multiple references |
| **BLEU-4** | 4-gram precision                   |
| **METEOR** | Semantic similarity                |

### LLM Benchmarks

| Benchmark      | Tests                             |
| -------------- | --------------------------------- |
| **MMLU**       | Multitask knowledge (57 subjects) |
| **HumanEval**  | Code generation                   |
| **GSM8K**      | Math reasoning                    |
| **TruthfulQA** | Factual accuracy                  |
| **HellaSwag**  | Commonsense reasoning             |

### Safety

| Metric                | Target |
| --------------------- | ------ |
| **Toxicity Rate**     | < 0.1% |
| **PII Leak Rate**     | 0%     |
| **Jailbreak Success** | < 1%   |

### Performance

| Metric                  | What It Measures                 |
| ----------------------- | -------------------------------- |
| **TTFT**                | Time to first token              |
| **Latency P50/P95/P99** | Response time distribution       |
| **Throughput**          | Requests/second or tokens/second |

## 12. COST FORMULA

```
Cost = (Input tokens × input price) + (Output tokens × output price)

Example: 2K input + 500 output @ Gemini Pro ($0.50/1M in, $1.50/1M out)
       = 2000 × 0.0000005 + 500 × 0.0000015 = $0.00175/request
```

**Cost Optimization Levers**:
| Lever | Savings | How |
|-------|---------|-----|
| **Response Cache** | 30-50% | Cache exact query matches |
| **Semantic Cache** | 20-30% | Cache similar queries |
| **Model Routing** | 40-60% | Small model for easy queries |
| **Prompt Optimization** | 10-30% | Shorter prompts |
| **Quantization** | 20-40% | INT8/INT4 models |

## 13. SECURITY & GUARDRAILS

### Threat Model

| Attack                        | Vector                         | Defense                          |
| ----------------------------- | ------------------------------ | -------------------------------- |
| **Direct Prompt Injection**   | User input                     | Input filter, spotlighting       |
| **Indirect Prompt Injection** | Retrieved docs contain attacks | Sanitize, instruction hierarchy  |
| **Jailbreaking**              | Trick model to bypass rules    | RLHF, guardrails                 |
| **Data Leakage**              | Model reveals training data    | Differential privacy, guardrails |
| **Tool Abuse**                | Agent misuses APIs             | Least privilege, validation      |

### Defense Layers

```
HTTP Layer → Auth → INPUT GUARDRAILS → LLM → OUTPUT GUARDRAILS → Response
                         ↑                          ↑
                   Injection detection         PII filter
                   Blocklist                   Toxicity check
                   Spotlighting                Hallucination check
```

**Spotlighting**: Mark external content → `<<DATA>>user docs here<</DATA>>` → model treats as data, not instructions. **Indirect injection**: Attacker puts instructions inside retrieved docs; defend with sanitization and instruction hierarchy (system prompt overrides doc content).

### Platform Services

| Purpose           | Google Cloud   | AWS                |
| ----------------- | -------------- | ------------------ |
| **Guardrails**    | Model Armor    | Bedrock Guardrails |
| **WAF**           | Cloud Armor    | AWS WAF            |
| **PII Detection** | Cloud DLP      | Amazon Macie       |
| **Secrets**       | Secret Manager | Secrets Manager    |

## 14. SCALABILITY PATTERNS

### GPU Quick Reference

| GPU         | Memory   | FP16 TFLOPS | Cost      | Best For                        |
| ----------- | -------- | ----------- | --------- | ------------------------------- |
| **V100**    | 16/32 GB | 125         | ~$2/hr    | Legacy, small models            |
| **A100**    | 40/80 GB | 312         | ~$4/hr    | Production standard             |
| **H100**    | 80 GB    | 990         | ~$8/hr    | Large training, high throughput |
| **L4**      | 24 GB    | 121         | ~$0.80/hr | Cost-effective inference        |
| **TPU v5e** | 16 GB    | —           | ~$1.20/hr | Google Cloud                    |

**Model → GPU sizing**: 7B = 14GB (1× L4) | 70B = 140GB (2× H100) | 405B = 810GB (8× H100)

### Inference Scaling

| Pattern                  | What It Does             | When              |
| ------------------------ | ------------------------ | ----------------- |
| **Horizontal**           | More replicas            | Stateless serving |
| **Tensor Parallelism**   | Split layers across GPUs | Model > 1 GPU     |
| **Pipeline Parallelism** | Sequential layer split   | Very large models |
| **Continuous Batching**  | Dynamic batch management | High throughput   |

### Training Scaling

| Pattern                      | What It Does                              |
| ---------------------------- | ----------------------------------------- |
| **Data Parallelism**         | Same model on each GPU, different data    |
| **Model/Tensor Parallelism** | Split model across GPUs                   |
| **Pipeline Parallelism**     | Split layers sequentially                 |
| **ZeRO/FSDP**                | Shard optimizer states, gradients, params |
| **Gradient Checkpointing**   | Trade compute for memory                  |
| **Mixed Precision (AMP)**    | FP16 forward, FP32 gradients              |

**ZeRO Levels**:
| Level | Shards | Memory Reduction |
|-------|--------|------------------|
| ZeRO-1 | Optimizer states | ~4× |
| ZeRO-2 | + Gradients | ~8× |
| ZeRO-3 | + Parameters | ~N× (N = GPUs) |

## 15. INTERVIEW FRAMEWORK (45 min)

| Phase               | Time      | What to Cover                                |
| ------------------- | --------- | -------------------------------------------- |
| **1. Requirements** | 5-10 min  | Token budget, latency, quality, cost, safety |
| **2. Architecture** | 10-15 min | Draw: API → Orchestration → LLM → Response   |
| **3. Deep Dive**    | 15-20 min | RAG, model choice, eval, security            |
| **4. Trade-offs**   | 5-10 min  | Quality vs cost, latency vs throughput       |

**Always do**: Back-of-envelope (tokens × price, latency breakdown)

## 16. QUICK DECISION GUIDE

| Need                 | Solution                                     |
| -------------------- | -------------------------------------------- |
| Ground in docs       | **RAG** (chunk + vector DB + rerank)         |
| Real-time tools      | **Function calling** / **Agents**            |
| Custom style         | **Fine-tuning** (SFT)                        |
| Ultra-low latency    | **Small model** + **edge** + **cache**       |
| Reduce hallucination | **RAG** + **grounding** + **citations**      |
| Long documents       | **Chunking** or **long-context model**       |
| Multi-step reasoning | **Agents** (ReAct)                           |
| Cost at scale        | **Routing** + **caching** + **quantization** |
| Generate images      | **Diffusion** (Stable Diffusion, Imagen)     |
| Generate video       | **DiT + temporal layers** (Sora)             |
| Generate faces       | **StyleGAN**                                 |

## 17. PLATFORM COMPARISON

|                | Google Cloud               | AWS                       |
| -------------- | -------------------------- | ------------------------- |
| **Models**     | Gemini (Vertex AI)         | Claude, Titan (Bedrock)   |
| **RAG**        | Vertex RAG Engine          | Bedrock Knowledge Bases   |
| **Agents**     | ADK, Conversational Agents | AgentCore, Bedrock Agents |
| **Vector DB**  | Vertex AI Vector Search    | OpenSearch, Pinecone      |
| **Guardrails** | Model Armor                | Bedrock Guardrails        |
| **Image Gen**  | Imagen 3                   | Titan Image, SD           |
| **Video Gen**  | Veo                        | -                         |

## 17b. TABLE OF TOOLS (by category)

Quick lookup: which tool to use for what. Aligned with full guide (E.1–E.10, D.2, F.1).

| Category                  | Tool / stack                                   | Purpose                                            | When to use                                 |
| ------------------------- | ---------------------------------------------- | -------------------------------------------------- | ------------------------------------------- |
| **LLM serving**           | **vLLM**                                       | OSS inference; PagedAttention, continuous batching | Self-hosted, high throughput, open models   |
|                           | **TGI** (Text Generation Inference)            | Hugging Face serving; similar to vLLM              | HF ecosystem, Inference Endpoints           |
|                           | **TensorRT-LLM**                               | NVIDIA-optimized inference                         | Max perf on NVIDIA GPUs                     |
|                           | **Vertex AI / Bedrock**                        | Managed model hosting                              | No infra; enterprise; Gemini/Claude         |
| **Orchestration**         | **LangChain**                                  | Chains, agents, tools, RAG helpers                 | Rapid prototyping; broad ecosystem          |
|                           | **LlamaIndex**                                 | RAG-focused; data loaders, indices                 | Document-heavy RAG, diverse data sources    |
|                           | **ADK** (Agent Development Kit)                | Google agent framework; workflows, multi-agent     | GCP; production agents; Vertex Agent Engine |
| **RAG / vector**          | **Vertex AI RAG Engine**                       | Managed RAG (ingest, index, query)                 | GCP; minimal RAG plumbing                   |
|                           | **Bedrock Knowledge Bases**                    | Managed RAG on AWS                                 | AWS; same idea as Vertex RAG                |
|                           | **FAISS**                                      | OSS vector search (HNSW, IVF)                      | Self-hosted; library inside your app        |
|                           | **Pinecone / Weaviate / Qdrant**               | Managed or OSS vector DBs                          | Production vector search; scale             |
|                           | **Vertex AI Vector Search**                    | Managed vector index (GCP)                         | GCP; managed HNSW                           |
|                           | **pgvector**                                   | Postgres extension for vectors                     | Already on Postgres; smaller scale          |
| **Embeddings**            | **text-embedding-004** (Vertex)                | Google embeddings                                  | GCP RAG                                     |
|                           | **OpenAI text-embedding-3**                    | OpenAI embeddings                                  | When using OpenAI models                    |
|                           | **e5 / BGE** (OSS)                             | Open-source embed models                           | Self-hosted; no API cost                    |
| **Document parsing**      | **Document AI** (GCP)                          | PDF/docs → text, tables, structure                 | GCP; high-quality parsing                   |
|                           | **Textract** (AWS)                             | Same idea on AWS                                   | AWS                                         |
|                           | **PyMuPDF / pdfplumber**                       | Rule-based PDF extraction                          | Simple, consistent layouts                  |
| **Chunking**              | **RecursiveCharacterTextSplitter** (LangChain) | Split by separators, token-aware                   | Default in many RAG tutorials               |
| **Guardrails**            | **Model Armor** (GCP)                          | Input/output safety; injection, jailbreak          | Vertex AI apps                              |
|                           | **Bedrock Guardrails**                         | Same on AWS                                        | Bedrock apps                                |
| **Evaluation**            | **RAGAS**                                      | RAG metrics (faithfulness, relevancy, context)     | RAG quality without gold answers            |
| **Dev / experimentation** | **Google AI Studio**                           | Play with Gemini; quick prototypes                 | Learning; no production                     |
|                           | **Vertex AI Studio**                           | Enterprise; fine-tune, eval, deploy                | Move from AI Studio to production           |

**Stack snapshots (full guide):** Real-world examples in F.1 combine these (e.g. Document AI + LangChain chunking + Vertex Vector Search + Gemini + RAGAS; or vLLM on GKE + LangChain + Pinecone).

## 18. MODEL QUICK REFERENCE

| Model                | Type             | Params                 | Notes              |
| -------------------- | ---------------- | ---------------------- | ------------------ |
| **GPT-4o**           | Decoder-only     | ~1.7T (rumored MoE)    | Multimodal, SOTA   |
| **Gemini 1.5 Pro**   | MoE              | Undisclosed            | 1M context         |
| **Claude 3.5**       | Decoder-only     | Undisclosed            | Strong reasoning   |
| **LLaMA 3**          | Decoder-only     | 8B-405B                | Open-source        |
| **Mixtral 8x22B**    | MoE              | 141B total, 39B active | Open-source        |
| **Stable Diffusion** | Latent Diffusion | ~1B                    | Open-source images |
| **Sora**             | DiT + temporal   | Undisclosed            | Video generation   |

## 19. RED FLAGS IN INTERVIEWS

| Red Flag              | Better Answer                 |
| --------------------- | ----------------------------- |
| "No RAG needed"       | How prevent hallucination?    |
| "One model for all"   | What about cost optimization? |
| "Skip guardrails"     | Prompt injection? PII?        |
| "No eval plan"        | How know it's working?        |
| "Unbounded context"   | Cost and latency?             |
| "Train from scratch"  | Why not fine-tune?            |
| "Real-time video gen" | Minutes is realistic          |

---

## COVERAGE: WHY NO GOLD CONCEPT IS MISSING

Every major concept from the full _system-design-genai_ guide is either in the body above or in the **Glossary (Adapted)** below. This table maps full-guide structure → key concepts → where they appear here.

| Full guide               | Key concepts                                                                          | In this doc                                                 |
| ------------------------ | ------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **A.1 Intro**            | Mental model (3 questions), GenAI vs ML, six challenges, what’s in the guide          | Intro + §1–19, Glossary                                     |
| **B.1 Big picture**      | Request path: User → Gateway → Orchestration → LLM → Response; E.1–E.10 cross-cutting | Intro (mental model), §4 (techniques), §16 (decision guide) |
| **B.2 GenAI vs ML**      | Calculator vs person typing; latency, memory, cost, control; design impact            | Intro (GenAI vs ML), §1 (KV cache, context), §7 (serving)   |
| **Part C (algorithms)**  | VAE, GAN, Diffusion, Autoregressive; text-to-image, video, multimodal                 | §8 (image), §9 (video), §10 (multimodal), Glossary          |
| **Part D (LLM)**         | Tokenization, architectures, pretraining/SFT/RLHF, sampling, eval metrics             | §1–3, §11 (metrics), Glossary                               |
| **E.1 Serving**          | KV cache, batching, vLLM, parallelism                                                 | §1, §7, §14, Glossary                                       |
| **E.2 RAG**              | Chunking, embeddings, vector DB, HNSW/IVF, rerank, RAGAS                              | §5, §11 (RAG metrics), Glossary                             |
| **E.3 RAG vs fine-tune** | When RAG vs when fine-tune; LoRA, PEFT                                                | §4, §16, Glossary                                           |
| **E.4 Agents**           | ReAct, tools, multi-agent, ADK, MCP                                                   | §6, §17 (platform), Glossary                                |
| **E.5 Evaluation**       | RAGAS, LLM-as-judge, human eval, A/B                                                  | §5 (RAGAS), §11 (all metrics), Glossary                     |
| **E.6 Data pipeline**    | Events, labeling, feedback loops                                                      | §4 (techniques), Glossary (implicit in fine-tune/RAG)       |
| **E.7 Cost**             | Per-token pricing, routing, caching                                                   | §12, §16, Glossary                                          |
| **E.8 Scalability**      | Batching, parallelism, ZeRO/FSDP, GPU sizing                                          | §7, §14, Glossary                                           |
| **E.9 Monitoring**       | TTFT, latency, throughput, drift                                                      | §11 (performance), Glossary                                 |
| **E.10 Security**        | Guardrails, injection, jailbreak, spotlighting                                        | §13, Glossary                                               |
| **Part F (examples)**    | 11 system designs (Copilot, chatbot, RAG, etc.)                                       | §16 (decision guide), §17 (platform)                        |
| **Part G (interview)**   | 45-min framework, trade-offs, red flags                                               | §15, §19                                                    |

**Glossary role:** Terms that appear only in the full guide’s deep dives (e.g. PagedAttention, Triton, DPO, nprobe, CIDEr, RoPE, bi-encoder vs cross-encoder) are in **Glossary (Adapted)** so meaning and “why it matters” are still available here without re-reading the long doc.

---

## GLOSSARY (ADAPTED)

Short definitions for terms used in the full guide. **Why** = why it matters for system design.

### Fundamentals & core

| Term                        | Definition                                                                                  | Why                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **AI / ML / GenAI**         | AI = broad field. ML = learn from data. GenAI = create content (text, image, video).        | GenAI = LLMs + diffusion + agents; design for generation, not just classification. |
| **LLM**                     | Large language model; predicts next token; billions of params (GPT, Gemini, Claude, LLaMA). | Core of most GenAI systems; cost and latency are token-driven.                     |
| **Token**                   | Smallest unit of text (~4 chars, ~0.75 words). Models charge and limit by tokens.           | Cost, context window, and latency all scale with tokens.                           |
| **Context window**          | Max tokens in one request (prompt + response). 4K–2M depending on model.                    | Limits how much you can retrieve and inject; larger = more cost/latency.           |
| **Inference**               | Running a trained model (prompt → response).                                                | Most cost is inference, not training.                                              |
| **Attention / Transformer** | Mechanism to focus on relevant input; architecture behind LLMs.                             | Enables long context and reasoning; KV cache is the main memory cost.              |
| **Encoder / Decoder**       | Encoder = understand (BERT). Decoder = generate (GPT). Encoder-decoder = transform (T5).    | Decoder-only for chat/code; encoder-decoder for translation/summarization.         |

### Tokens & generation

| Term               | Definition                                                                   | Why                                                             |
| ------------------ | ---------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Tokenization**   | Text → tokens (BPE, SentencePiece, WordPiece).                               | Same text can be different token count per model; affects cost. |
| **Autoregressive** | Generate one token at a time; each depends on previous.                      | Latency scales with output length.                              |
| **Temperature**    | Randomness in token choice. 0 = deterministic; 1 = sample; >1 = more random. | Low for factual, high for creative.                             |
| **Top-p / Top-k**  | Restrict candidate tokens by cumulative probability (top-p) or top k.        | Control diversity and weird outputs.                            |
| **Sampling**       | How next token is chosen: greedy vs random (with temp/top-p/top-k).          | Greedy = reproducible; sampling = varied.                       |

### Memory & caching

| Term               | Definition                                                         | Why                                                                  |
| ------------------ | ------------------------------------------------------------------ | -------------------------------------------------------------------- |
| **KV cache**       | Stores attention keys/values for previous tokens; avoid recompute. | Without it, generation is O(N²); with it, memory grows with context. |
| **PagedAttention** | KV cache in non-contiguous pages (vLLM).                           | Less fragmentation, 2–4× more throughput.                            |
| **Semantic cache** | Cache responses by embedding similarity.                           | Cuts cost 20–50% for similar queries; risk of stale/wrong answers.   |
| **Prompt cache**   | Cache KV for common prefixes (e.g. system prompt).                 | Saves compute and TTFT.                                              |

### RAG

| Term                           | Definition                                                                        | Why                                                    |
| ------------------------------ | --------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **RAG**                        | Retrieve docs → inject into prompt → generate.                                    | Ground answers in your data without fine-tuning.       |
| **Embedding**                  | Dense vector for semantic meaning (768–1536 dims).                                | Similar text → close vectors; enables semantic search. |
| **Vector DB**                  | Store embeddings; nearest-neighbor search (Pinecone, FAISS, Vertex).              | Needed for retrieval at scale; use HNSW or IVF.        |
| **Chunking**                   | Split docs into 200–1000 token chunks; overlap 50–100.                            | Too small = no context; too large = noise.             |
| **HNSW / IVF**                 | HNSW = graph ANN, best recall/speed. IVF = cluster, search few clusters (nprobe). | HNSW default; IVF when memory-constrained.             |
| **nprobe**                     | In IVF: how many clusters to search per query.                                    | Higher = better recall, slower. Typical 10–50.         |
| **Reranking**                  | After retrieval: cross-encoder re-scores top-K (e.g. 20→5).                       | More accurate than embedding-only; slower.             |
| **Bi-encoder / Cross-encoder** | Bi: query and doc embedded separately, compare. Cross: query+doc in one pass.     | Bi for retrieval; cross for reranking.                 |
| **Hybrid search**              | Vector + keyword (BM25); merge with RRF.                                          | Better than dense or keyword alone.                    |
| **BM25**                       | Keyword ranking by term frequency.                                                | Good for exact matches; combine with vector.           |
| **Grounding**                  | Tie model answers to retrieved sources; cite.                                     | Reduces hallucination; verifiable.                     |
| **RAGAS**                      | RAG eval: Faithfulness, Answer Relevancy, Context Precision/Recall.               | Standard RAG quality metrics.                          |

### Fine-tuning

| Term                  | Definition                                                         | Why                                                              |
| --------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------- |
| **Fine-tuning / SFT** | Train on (input, output) pairs; model learns your task/style.      | When RAG isn’t enough; need 100s–1000s of examples.              |
| **LoRA / QLoRA**      | Train small adapters (rank 8–64); base frozen. QLoRA = 4-bit base. | 10–100× cheaper than full fine-tune; swap adapters at inference. |
| **PEFT**              | Parameter-efficient fine-tuning (LoRA, adapters, prefix tuning).   | Same idea: train a small part.                                   |
| **RLHF**              | Reward model from human preferences; RL to maximize reward.        | How chat models get “helpful, harmless.”                         |
| **DPO**               | Direct preference optimization; no separate reward model.          | Simpler than RLHF for alignment.                                 |

### Agents & tools

| Term                 | Definition                                                            | Why                                    |
| -------------------- | --------------------------------------------------------------------- | -------------------------------------- |
| **Agent**            | LLM in loop: perceive → decide → act (tool) → observe → repeat.       | Multi-step tasks, APIs, DBs, code.     |
| **ReAct**            | Reason + Act + Observe; interleave reasoning and tool calls.          | Default pattern for tool-using agents. |
| **CoT**              | Chain-of-thought: “think step by step” in the prompt.                 | Improves math/reasoning.               |
| **Function calling** | LLM outputs structured tool/API call; system executes and returns.    | How agents talk to external systems.   |
| **ADK**              | Agent Development Kit (Google); workflows, tools, multi-agent.        | Build agents on GCP.                   |
| **MCP**              | Model Context Protocol; standard for tools and context.               | Interop between agents and tools.      |
| **Orchestration**    | Layer that runs LLM, retrieval, tools, control flow (LangChain, ADK). | Glue between your app and models.      |

### Serving & scaling

| Term                              | Definition                                             | Why                                                     |
| --------------------------------- | ------------------------------------------------------ | ------------------------------------------------------- |
| **TTFT**                          | Time to first token.                                   | Drives perceived latency; optimize for interactive use. |
| **Continuous batching**           | New requests join running batch as slots free.         | Much better GPU use than static batching.               |
| **Quantization**                  | FP16→INT8→INT4; smaller, faster, small quality loss.   | Deploy large models on limited hardware.                |
| **Tensor / Pipeline parallelism** | Split model across GPUs (per layer vs per stage).      | When model doesn’t fit on one GPU.                      |
| **vLLM / TGI**                    | OSS LLM serving; PagedAttention, continuous batching.  | Production inference.                                   |
| **ZeRO / FSDP**                   | Shard optimizer, grads, params across GPUs (training). | Train 70B+ models.                                      |
| **Speculative decoding**          | Draft model proposes tokens; big model verifies.       | 2–3× speedup when draft is good.                        |

### Image & video

| Term                       | Definition                                                              | Why                                                  |
| -------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------- |
| **Diffusion**              | Learn to denoise; generate by iteratively denoising from noise.         | Dominant for text-to-image (SD, DALL·E, Imagen).     |
| **Latent diffusion (LDM)** | Diffuse in compressed latent space; VAE encode/decode.                  | 64× cheaper; Stable Diffusion.                       |
| **U-Net / DiT**            | U-Net = hourglass denoiser. DiT = Transformer denoiser (Sora).          | DiT scales better for video.                         |
| **VAE**                    | Encoder/decoder for images; compress to latent.                         | Used inside latent diffusion.                        |
| **CFG**                    | Classifier-free guidance; amplify effect of text conditioning (w=7–15). | Better prompt adherence.                             |
| **DDIM**                   | Few-step sampling (20–50 vs 1000).                                      | Production image gen.                                |
| **FID / CLIPScore**        | FID = distribution match to real images. CLIPScore = text-image match.  | FID = quality; CLIPScore = alignment.                |
| **CIDEr**                  | Caption metric: consensus over multiple references (TF-IDF n-grams).    | Best correlation with human judgment for captioning. |
| **FVD / I3D**              | FVD = video quality (frames + motion); I3D = network used inside.       | Main metric for video gen.                           |

### Evaluation & safety

| Term                         | Definition                                                            | Why                                   |
| ---------------------------- | --------------------------------------------------------------------- | ------------------------------------- |
| **Hallucination**            | Model states false things confidently.                                | RAG, grounding, guardrails reduce it. |
| **Faithfulness / Relevancy** | Faithfulness = claims from context. Relevancy = answers the question. | Both needed for RAG eval.             |
| **LLM-as-judge**             | Use an LLM to score another LLM’s output.                             | Scalable eval; use strong model.      |
| **Guardrails**               | Input/output filters: toxicity, PII, jailbreak, policy.               | Required in production.               |
| **Spotlighting**             | Mark external content so model treats it as data, not instructions.   | Fights prompt injection.              |
| **Model Armor**              | Google Cloud guardrail service.                                       | Managed guardrails on GCP.            |

### Infrastructure & cost

| Term                    | Definition                                                 | Why                                   |
| ----------------------- | ---------------------------------------------------------- | ------------------------------------- |
| **Vertex AI / Bedrock** | GCP / AWS managed GenAI (models, RAG, agents, guardrails). | Deploy without building everything.   |
| **Per-token pricing**   | Charge by input + output tokens; output often 3–4× input.  | Design for token budgets and caching. |
| **Token budget**        | Max tokens per request or per user/session.                | Prevents runaway cost.                |
| **Model routing**       | Easy query → small/cheap model; hard → large.              | 40–60% cost savings.                  |

---

**Core Formula**: GenAI design = Traditional system design + Token economics + Quality evaluation + Safety

**Remember**: Always discuss metrics, always do back-of-envelope, always mention trade-offs
