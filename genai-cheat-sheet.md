# GenAI System Design Cheat Sheet (Complete Interview Prep)

## 1. CORE COMPONENTS

| Component | What It Is | Purpose | Key Choice |
|-----------|------------|---------|------------|
| **LLM** | Large language model (GPT, Gemini, Claude) | Text generation, reasoning, tool use | Size vs cost vs latency |
| **Embeddings** | Dense vector representation of text/images | Semantic similarity, retrieval | Dimension (768-3072), model |
| **Vector DB** | Database optimized for similarity search | Store & retrieve embeddings | FAISS, Pinecone, Vertex AI |
| **Tokenizer** | Splits text into tokens (BPE, SentencePiece) | Model input preprocessing | Vocab size, subword handling |
| **Prompt** | Instructions + context sent to LLM | Control model behavior | System prompt, few-shot |
| **Context Window** | Max tokens model can process (4K-1M) | Limits input + output length | Cost grows with length |
| **KV Cache** | Stores computed attention keys/values | Avoid recomputation | Grows with context length |

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

| Stage | Data | Objective | Output |
|-------|------|-----------|--------|
| **Pretraining** | Trillions of tokens (web, books, code) | Next-token prediction | Base model (predicts) |
| **SFT** | 10K-100K (prompt, response) pairs | Next-token on instruction format | Instruction-tuned |
| **RLHF** | Human preference rankings | Maximize reward model score | Aligned (helpful, harmless) |

## 4. KEY TECHNIQUES

| Technique | What It Does | When to Use | Trade-off |
|-----------|--------------|-------------|-----------|
| **RAG** | Retrieve docs → inject into prompt | Ground answers in your data | Latency vs accuracy |
| **Fine-tuning** | Train model on your data | Specialized style/domain | Cost vs customization |
| **Prompt Engineering** | Craft instructions for behavior | Quick iteration, no training | Limited vs flexible |
| **Function Calling** | LLM outputs structured tool calls | Connect to APIs, DBs | Reliability vs capability |
| **Agents** | LLM in loop: think → act → observe | Multi-step tasks, tools | Complexity vs power |
| **Guardrails** | Filter input/output for safety | Block harmful content | Safety vs helpfulness |
| **Grounding** | Verify claims against sources | Reduce hallucination | Latency vs accuracy |

## 5. RAG PIPELINE (Most Common Pattern)

```
Query → Embed → Vector Search → Rerank → Top-K chunks → LLM + context → Answer
         │           │             │                          │
    text-embedding  HNSW/IVF   Cross-encoder            Cite sources
```

| RAG Component | Options | Notes |
|---------------|---------|-------|
| **Chunking** | 500 tokens, 200 overlap | RecursiveTextSplitter |
| **Embeddings** | text-embedding-004, CLIP | 768-3072 dimensions |
| **Vector Index** | HNSW (fast), IVF (memory efficient) | HNSW for <100M vectors |
| **Retrieval** | Dense, BM25, Hybrid (RRF) | Hybrid often best |
| **Reranking** | Cross-encoder (ms-marco) | Top-20 → Top-5 |

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

| Concept | What It Is | Why It Matters |
|---------|------------|----------------|
| **KV Cache** | Stores computed attention keys/values | Avoid recomputation; grows with context |
| **Continuous Batching** | Add new requests to running batch | Better GPU utilization |
| **Quantization** | Reduce precision (FP16→INT8→INT4) | Faster, smaller, slight quality loss |
| **Model Routing** | Small model for easy, large for hard | Cost optimization |
| **Caching** | Response/semantic/KV cache | Reduce redundant LLM calls |
| **Tensor Parallelism** | Split layers across GPUs | Serve models > 1 GPU memory |
| **Pipeline Parallelism** | Split layers sequentially | Very large models |

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

| Component | Purpose |
|-----------|---------|
| **Text Encoder** | CLIP or T5 → text embeddings |
| **U-Net / DiT** | Predicts noise to remove |
| **DDIM Sampler** | Faster sampling (20-50 steps vs 1000) |
| **CFG (Classifier-Free Guidance)** | Balance text adherence vs diversity (w=7-15) |
| **VAE** | Latent diffusion: compress 512× for efficiency |
| **Super-Resolution** | Upscale 64×64 → 256×256 → 1024×1024 |

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

| GAN Concept | What It Is |
|-------------|------------|
| **Mode Collapse** | G produces limited variety |
| **Truncation (ψ)** | Trade diversity for quality |
| **Latent Space** | 512-dim noise vector → style control |

## 9. VIDEO GENERATION (Sora, Movie Gen)

```
Text → DiT with Temporal Layers → Low-res video → Spatial SR → Temporal SR → Final
              ↑
    3D patches + temporal attention for consistency
```

| Video Component | Purpose |
|-----------------|---------|
| **Temporal Attention** | Consistency across frames |
| **Temporal Convolution** | Local motion patterns |
| **VAE Compression** | 8× temporal + 8×8 spatial = 512× smaller |
| **Spatial Super-Resolution** | 160×90 → 1280×720 |
| **Temporal Super-Resolution** | 8fps → 24fps |

## 10. MULTIMODAL (Vision-Language)

```
IMAGE CAPTIONING:
Image → ViT/CLIP Encoder → Patch embeddings → Cross-attention → Text Decoder → Caption

VQA (Visual Q&A):
Image + Question → Encoder + Cross-attention → Decoder → Answer
```

| Model | Architecture | Use Case |
|-------|--------------|----------|
| **BLIP-2** | Frozen encoder + Q-Former + LLM | Captioning, VQA |
| **LLaVA** | ViT + LLaMA | Open-source VQA |
| **Gemini Vision** | Multimodal encoder + decoder | Everything |

## 11. ALL METRICS YOU NEED TO KNOW

### Text Generation
| Metric | Measures | Higher/Lower Better |
|--------|----------|---------------------|
| **Perplexity** | Model uncertainty | Lower |
| **BLEU** | N-gram precision vs reference | Higher |
| **ROUGE** | Recall vs reference | Higher |
| **METEOR** | Semantic similarity | Higher |

### RAG Quality
| Metric | Measures | Target |
|--------|----------|--------|
| **Faithfulness** | Claims supported by context | > 0.9 |
| **Answer Relevancy** | Addresses the question | > 0.85 |
| **Context Precision** | Retrieved chunks relevant | > 0.8 |
| **Context Recall** | Relevant chunks retrieved | > 0.8 |

### Retrieval
| Metric | Measures |
|--------|----------|
| **Precision@K** | % of top-K that are relevant |
| **Recall@K** | % of relevant in top-K |
| **MRR** | Rank of first relevant result |
| **NDCG** | Ranking quality with graded relevance |
| **Hit Rate** | % queries with ≥1 relevant in top-K |

### Image Generation
| Metric | Measures | Target |
|--------|----------|--------|
| **FID** | Distribution similarity to real images | < 10 |
| **Inception Score (IS)** | Quality × diversity | > 10 |
| **CLIPScore** | Text-image alignment | > 0.3 |
| **LPIPS** | Perceptual similarity | Lower |

### Video Generation
| Metric | Measures | Target |
|--------|----------|--------|
| **FVD** | Fréchet Video Distance (I3D features) | < 300 |
| **FID (per-frame)** | Average frame quality | < 15 |
| **Temporal Consistency** | Smoothness across frames | Human > 4/5 |

### Captioning
| Metric | Measures |
|--------|----------|
| **CIDEr** | Consensus with multiple references |
| **BLEU-4** | 4-gram precision |
| **METEOR** | Semantic similarity |

### LLM Benchmarks
| Benchmark | Tests |
|-----------|-------|
| **MMLU** | Multitask knowledge (57 subjects) |
| **HumanEval** | Code generation |
| **GSM8K** | Math reasoning |
| **TruthfulQA** | Factual accuracy |
| **HellaSwag** | Commonsense reasoning |

### Safety
| Metric | Target |
|--------|--------|
| **Toxicity Rate** | < 0.1% |
| **PII Leak Rate** | 0% |
| **Jailbreak Success** | < 1% |

### Performance
| Metric | What It Measures |
|--------|------------------|
| **TTFT** | Time to first token |
| **Latency P50/P95/P99** | Response time distribution |
| **Throughput** | Requests/second or tokens/second |

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
| Attack | Vector | Defense |
|--------|--------|---------|
| **Direct Prompt Injection** | User input | Input filter, spotlighting |
| **Indirect Prompt Injection** | Retrieved docs contain attacks | Sanitize, instruction hierarchy |
| **Jailbreaking** | Trick model to bypass rules | RLHF, guardrails |
| **Data Leakage** | Model reveals training data | Differential privacy, guardrails |
| **Tool Abuse** | Agent misuses APIs | Least privilege, validation |

### Defense Layers
```
HTTP Layer → Auth → INPUT GUARDRAILS → LLM → OUTPUT GUARDRAILS → Response
                         ↑                          ↑
                   Injection detection         PII filter
                   Blocklist                   Toxicity check
                   Spotlighting                Hallucination check
```

**Spotlighting**: Mark external content → `<<DATA>>user docs here<</DATA>>` → model treats as data, not instructions

### Platform Services
| Purpose | Google Cloud | AWS |
|---------|--------------|-----|
| **Guardrails** | Model Armor | Bedrock Guardrails |
| **WAF** | Cloud Armor | AWS WAF |
| **PII Detection** | Cloud DLP | Amazon Macie |
| **Secrets** | Secret Manager | Secrets Manager |

## 14. SCALABILITY PATTERNS

### GPU Quick Reference
| GPU | Memory | FP16 TFLOPS | Cost | Best For |
|-----|--------|-------------|------|----------|
| **V100** | 16/32 GB | 125 | ~$2/hr | Legacy, small models |
| **A100** | 40/80 GB | 312 | ~$4/hr | Production standard |
| **H100** | 80 GB | 990 | ~$8/hr | Large training, high throughput |
| **L4** | 24 GB | 121 | ~$0.80/hr | Cost-effective inference |
| **TPU v5e** | 16 GB | — | ~$1.20/hr | Google Cloud |

**Model → GPU sizing**: 7B = 14GB (1× L4) | 70B = 140GB (2× H100) | 405B = 810GB (8× H100)

### Inference Scaling
| Pattern | What It Does | When |
|---------|--------------|------|
| **Horizontal** | More replicas | Stateless serving |
| **Tensor Parallelism** | Split layers across GPUs | Model > 1 GPU |
| **Pipeline Parallelism** | Sequential layer split | Very large models |
| **Continuous Batching** | Dynamic batch management | High throughput |

### Training Scaling
| Pattern | What It Does |
|---------|--------------|
| **Data Parallelism** | Same model on each GPU, different data |
| **Model/Tensor Parallelism** | Split model across GPUs |
| **Pipeline Parallelism** | Split layers sequentially |
| **ZeRO/FSDP** | Shard optimizer states, gradients, params |
| **Gradient Checkpointing** | Trade compute for memory |
| **Mixed Precision (AMP)** | FP16 forward, FP32 gradients |

**ZeRO Levels**:
| Level | Shards | Memory Reduction |
|-------|--------|------------------|
| ZeRO-1 | Optimizer states | ~4× |
| ZeRO-2 | + Gradients | ~8× |
| ZeRO-3 | + Parameters | ~N× (N = GPUs) |

## 15. INTERVIEW FRAMEWORK (45 min)

| Phase | Time | What to Cover |
|-------|------|---------------|
| **1. Requirements** | 5-10 min | Token budget, latency, quality, cost, safety |
| **2. Architecture** | 10-15 min | Draw: API → Orchestration → LLM → Response |
| **3. Deep Dive** | 15-20 min | RAG, model choice, eval, security |
| **4. Trade-offs** | 5-10 min | Quality vs cost, latency vs throughput |

**Always do**: Back-of-envelope (tokens × price, latency breakdown)

## 16. QUICK DECISION GUIDE

| Need | Solution |
|------|----------|
| Ground in docs | **RAG** (chunk + vector DB + rerank) |
| Real-time tools | **Function calling** / **Agents** |
| Custom style | **Fine-tuning** (SFT) |
| Ultra-low latency | **Small model** + **edge** + **cache** |
| Reduce hallucination | **RAG** + **grounding** + **citations** |
| Long documents | **Chunking** or **long-context model** |
| Multi-step reasoning | **Agents** (ReAct) |
| Cost at scale | **Routing** + **caching** + **quantization** |
| Generate images | **Diffusion** (Stable Diffusion, Imagen) |
| Generate video | **DiT + temporal layers** (Sora) |
| Generate faces | **StyleGAN** |

## 17. PLATFORM COMPARISON

| | Google Cloud | AWS |
|-|--------------|-----|
| **Models** | Gemini (Vertex AI) | Claude, Titan (Bedrock) |
| **RAG** | Vertex RAG Engine | Bedrock Knowledge Bases |
| **Agents** | ADK, Conversational Agents | AgentCore, Bedrock Agents |
| **Vector DB** | Vertex AI Vector Search | OpenSearch, Pinecone |
| **Guardrails** | Model Armor | Bedrock Guardrails |
| **Image Gen** | Imagen 3 | Titan Image, SD |
| **Video Gen** | Veo | - |

## 18. MODEL QUICK REFERENCE

| Model | Type | Params | Notes |
|-------|------|--------|-------|
| **GPT-4o** | Decoder-only | ~1.7T (rumored MoE) | Multimodal, SOTA |
| **Gemini 1.5 Pro** | MoE | Undisclosed | 1M context |
| **Claude 3.5** | Decoder-only | Undisclosed | Strong reasoning |
| **LLaMA 3** | Decoder-only | 8B-405B | Open-source |
| **Mixtral 8x22B** | MoE | 141B total, 39B active | Open-source |
| **Stable Diffusion** | Latent Diffusion | ~1B | Open-source images |
| **Sora** | DiT + temporal | Undisclosed | Video generation |

## 19. RED FLAGS IN INTERVIEWS

| Red Flag | Better Answer |
|----------|---------------|
| "No RAG needed" | How prevent hallucination? |
| "One model for all" | What about cost optimization? |
| "Skip guardrails" | Prompt injection? PII? |
| "No eval plan" | How know it's working? |
| "Unbounded context" | Cost and latency? |
| "Train from scratch" | Why not fine-tune? |
| "Real-time video gen" | Minutes is realistic |

---

**Core Formula**: GenAI design = Traditional system design + Token economics + Quality evaluation + Safety

**Remember**: Always discuss metrics, always do back-of-envelope, always mention trade-offs
