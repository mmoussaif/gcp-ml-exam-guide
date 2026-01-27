# ML & GenAI System Design Guide

A comprehensive guide to designing **ML (Machine Learning)** and **GenAI (Generative AI)** systems at scale, covering **LLM (Large Language Model)** serving, **RAG** (retrieval-augmented generation) systems, agentic AI, **MLOps** (ML operations) pipelines, and production considerations.

---

## Prerequisites

This guide focuses specifically on **ML and GenAI system design**. For foundational system design concepts (databases, caching, load balancing, networking, CAP theorem, etc.), see:

📖 **[System Design Essentials](./system-design-essentials.md)** - Core system design knowledge applicable to all distributed systems.

---

## Table of Contents

- [Introduction](#introduction)
- [GenAI System: Big Picture (Frontend to Backend)](#genai-system-big-picture-frontend-to-backend)
- [GenAI vs Traditional ML](#genai-vs-traditional-ml)
- [Using Models & Sampling Parameters](#using-models--sampling-parameters)
- [Google Generative AI Development Tools](#google-generative-ai-development-tools)
- [1. LLM Serving Architecture](#1-llm-serving-architecture-at-scale)
- [2. RAG Systems](#2-rag-retrieval-augmented-generation-system)
- [3. RAG vs Fine-Tuning](#3-rag-vs-fine-tuning-decision-framework)
- [4. Agentic AI Systems](#4-agentic-ai-systems)
- [5. LLM Evaluation & Quality](#5-llm-evaluation--quality)
- [6. GenAI Data Pipeline](#6-genai-data-pipeline-architecture)
- [7. Cost Optimization & Model Routing](#7-cost-optimization-for-genai-systems)
- [8. Scalability Patterns](#8-scalability-patterns-for-genai)
- [9. Monitoring & Observability](#9-monitoring--observability-for-genai)
- [10. Security & Guardrails](#10-security--guardrails)
- [11. Real-World Examples](#11-real-world-examples-applying-the-stack)
- [Resources](#resources)

---

## Introduction

Generative AI applications introduce unique challenges that differ significantly from traditional software systems:

- **Token-by-token generation**: Sequential decoding (unlike batch predictions)
- **Variable latency**: Generation time depends on output length
- **High memory requirements**: **KV cache** (key-value cache: stored attention keys and values in transformers) for attention mechanisms
- **Cost optimization**: Balance between latency and throughput
- **Hallucination management**: Ensuring factual accuracy
- **Agent orchestration**: Multi-step reasoning and tool use

This guide covers how to design, build, and operate GenAI systems at scale.

**Aha:** GenAI system design is different because you're optimizing for **non-determinism** (same prompt → different outputs), **token economics** (cost and latency scale with length), and **orchestration** (models + retrieval + tools), not just throughput of identical requests.

---

## GenAI System: Big Picture (Frontend to Backend)

Before diving into components, here is the end-to-end shape of a GenAI system. The **request path** runs from frontend to backend; **supporting systems** (data pipelines, evaluation, monitoring, security) surround that path. Each numbered section later in this guide is a T-shaped deep dive on one layer or concern: broad role in this picture first, then detail.

**Request path (frontend → backend):**

```
  User / Frontend
        │
        ▼
  ┌─────────────────┐
  │  API Gateway    │  Auth, rate limit, route
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Orchestration  │  Agent, RAG, tools (sections 2, 4)
  │  (Agent / RAG)  │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  LLM(s)         │  Inference, model routing (section 1)
  └────────┬────────┘
           │
           ▼
  Response (→ user, or → tools, then back into orchestration)
```

**Supporting systems (around the request path):**

| System | Role in the big picture | Deep dive |
|--------|--------------------------|-----------|
| **Evaluation** | "Did we build the right thing?" — quality, grounding, safety on a sample of requests | §5 Evaluation & Quality (metrics + eval *data* pipeline at scale) |
| **Training data pipeline** | "Where do fine-tuning examples come from?" — user interactions → events → lake → training prep | §6 GenAI Data Pipeline |
| **Cost** | "How do we keep inference affordable?" — tokens, caching, model routing, quantization | §7 Cost Optimization |
| **Scale** | "How do we serve more load?" — horizontal scaling, model/pipeline parallelism, KV cache | §8 Scalability |
| **Monitoring** | "How do we observe the system?" — metrics, traces, drift | §9 Monitoring & Observability |
| **Security** | "How do we protect inputs, outputs, and access?" — guardrails, Model Armor, IAM | §10 Security & Guardrails |
| **Real-world examples** | "How do I build this with real tools?" — apply §1–§10 with LangChain, AWS, Google, open source | §11 Real-World Examples |

**Rationale in one line:** The **request path** (gateway → orchestration → LLM) is what users hit. **Evaluation** and **training data** are two different data flows: eval = "log predictions → run quality metrics" (§5); training = "log interactions → clean → fine-tune" (§6). **Cost** (§7) is *spend per request*; **scale** (§8) is *throughput and capacity*. **Monitoring** (§9) and **security** (§10) are cross-cutting. **Examples** (§11) come last so you can apply everything with concrete stacks.

**Logical flow of this guide:** Big Picture → foundations (GenAI vs ML, sampling, Google tools) → **request path** (§1 Serving, §2 RAG, §3 RAG vs FT, §4 Agents) → **evaluation** (§5: what to measure + eval data pipeline at scale; *consolidated* so "evaluation" is one place) → **training data** (§6) → **efficiency** (§7 Cost, §8 Scale) → **operations** (§9 Monitoring, §10 Security) → **§11 Real-World Examples** (apply §1–§10 with LangChain, AWS, Google, open source). Examples are last so every concept is already defined when you see concrete solutioning.

---

## GenAI vs Traditional ML

Understanding the fundamental differences between traditional ML systems and **GenAI** / **LLM (Large Language Model)** systems is crucial for making the right architectural decisions.

| Aspect         | Traditional ML       | GenAI/LLM                              |
| -------------- | -------------------- | -------------------------------------- |
| **Prediction** | Single forward pass  | Token-by-token generation              |
| **Latency**    | Fixed (milliseconds) | Variable (seconds to minutes)          |
| **Memory**     | Model weights        | Model + KV cache (grows with sequence) |
| **Batching**   | Static batches       | Dynamic/continuous batching            |
| **Cost**       | Per-request          | Per-token (input + output)             |
| **Control**    | Fixed weights        | Sampling parameters (temp, top-p, top-k) |

**Why these differences matter:**

- **Token-by-token generation** means you can't predict exact response time—a 10-token response is much faster than a 1000-token response.
- **KV cache growth** means memory requirements increase with context length, limiting how many concurrent requests you can serve.
- **Per-token pricing** means prompt engineering and response length directly impact costs.

**Aha:** Traditional ML is "one input → one prediction." GenAI is "one prompt → a stream of tokens, each depending on the last." That shifts bottlenecks from GPU compute to memory (KV cache), latency (time-to-first-token vs total time), and cost (every token billed).

---

## Using Models & Sampling Parameters

Generative AI agents are powered by models that act as the "brains" of the operation. While models are pre-trained, their behavior during inference can be customized using **sampling parameters**—the "knobs and dials" of the model.

### Common Sampling Parameters

**1. Temperature**

Controls the "creativity" or randomness of the output by rescaling logits before softmax.

- **High Temperature (T > 1)**: Flattens the distribution, making output more random, diverse, and unpredictable. Increases risk of incoherence.
- **Low Temperature (T < 1)**: Sharpens the distribution, making it more focused, deterministic, and repeatable.
- **Extreme (T → 0)**: Collapses into greedy decoding (always picks the highest probability token).

*Use low temperature (0.1-0.3) for factual tasks, higher (0.7-1.0) for creative tasks.*

**Aha:** Temperature rescales logits before sampling. Low T makes the top token dominate (nearly deterministic); high T flattens the distribution so unlikely tokens get a real chance. You're tuning "how much to trust the model's confidence."

**2. Top-p (Nucleus Sampling)**

Selects the smallest set of tokens whose cumulative probability mass reaches threshold *p*.

- **High Top-p (0.9-1.0)**: Allows for more diversity by extending to lower probability tokens.
- **Low Top-p (0.1-0.5)**: Leads to more focused responses.
- **Adaptive**: Unlike Top-K, adapts to the distribution's shape—in confident contexts, the "nucleus" is small.

**Aha:** Top-p says "consider only tokens that together account for probability mass *p*." When the model is sure, that might be 2–3 tokens; when unsure, many more. So Top-p scales with confidence; Top-K does not.

**3. Top-K**

Restricts the model's choice to only the *k* most probable tokens at each step.

- Improves output stability by eliminating the "long tail" of extremely unlikely tokens.
- **Limitation**: Unlike Top-p, it is not adaptive to the distribution's shape.

**4. Maximum Length (Max New Tokens)**

Determines the maximum number of tokens to generate before stopping.

- Prevents runaway generation ("rambling") and controls compute costs.
- Models stop early if they hit an end-of-sequence (`<EOS>`) token.

**5. Repetition Penalty**

A factor (usually > 1.0) used to discount the probability of tokens that have already appeared in the output.

- Prevents the model from getting stuck in repetitive loops (e.g., "I'm not sure. I'm not sure.").

**6. Safety Settings**

Filters that block potentially harmful or inappropriate content (hate speech, harassment, etc.).

- Essential for enterprise-grade applications to ensure outputs align with safety policies.

### Accessing Parameters via APIs

Most generative AI models are accessed via **APIs**. The flow:

1. Your application sends a **Prompt** + **Sampling Parameters**
2. The API delivers these to the model
3. The model generates a response based on those specific parameters
4. The API returns the response to your application

---

## Google Generative AI Development Tools

Google provides two primary environments for experimenting with and deploying Gemini models:

| Attribute | Google AI Studio | Vertex AI Studio |
| :--- | :--- | :--- |
| **Focus** | Streamlined, easy-to-use interface for rapid prototyping | Comprehensive environment for building, training, and deploying ML models |
| **Target Users** | Beginners, hobbyists, initial project stages | Professionals, researchers, enterprise developers |
| **Access** | Standard Google Account login | Google Cloud Console (Enterprise account) |
| **Limitations** | Usage limits (**QPM** queries/min, **RPM** requests/min, **TPM** tokens/min); small-scale projects | Service charges based on usage; enterprise-grade quotas |
| **Advantages** | Simplified interface; easy to get started | Enterprise-grade security, compliance, flexible quotas |

**Key Takeaway**: Use **Google AI Studio** for fast, small-scale prototyping. Transition to **Vertex AI Studio** for large-scale, production-ready enterprise applications.

### Google's Generative AI APIs

Google's generative AI APIs offer pre-trained foundation models that can be fine-tuned for specific tasks:

- **Text Completion**: Generating long-form content or completing snippets
- **Multi-turn Chat**: Maintaining state across several turns of conversation
- **Code Generation**: Specialized models for writing and debugging code
- **Image Generation**: Using the Imagen API to create and customize images

---

## 1. LLM Serving Architecture at Scale

### Use Case: Design a Chatbot Service (like ChatGPT)

**Requirements:**
- Support 1M concurrent users
- Average response time < 2 seconds
- Handle 10,000 requests/second
- Support multiple models (GPT-4, Claude, Gemini)
- Cost-effective serving

**High-Level Design:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM SERVING ARCHITECTURE                     │
│                                                                 │
│   ┌──────────┐     ┌─────────────┐     ┌───────────────────┐   │
│   │ Clients  │────►│ API Gateway │────►│  Request Router   │   │
│   │ Web/API  │     │ Auth, Rate  │     │  Load Balancer    │   │
│   └──────────┘     │ Limiting    │     └─────────┬─────────┘   │
│                    └─────────────┘               │             │
│                                                  │             │
│                          ┌───────────────────────┼─────────┐   │
│                          │                       │         │   │
│                          ▼                       ▼         ▼   │
│                    ┌───────────┐           ┌───────────────────┐│
│                    │   Cache   │           │  LLM Serving      ││
│                    │  (Redis)  │           │  Infrastructure   ││
│                    │           │           │                   ││
│                    │• Prompt   │           │ • Vertex AI       ││
│                    │  Cache    │           │ • SageMaker       ││
│                    │• Response │           │ • vLLM/TensorRT   ││
│                    │  Cache    │           │                   ││
│                    │• Semantic │           │ Continuous batch  ││
│                    │  Cache    │           │ KV cache mgmt     ││
│                    └───────────┘           └───────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**1. Model Serving Infrastructure**

| Option | Pros | Cons | Best For |
|--------|------|------|----------|
| **Managed (Vertex AI / SageMaker)** | Zero infra management, auto-scaling, built-in monitoring | Less optimization control, vendor lock-in, higher costs at scale | Startups, rapid prototyping, small ops teams |
| **Self-hosted (vLLM / TensorRT-LLM)** | Full control, better cost efficiency at scale, customizable | Requires ML infra expertise, GPU management complexity | High volume (millions/day), cost-sensitive |

**2. Continuous Batching**

**Problem**: Static batching wastes GPU when requests finish at different times.

**Why this happens**: LLM generation is sequential (token-by-token), so requests in a batch finish at different times. With static batching, the GPU waits for the slowest request before processing the next batch.

**Solution**: Dynamic batching—add new requests to batch as others complete.

```
Time 0: [Request A (100 tokens)]
Time 1: [Request A (50 tokens), Request B (100 tokens)] ← Added B
Time 2: [Request B (50 tokens), Request C (100 tokens)] ← A finished, added C
Time 3: [Request C (50 tokens), Request D (100 tokens)] ← B finished, added D
```

**Benefit**: 2-3x higher throughput because GPU utilization increases from ~40% to ~85%.

**Aha:** With static batching, one long answer blocks the whole batch. Continuous batching **refills** the batch as soon as any request completes, so the GPU rarely idles. The "aha" is: treat the batch as a **queue**, not a fixed group.

**3. KV Cache Management**

**What**: Store the **Key** and **Value** matrices produced by each attention head so they are not recomputed. In standard attention, the score matrix has shape `[batch, heads, sequence_length, sequence_length]`; each new token would require recomputing scores over all previous tokens.

**Why KV cache is needed**: Autoregressive decoding feeds all prior tokens into the next step. Without caching, every generation step recomputes keys and values for the entire prefix, giving O(n²) work per token. Caching lets you compute K and V only for the new token and reuse the rest, reducing to O(n) per token. Reported speedups from KV caching are on the order of ~30–40% in standard implementations.

**How it works**: For each new token, compute and store its K and V; look up cached K/V for all previous positions when computing attention. Only the new token’s key/value are written each step.

**Challenge**: Cache size grows linearly with sequence length (and with layers × heads × head_dim). For a 32-layer model with 768-dim embeddings, each token can use on the order of ~50KB of cache; a 2K-token sequence can need ~100MB of KV cache. Long contexts and many concurrent requests make this the main memory bottleneck.

**Solution — PagedAttention (vLLM)**: Inspired by OS virtual memory and paging. The KV cache is split into **fixed-size blocks** and stored in non-contiguous memory. That reduces fragmentation and allows sharing (e.g. shared prompt prefix across requests). vLLM reports near-zero wasted KV memory and roughly **2–4× throughput** versus non-paged systems on long sequences and large models.

**5. Speculative Decoding**

**Problem**: Token-by-token autoregressive generation is slow because each new token requires a full forward pass of the large model.

**Solution**: A small **draft** model proposes several candidate tokens in a row. The **target** (large) model does a single forward pass over the whole candidate sequence and accepts tokens that match its predictions; the first mismatch stops the run and the rest are discarded. Accepted tokens advance the sequence without extra target-model steps. Typical reported speedups are **2–2.5×**; variants (multiple draft models, tree-based decoding) can reach ~3–4× or more at the cost of extra memory and complexity.

| Technique | Speedup | Trade-off |
|-----------|---------|-----------|
| **Standard Speculative** | 2–2.5× (often up to ~3×) | Needs a separate draft model |
| **Self-Speculative** | ~2.5× | Uses smaller/quantized version of same model |
| **Tree-based** | Up to ~4–6× | More memory and logic for tree search |

**Why it works**: The target model verifies **N** candidates in one forward pass (over a sequence of length N). That cost is similar to generating a single token, so you effectively get several tokens per large-model step when the draft is accurate. **Draft latency** (how fast the draft runs) usually matters more for end-to-end speedup than the draft’s raw language quality.

**4. Caching Strategy**

| Strategy | Hit Rate | Latency | Best For |
|----------|----------|---------|----------|
| **Prompt caching** | High for system prompts | 2-5x speedup | Common prefixes, few-shot examples |
| **Response caching** | 10-30% | Instant | Identical requests |
| **Semantic caching** | 30-50% | +5-10ms overhead | Paraphrased queries |

---

## 2. RAG (Retrieval-Augmented Generation) System

### Use Case: Design a Document Q&A System

**Requirements:**
- Answer questions from 1M documents
- Support real-time queries (< 3 seconds)
- Handle 1,000 **QPS** (queries per second)
- Ensure factual accuracy (grounding)

**High-Level Design:**

```
┌─────────────────────────────────────────────────────────────────┐
│                      RAG ARCHITECTURE                           │
│                                                                 │
│   INGESTION PIPELINE                                            │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│   │Documents │──►│ Chunking │──►│Embedding │──►│ Vector   │   │
│   │          │   │          │   │  Model   │   │   DB     │   │
│   └──────────┘   └──────────┘   └──────────┘   └──────────┘   │
│                                                                 │
│   QUERY PIPELINE                                                │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│   │  Query   │──►│ Embed    │──►│Similarity│──►│ Top-K    │   │
│   │          │   │  Query   │   │  Search  │   │  Docs    │   │
│   └──────────┘   └──────────┘   └──────────┘   └────┬─────┘   │
│                                                      │         │
│                                              ┌───────▼───────┐ │
│                                              │   Reranker    │ │
│                                              │  (optional)   │ │
│                                              └───────┬───────┘ │
│                                                      │         │
│   ┌──────────────────────────────────────────────────▼───────┐ │
│   │                        LLM                                │ │
│   │   Query + Retrieved Context → Generated Answer           │ │
│   └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Aha:** RAG doesn't cram everything into the model's weights. It keeps the LLM general and **fetches** relevant knowledge at query time. That gives you updatable knowledge, smaller models, and citations—but you must design retrieval and chunking well or the model "makes it up."

### Key Components

**1. Document Ingestion Pipeline**

| Service | Google Cloud | AWS |
|---------|--------------|-----|
| RAG Engine | Vertex AI RAG Engine | Bedrock Knowledge Bases |
| Vector Search | Vertex AI Vector Search | OpenSearch Serverless |
| Processing | Dataflow | Glue/EMR |

**2. Vector Database Options**

- **Managed**: Vertex AI Vector Search, Amazon OpenSearch
- **Self-hosted**: Pinecone, Weaviate, Qdrant, Milvus

**3. Embedding Models**

- **Google**: text-embedding-004 (Vertex AI)
- **AWS**: Amazon Titan Embeddings (Bedrock)
- **Open Source**: sentence-transformers, **BGE** (BAAI General Embeddings)—embedding models from BAAI (Beijing Academy of Artificial Intelligence), e.g. bge-base, BGE-M3 for multilingual

### Chunking Strategy Trade-offs

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| **Fixed-size (512 tokens)** | Simple, predictable | May split concepts | Uniform documents |
| **Semantic chunking** | Preserves coherence | Complex, variable sizes | Complex content |
| **Hybrid (fixed + overlap)** | Balanced | More storage | Most production systems |

**Why chunking matters**: LLMs have context windows. Documents often exceed this, so we must break them into chunks. Smaller chunks improve retrieval precision—a query about "Python loops" matches better to a 500-token chunk about loops than a 5000-token document about Python.

**Aha:** Chunk size is a **precision vs context** trade-off. Too small → you retrieve the right idea but maybe miss surrounding explanation. Too large → you get more context but dilute relevance. Overlap and semantic boundaries help keep "one concept per chunk."

### Retrieval Strategy Trade-offs

| Strategy | Latency | Semantic | Keywords | Best For |
|----------|---------|----------|----------|----------|
| **Dense (Vector)** | 10-50ms | ✓ | ✗ | Conceptual queries |
| **Sparse (BM25)** | 1-5ms | ✗ | ✓ | Exact matches |
| **Hybrid** | 15-60ms | ✓ | ✓ | Production (recommended) |

**BM25** = keyword-based ranking using term frequency and inverse document frequency; no embeddings, just lexical match.

**Why hybrid works**: Dense retrieval captures meaning ("iterate" ≈ "loop"), sparse captures exact keywords ("Python"). Combining both via **RRF (Reciprocal Rank Fusion)** gives best results.

**Aha:** **Dense** = "these two *mean* the same thing" (embedding similarity). **Sparse** = "these two *contain* the same words" (e.g. BM25). Queries need both: "how do I loop in Python?" benefits from semantic match on "loop" and exact match on "Python." Hybrid + RRF merges the two rank lists without a single embedding doing everything.

### Reranking Trade-offs

**No Reranking**: Lower latency, simpler pipeline, but lower quality.

**Cross-Encoder Reranking**: Much higher accuracy because it processes query-document pairs together (sees interactions), but adds ~10ms per document.

**Best practice**: Retrieve K=20, rerank to top 5. The two-stage approach combines speed (bi-encoder retrieval) with accuracy (cross-encoder reranking).

**Aha:** **Bi-encoder** = query and doc are embedded *separately*; similarity is dot product. Fast (one pass each) but the model never sees "query + doc together." **Cross-encoder** = one forward pass with "[query] [doc]"; the model sees the *pair* and scores relevance directly. Slower, but much more accurate. So: retrieve broadly with bi-encoder, then rerank the top K with a cross-encoder.

### Advanced RAG Techniques

These techniques improve retrieval when plain “embed query → top‑k chunks” is not enough: when answers span multiple hops, when queries vary in difficulty, or when user wording doesn’t match document wording.

---

**1. Graph RAG**

**What it is:** You build a **knowledge graph** from your corpus (entities as nodes, relations as edges) and combine it with vector search. Retrieval can follow *links* (e.g. “this person → worked at → this company”) as well as semantic similarity.

**How it helps:** Many questions need **multi-hop** reasoning: “Who was the CEO of the company that acquired X?” requires (X → acquired by → company → CEO → person). Flat vector search often returns only one hop. Graph RAG retrieves **subgraphs** (e.g. k-hop neighborhoods) so the LLM sees not just similar text but explicit *who–what–where* structure.

**When to use:** Strong fit for domains rich in **entities and relations** (people, orgs, products, events) and questions that chain them. Overkill for unstructured long-form text with few named relations.

**Aha:** Vector search answers “what text is similar?” Graph RAG adds “how are these things *connected*?” so the model can follow paths, not only similarity.

---

**2. Adaptive Retrieval**

**What it is:** Instead of always retrieving the same number of documents (e.g. k=10), you **change k per query**. Simple factoid questions get fewer docs; broad or multi-fact questions get more.

**How it helps:** With a **fixed k**, easy questions get unnecessary context (wasted tokens, more noise) and hard questions may get too few (missing evidence). Adaptive retrieval uses a small classifier, heuristics (e.g. query length, question type), or the **shape of similarity scores** (e.g. “biggest drop” between consecutive docs) to choose k. Some methods need no extra model—e.g. set k at the largest score gap in the ranked list.

**When to use:** When your traffic mixes **simple lookups** and **complex / multi-document** questions. Saves tokens and latency on easy queries and improves recall on hard ones.

**Aha:** One size doesn’t fit all: “What is the capital of France?” needs 1–2 chunks; “Compare the economic policies of France and Germany in the 1980s” needs many. Adaptive k tunes retrieval to each question.

---

**3. Query Decomposition**

**What it is:** Before retrieval, an LLM **splits** the user question into 2–5 **sub-questions** that are answered by different parts of the corpus. You run retrieval once per sub-question, then merge and deduplicate the chunks and pass that combined context to the final answer model.

**How it helps:** Questions like “How does X differ from Y?” or “Which of A, B, C had the highest Z?” don’t match one passage—they need **several**. One query embedding often misses some of them. Decomposing into “What is X?”, “What is Y?”, “How do they differ?” (or “What is Z for A?”, “What is Z for B?”, …) yields focused sub-queries and better coverage.

**When to use:** **Multi-part** or **comparison** questions, and whenever a single embedding tends to retrieve only one “side” of the answer. Adds latency (one LLM call to decompose, then multiple retrievals) but can significantly improve accuracy.

**Aha:** One query → one vector → one retrieval set often undersamples. Decomposing “How does A differ from B?” into “What is A?” and “What is B?” (and optionally “How do they differ?”) pulls in the right evidence for each piece, then the model synthesizes.

---

**4. HyDE (Hypothetical Document Embeddings)**

**What it is:** You **don’t** embed the user query directly. Instead, you ask an LLM: “Write a short passage that would answer this question.” You get 1–5 such **hypothetical** passages, embed *those*, and (often) **average** their vectors. That single vector is used to search the real document index.

**How it helps:** Query and documents often use **different words** for the same idea (e.g. user: “loop,” docs: “iteration construct”). The query embedding can sit in a different region of the embedding space than the best-matching docs. Hypothetical answers “translate” the question into **passage-like** text, so their embeddings sit closer to real relevant passages. Averaging smooths noise from any one generation.

**When to use:** When **vocabulary mismatch** hurts recall (e.g. lay users vs technical docs, or one language vs translated corpus) and when you can afford one extra LLM call before retrieval. Less useful when queries already look like document sentences.

**Aha:** You’re searching with “what an answer would look like” instead of “what the question looks like.” The hypothetical doc is in the same “language” as your corpus, so similarity search works better.

---

**Quick reference**

| Technique | Main idea | Best for |
|-----------|-----------|----------|
| **Graph RAG** | Vector search + graph structure (entities, relations); retrieve subgraphs for multi-hop | Entity-heavy domains, “who/what/where” chains |
| **Adaptive Retrieval** | Vary number of retrieved docs (k) by query complexity | Mix of simple and complex questions |
| **Query Decomposition** | Split question into sub-questions; retrieve per sub-question; merge context | Multi-part, comparison, “A vs B” style questions |
| **HyDE** | Generate hypothetical answer(s), embed those, search with that vector | Vocabulary mismatch between user and corpus |

---

## 3. RAG vs Fine-Tuning Decision Framework

**Key insight:** This is not a binary choice. Think of it as a **spectrum of adaptation**: RAG and fine-tuning solve different problems and are often used **together**. The right question is not "RAG or fine-tuning?" but "What does the model lack—**knowledge** or **behavior**?"

- **"The model doesn't *know* X"** → Add knowledge via RAG (or long context, or caching).
- **"The model doesn't *behave* like Y"** → Change behavior via fine-tuning (tone, format, schema, jargon).
- **"We need both fresh facts and consistent style"** → Use both: RAG for what to say, fine-tuning for how to say it.

---

### When to Use RAG

**What RAG fixes:** Gaps in **knowledge** and **freshness**. The model is good at reasoning and language but hasn't seen your data (policies, tickets, docs, logs). RAG injects that at query time: you retrieve relevant chunks and put them in the prompt, so the model "reads" your corpus on demand.

**Use RAG when:** The model **lacks knowledge** about your domain (e.g. internal docs, product specs, support history). Your **data changes often** (e.g. daily reports, new releases, tickets)—RAG lets you update the index without retraining. You want to **reduce hallucinations** by **grounding** answers in retrieved text and to **cite sources** (chunk or doc IDs).

**RAG does *not* fix:** Tone, format, or jargon. If the base model is too informal or ignores your schema, RAG alone won't change that—you need behavior change (prompts or fine-tuning).

---

### When to Use Fine-Tuning

**What fine-tuning fixes:** **Behavior** and **style**. The model "knows" enough from pretraining, but its outputs don't match how you want it to answer: tone (formal vs casual), structure (e.g. JSON with fixed keys), or vocabulary (your domain terms). Fine-tuning adjusts the model's weights so it reliably produces that style.

**Use fine-tuning when:** You need a **specific tone or voice** (e.g. brand guidelines, compliance-friendly wording). You need **strict output format** (e.g. JSON, bullet lists, section headings)—fine-tuning helps the model adhere to schemas. The model **misuses or avoids domain jargon**; training on in-domain examples teaches it to use your terms correctly.

**Fine-tuning does *not* fix:** Missing or outdated facts. Weights are fixed until the next train run. For fast-changing knowledge, use RAG (or both).

---

### When to Use Both

**Use RAG + fine-tuning when** you need **accurate, up-to-date content** *and* **consistent presentation**: RAG supplies the **facts** (from docs, KB, logs); fine-tuning shapes **how** those facts are expressed (tone, format, terminology). Example: A support bot that answers from your knowledge base (RAG) but must always respond in a compliant, on-brand style (fine-tuned). Or a report generator that pulls from live data (RAG) and always outputs the same JSON schema (fine-tuned).

---

### Scenario Cheat Sheet

| Scenario | RAG | Fine-Tuning | Both |
|----------|:---:|:-----------:|:----:|
| Model lacks knowledge about your domain | ✅ | ❌ | |
| Data changes frequently (docs, tickets, metrics) | ✅ | ❌ | |
| Need specific tone, style, or brand voice | ❌ | ✅ | |
| Domain-specific jargon or terminology | ❌ | ✅ | |
| Reduce hallucinations by grounding in retrieved text | ✅ | | |
| Change output format or schema (e.g. JSON, sections) | ❌ | ✅ | |
| High accuracy *and* fresh data *and* consistent style | | | ✅ |

### Cost Comparison

Cost structure is different, not just "cheaper vs more expensive":

| Approach | Cost model | What you pay for | Example ballpark |
|----------|------------|------------------|------------------|
| **RAG** | **Per query** | Retrieval (embeddings, vector search) + LLM tokens (context + answer) | ~$0.01-0.05 per query; 1M queries/month ≈ $10-50K |
| **Fine-tuning (e.g. LoRA)** | **One-time** | Training compute + data prep; then inference cost as usual | ~$500-2,000 for **LoRA** (Low-Rank Adaptation) on 7-70B model; amortizes over all future requests |
| **Full fine-tune** | **One-time, large** | Full training run on your data | $10K-100K+ depending on model size and data |

**How to think about it:** RAG cost grows with **usage** (every query pays). Fine-tuning cost is **upfront**; after that, marginal cost per request is similar to the base model (or lower if you use a smaller fine-tuned model). Break-even depends on volume: at very high QPS, RAG can exceed the amortized cost of a one-time fine-tune; at low QPS, RAG is often cheaper than investing in fine-tuning.

### Decision Flow

Start with the **cheapest, fastest** lever (prompts and few-shot examples). Only add RAG or fine-tuning when you've identified a clear gap: knowledge vs behavior.

```
Start with: System prompt + few-shot examples
        │
        ▼
Does the model lack KNOWLEDGE about your domain?
(e.g. your docs, products, policies, tickets)
        │
    Yes ─┴─ No
        │     │
        ▼     ▼
   Add RAG   Does the model need BEHAVIOR change?
            (e.g. tone, format, schema, jargon)
                    │
               Yes ─┴─ No
                    │     │
                    ▼     ▼
            Fine-tune   Done
```

You can **add RAG and then fine-tune** (or the reverse) if you need both knowledge and behavior. Many production systems use prompts + RAG + fine-tuning together.

---

### Best Practice

1. **Start simple:** Prompt engineering + a few examples. Ship and measure.
2. **Add RAG** when the main gap is "model doesn't know our content" or "content changes often."
3. **Add fine-tuning** when the main gap is "model doesn't answer in our tone/format/terms."
4. **Combine** when you need both correct, up-to-date content and consistent presentation.

**Aha:** RAG = **external memory** you can change without retraining (add docs, edit, delete). Fine-tuning = **internalized behavior** (tone, format, jargon) that’s fixed until the next train run. Use RAG when the world changes; use fine-tuning when you want the model itself to change how it answers.

---

## 4. Agentic AI Systems

### What Is an Agent? Why Do We Need One?

**Definition:** An **agent** is an LLM that **repeatedly** decides, acts, and observes until a task is done. It has access to **tools** (APIs, databases, search, code) and runs in a **loop**: perceive the current state → decide the next step → call a tool → observe the result → repeat. That loop is what makes it an agent, not "one prompt → one answer."

**Why we need agents:** A single LLM call is stateless and one-shot. It can't look up live data, call your CRM, or run multi-step workflows. **RAG** adds retrieval at query time but still produces one answer from one retrieved context—no tool calls, no iterative refinement. **Agents** add the ability to *use the world*: query systems, run code, search, then decide what to do next from the results. So you need an agent when the task requires **multiple steps**, **live data** (orders, DB, APIs), or **decisions that depend on tool outputs** (e.g. "if order status is X, do Y").

**When to use agents vs. not:**

| Use an agent when… | Use a single call or RAG when… |
|-------------------|-------------------------------|
| The task needs **multiple tool calls** or steps (e.g. check order → update CRM → create ticket) | The task is **one question → one answer** (e.g. "what is our return policy?") |
| The **next step depends on live results** (e.g. "if refund approved, then…") | The pipeline is **fixed** (e.g. embed query → retrieve → generate) |
| You need **orchestration across systems** (APIs, DBs, search) | You only need **retrieval + generation** (RAG) or pure generation |
| Decisions are **context-sensitive** and hard to encode as rules | The flow is **deterministic** and easy to script |

**Aha:** Start with the simplest thing that works (single call, or RAG). Add an agent only when you need **loop + tools**—when the model must *use* external systems and *iterate* based on what it sees.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            SINGLE CALL / RAG vs AGENT                                        │
│                                                                              │
│   SINGLE CALL or RAG                    AGENT                               │
│   ────────────────────                  ─────                               │
│   User → Prompt (+ RAG?) → LLM → Answer  User → Prompt → LLM → Thought       │
│   (one shot)                                  │                              │
│                                         Tool call → Observation → (repeat)   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Use Case: Design a Customer Support Agent

**Requirements:**
- Handle customer inquiries autonomously
- Access multiple tools (CRM, knowledge base, order system)
- Support multi-turn conversations
- Escalate to human when needed
- Handle 10,000 conversations/day

**Why an agent fits here:** Support often needs *multi-step* actions (look up order → check policy → create ticket or escalate) and *live data* (order status, account history). One LLM call or RAG-only can't do that; you need a loop + tools.

**High-Level Design:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC AI ARCHITECTURE                      │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │                  USER MESSAGE                             │ │
│   └────────────────────────┬─────────────────────────────────┘ │
│                            │                                    │
│                            ▼                                    │
│   ┌──────────────────────────────────────────────────────────┐ │
│   │              AGENT ORCHESTRATOR (LLM)                     │ │
│   │                                                           │ │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │
│   │   │  REASONING  │─►│   ACTING    │─►│ OBSERVATION │     │ │
│   │   │  (Analyze)  │  │(Tool call)  │  │  (Result)   │     │ │
│   │   └─────────────┘  └─────────────┘  └──────┬──────┘     │ │
│   │                            ▲                │             │ │
│   │                            └────────────────┘             │ │
│   │                         (Iterate until done)              │ │
│   └────────────────────────┬─────────────────────────────────┘ │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                │
│         │                  │                  │                │
│         ▼                  ▼                  ▼                │
│   ┌───────────┐      ┌───────────┐      ┌───────────┐         │
│   │  Tool 1   │      │  Tool 2   │      │  Tool 3   │         │
│   │ Knowledge │      │  Order    │      │  Create   │         │
│   │   Base    │      │  Status   │      │  Ticket   │         │
│   └───────────┘      └───────────┘      └───────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

**Aha:** An agent is an LLM in a **loop** with tools. The model doesn’t just answer once; it *reasons → acts (calls a tool) → observes (gets result) → reasons again* until it can respond. That turns the LLM into a controller over APIs, DBs, and search—so the "aha" is: the value is in the **loop + tools**, not in a bigger model.

### Agent Frameworks

Choose **no-code** (Vertex AI Agent Builder, Bedrock Agents) when you want to configure agents in a UI with minimal code. Choose **programmatic** (ADK, LangChain, LlamaIndex) when you need custom logic, complex workflows, or fine-grained control.

| Platform | Google Cloud | AWS | Open Source |
|----------|--------------|-----|-------------|
| No-code | Vertex AI Agent Builder | Bedrock Agents | - |
| Programmatic | Agent Development Kit (ADK) | AgentCore | LangChain, LlamaIndex, AutoGen |

---

### Tool Types

**Tools** are how the agent interacts with the world: APIs, DBs, search, code. The agent chooses *which* tool to call and *with what arguments*; the tool runs and returns a result, which the agent uses for the next step.

| Tool Type | Execution | Description | Best For |
|-----------|-----------|-------------|----------|
| **Extensions (APIs)** | Agent-side | Standardized bridges to external APIs | Multi-service access |
| **Function Calling** | Client-side | Model outputs function name + args; your app executes | Security, audit, human-in-loop |
| **Data Stores** | Agent-side | Connect to vector DBs, knowledge bases | RAG, real-time info |
| **Plugins** | Agent-side | Pre-built integrations (calendar, CRM) | Rapid capability addition |

**Aha:** **Function calling** (client-side) gives you control: the model outputs a tool name + args, and *your app* decides whether to run it. Use it when you need security, audit, or human-in-the-loop. **Agent-side** tools run automatically when the model requests them—faster but less control.

---

### Agent Protocols: MCP and A2A

**MCP (Model Context Protocol)** and **A2A (Agent-to-Agent / Agent2Agent)** are open standards that define how agents get **tools and context** (MCP) and how **agents talk to other agents** (A2A). Both matter when you build multi-tool or multi-agent systems.

**MCP (Model Context Protocol)**

**MCP** is an open protocol (Anthropic, 2024) that standardizes how applications provide **tools and context** to LLMs. It acts as a universal connector: an LLM or agent connects to **MCP servers**, which expose tools, prompts, and resources (files, DBs, APIs) in a consistent way. So instead of each vendor defining its own tool format, you run or connect to MCP servers and the model gets a uniform interface.

| Aspect | Description |
|--------|-------------|
| **Purpose** | Standardize how models get tools, prompts, and resources from external systems |
| **Adoption** | Anthropic (Claude), OpenAI (Agents SDK), Microsoft (Agent Framework) |
| **Use cases** | AI-powered IDEs, custom workflows, connecting agents to Slack, Figma, databases, etc. |

**When it matters:** Use MCP when you want **portable tooling**—the same MCP server can back multiple agents or products. It also helps when you integrate many external systems (CRMs, docs, search) without writing custom glue per vendor.

**A2A (Agent-to-Agent / Agent2Agent Protocol)**

**A2A** is an open standard (Google, 2025) for **communication and collaboration between AI agents** built by different vendors and frameworks. It addresses interoperability: agents from different stacks (e.g. Vertex AI, LangChain, Salesforce) can discover each other, negotiate UX, and exchange tasks and state **without** sharing internal memory, resources, or tools.

| Aspect | Description |
|--------|-------------|
| **Purpose** | Enable agent-to-agent collaboration across vendors and frameworks |
| **Mechanisms** | **Agent Cards** (JSON metadata: identity, capabilities), capability discovery, task/state management, UX negotiation |
| **Transport** | JSON-RPC 2.0 over HTTP(S) |
| **Relationship to MCP** | A2A handles **agent ↔ agent**; MCP handles **model ↔ tools/context**. They complement each other. |

**When it matters:** Use A2A when you run **multi-agent** or **cross-vendor** workflows (e.g. your agent hands off to a partner’s agent, or you compose agents from different platforms). It gives you a shared protocol for discovery, tasks, and security instead of one-off integrations.

**Aha:** **MCP** = “how does *this* agent get its tools and context?” **A2A** = “how do *multiple* agents from different systems work together?” For a single agent with your own tools, MCP is the standard to consider. For agent-to-agent orchestration across products or vendors, A2A is the standard to consider.

---

### Reasoning Frameworks

**Chain-of-Thought (CoT):** The model generates **intermediate reasoning steps** ("think step-by-step") before the final answer. No tool use—just internal logic. Use when you need interpretability or multi-step reasoning without external data.

**ReAct (Reason + Act):** Combines **reasoning** with **tool use** in a loop. Each turn is either a *Thought* (what to do next), an *Action* (tool name + args), or an *Observation* (tool result). The model keeps going until it can give a final answer.

| Phase | What Happens |
|-------|--------------|
| **1. Reasoning** | Agent analyzes task, selects tools |
| **2. Acting** | Agent executes selected tool |
| **3. Observation** | Agent receives tool output |
| **4. Repeat** | Agent reasons from the observation, then next Thought/Action or final answer |

```
┌─────────────────────────────────────────────────────────────────┐
│                    ReAct LOOP (example)                           │
│   User: "What's the status of order #123? Can I get a refund?"   │
│      Thought: I need to look up order #123 first.                 │
│      Action: get_order_status(order_id="123")                    │
│      Observation: { "status": "delivered", "date": "2024-01-15" }│
│      Thought: Delivered. User asked about refund. Check policy.   │
│      Action: search_knowledge_base(query="refund policy")         │
│      Observation: "Refunds within 30 days of delivery..."         │
│      Thought: I have enough. Compose answer.                      │
│      Answer: "Order #123 was delivered Jan 15. Our policy..."     │
└─────────────────────────────────────────────────────────────────┘
```

**Aha:** ReAct makes the reasoning **visible** (Thought) and **grounded** (Action → Observation). The model can’t wander off; each step is either "I think…" or "I do X" followed by real tool output. That reduces hallucination in tool use because the next thought is conditioned on actual observations.

### Agent Design Patterns

**When to use which:** Start with **Single Agent** (one LLM + all tools). Add **Multi-Agent** or **Hierarchical** when one agent can't handle the diversity of tasks or when you want specialists (e.g. research vs writing vs coding) or clearer separation of concerns.

---

**1. Single Agent Pattern**

One LLM handles the entire conversation and has access to all tools. The model decides when to call which tool.

```
   User ──► LLM (orchestrator) ──► Tool A, Tool B, Tool C
              ▲         │
              └─────────┘  (loop until done)
```

- ✅ Simple, low latency, easy to debug
- ❌ Limited capabilities, may struggle with very complex or diverse tasks
- *Best for*: Simple use cases, single domain (e.g. support bot with KB + CRM + ticketing)

---

**2. Multi-Agent Pattern**

Multiple specialized agents, each with its own tools. **There is no single "boss."** Agents can **hand off** to each other (e.g. Agent A finishes and passes to B), **work in parallel** (A, B, C run at once and someone aggregates), or **negotiate** who does what. Control and flow are **distributed**—each agent or a lightweight router decides the next step, not one central planner.

```
   User ──► [Agent A] ←──► [Agent B] ←──► [Agent C] ──► combined result
              │               │               │
           Tools A        Tools B        Tools C
        (peer-to-peer handoffs or parallel, then aggregate)
```

- ✅ Specialists, parallel execution, modular, flexible routing
- ❌ Coordination logic lives in handoffs/aggregation; can be harder to reason about
- *Best for*: Domains where agents **collaborate as peers** (e.g. research agent + writing agent + fact-check agent that hand off or run in parallel; no one agent "owns" the plan)

---

**3. Hierarchical Pattern (Supervisor/Manager)**

**One supervisor** agent receives the user request, **owns the plan**, and **delegates** to specialist agents. Specialists do the work and **report back only to the supervisor**; they do **not** talk to each other. The supervisor decides the next step, assigns it, waits for the result, then repeats or synthesizes the final answer. Control and flow are **centralized** in the supervisor.

```
   User ──► Supervisor (LLM) ──► "Do step 1" ──► Specialist A ──► result ──► Supervisor
                    │
                    ├──► "Do step 2" ──► Specialist B ──► result ──► Supervisor
                    │
                    └──► synthesize ──► Answer
```

- ✅ Clear ownership of the plan, easier to debug and reason about, scalable workflow
- ❌ Supervisor is a bottleneck; more latency than flat handoffs when steps are independent
- *Best for*: Workflows with a **fixed or predictable sequence** (e.g. research → draft → review → publish) where one "conductor" should own the plan

---

**Multi-Agent vs Hierarchical: Clear distinction**

| Aspect | Multi-Agent | Hierarchical |
|--------|-------------|--------------|
| **Who decides the plan?** | Distributed: agents hand off, or a router chooses; no single owner | **One supervisor** owns the plan and assigns steps |
| **Who do specialists talk to?** | Each other (handoffs) or an aggregator; flow is peer-to-peer or fan-out | **Only the supervisor**; specialists do not talk to each other |
| **Control shape** | **Flat** or **peer-to-peer**: many agents, shared or emergent coordination | **Tree**: one node (supervisor) at the top, specialists as children |
| **Flow** | Emergent (handoffs, parallel, negotiate) | **Top-down**: Supervisor → assign step → Specialist → result → Supervisor |
| **When to use** | You want **peers** that hand off or run in parallel and someone (or the group) aggregates | You want **one conductor** that plans and delegates in sequence or in a clear DAG |

**Aha:** **Multi-agent** = "several agents, no single boss; they hand off or run in parallel." **Hierarchical** = "one boss (supervisor) that assigns tasks to specialists and gets results back; specialists don’t talk to each other." Use multi-agent when control should be shared or emergent; use hierarchical when one agent should own the plan and delegate.

---

**4. Additional Patterns**

Beyond single-, multi-, and hierarchical agents, three common *orchestration shapes* show up in production: stages in a fixed order, independent experts run in parallel, and adversarial roles that argue before a judge. Use these when the task has a natural flow (sequence), benefits from multiple viewpoints (fan-out), or must be stress-tested (debate).

---

**1. Sequential Pipeline**

**What it is:** A fixed chain of steps, A → B → C. Each stage consumes the prior stage's output and produces input for the next. No parallelism within the pipeline; order is part of the design (e.g. outline before draft, draft before edit).

**How it works:** One agent or model run handles each step. Outputs are passed as context or artifacts to the next. Handoffs are explicit (e.g. "outline," "draft," "edited_draft"). Failures or rewinds usually mean restarting from the failing step or the beginning, depending on your design.

**When to use:** **Content creation** (outline → draft → edit), **ETL-style** flows (extract → transform → load), or any process where step N truly depends on step N−1 and there's no benefit from running steps in parallel.

```
  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │Outline  │ ──▶ │ Draft   │ ──▶ │ Edit    │ ──▶ output
  └─────────┘     └─────────┘     └─────────┘
       A               B               C
```

---

**2. Parallel Fan-out**

**What it is:** One query (or task) is sent to **multiple agents or tools** at once; each runs independently. A separate **aggregator** (or router) collects their outputs and merges them into one answer or decision.

**How it works:** Fan-out: duplicate the request to A, B, C (and optionally more). No agent waits on another during the parallel phase. Aggregate: combine results via another LLM call (e.g. "synthesize these three analyses") or a rule (e.g. majority vote, weighted average). Latency is dominated by the slowest branch plus aggregation, not the sum of all branches.

**When to use:** **Research** or **multi-perspective analysis** (e.g. legal, market, technical views in parallel), **ensemble** answers (e.g. multiple retrieval strategies or models), or whenever you want **diversity** then **reconciliation** in one round.

```
       Query
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
  ┌───┐ ┌───┐ ┌───┐
  │ A │ │ B │ │ C │   (parallel)
  └─┬─┘ └─┬─┘ └─┬─┘
    └─────┼─────┘
          ▼
     Aggregate ──▶ final answer
```

---

**3. Debate / Adversarial**

**What it is:** Two (or more) **adversarial roles** argue opposite sides (e.g. Pro vs Con, attacker vs defender). A **judge** (or meta-agent) reads the debate and produces the final decision or output. The goal is to surface objections and reduce overconfidence.

**How it works:** Pro and Con (or Red / Blue) each get the same task and constraints; they may see each other's replies in one or more rounds. The judge receives the full transcript and possibly the original query, then outputs the chosen stance, a synthesis, or a "no decision" with reasons. You can cap rounds (e.g. 1–2) to control cost and latency.

**When to use:** **High-stakes decisions** (e.g. approvals, audits, policy), **red teaming** (stress-test an idea or policy before release), or when you want the system to **explicitly consider counterarguments** instead of one-shot answers.

```
  ┌─────┐                    ┌─────┐
  │ Pro │ ──── argue ───────▶│Judge│
  └─────┘                    └──┬──┘
       ▲                        │
       └── argue ──────────────┘
  ┌─────┐
  │ Con │
  └─────┘
```

---

**Quick reference**

| Pattern | Architecture | Use Case |
|---------|--------------|----------|
| **Sequential Pipeline** | A → B → C (fixed order) | Content creation (outline → draft → edit), ETL-style flows |
| **Parallel Fan-out** | Query → [A, B, C] → Aggregate | Research, multi-perspective analysis, ensembles |
| **Debate/Adversarial** | Pro vs Con → Judge | High-stakes decisions, red teaming, counterargument stress-test |

**Aha:** Single agent = one brain, many tools. Multi-agent = many brains, each with its own tools; you need handoffs. Hierarchical = one brain that delegates; specialists don't talk to each other directly.

### Context Engineering

**The Problem**: As agents run longer, context (chat history, tool outputs, documents) **explodes**. Simply using larger context windows is not a scaling strategy.

**Aha:** More context isn’t always better. Models often **underuse** the middle of long prompts ("lost in the middle"). So putting the most important instructions or retrieval at the **start and end** of the context, and keeping working context small and focused, improves both quality and cost. Tiered context (working / session / memory / artifacts) is how you scale *usage* of context without scaling *size* of every call.

**The Three-Way Pressure on Context:**

| Pressure | Problem |
|----------|---------|
| **Cost & latency spirals** | Cost and time-to-first-token grow with context size |
| **Signal degradation** | Irrelevant logs distract the model ("lost in the middle") |
| **Physical limits** | RAG results and traces eventually overflow even largest windows |

**The Solution: Tiered Context Model**

Keep **working context** (the prompt for this turn) small and focused. Push durable state into **Session** (conversation log), **Memory** (searchable, cross-session), and **Artifacts** (large files by reference, not pasted). Put the most important instructions and retrieval at the **start and end** of the prompt.

```
┌─────────────────────────────────────────────────────────────────┐
│                    TIERED CONTEXT                                 │
│   WORKING (this turn)   Session (this convo)   Memory (long-term) │
│   ┌──────────────┐      ┌──────────────────┐  ┌──────────────┐  │
│   │ System + key │      │ Chat history      │  │ Searchable   │  │
│   │ docs + query │      │ + tool I/O        │  │ facts, prefs │  │
│   └──────────────┘      └──────────────────┘  └──────────────┘  │
│   ARTIFACTS: Large files addressed by name, not pasted           │
└─────────────────────────────────────────────────────────────────┘
```

| Layer | Purpose | Lifecycle |
|-------|---------|-----------|
| **Working Context** | Immediate prompt for this call | Ephemeral |
| **Session** | Durable log of events | Per-conversation |
| **Memory** | Long-lived searchable knowledge | Cross-session |
| **Artifacts** | Large files | Addressed by name, not pasted |

**Multi-Agent Context Scoping:** When one agent delegates to another, control what the sub-agent sees. **Agents as Tools** = sub-agent gets only the instructions and inputs you pass. **Agent Transfer** = sub-agent gets a configurable view over Session (e.g. last N turns).

| Pattern | Description |
|---------|-------------|
| **Agents as Tools** | Sub-agent sees only specific instructions and inputs |
| **Agent Transfer** | Sub-agent inherits a configurable view over Session |

---

## 5. LLM Evaluation & Quality

**What "knowledge quality" means here:** For LLM and RAG systems, quality is **groundedness** (is the answer supported by the context?), **relevance** (does it address the question?), and **retrieval quality** (did we fetch the right chunks?). You rarely have gold labels for every request, so evaluation mixes **reference-free** automated metrics (e.g. faithfulness, relevancy) with **sampled human review** to calibrate and catch edge cases. This section is tool-first: each concept is tied to frameworks you can run today.

---

### Evaluation Frameworks & Metrics

**RAGAS** (Python: `pip install ragas`) is the de facto open-source choice for **reference-free** RAG evaluation. You pass a dataset of `(user_input, retrieved_contexts, response)` plus optional `reference`; RAGAS runs LLM-as-judge and embedding-based metrics and returns scores. Used by LangChain, LlamaIndex, and LangSmith integrations.

| Metric | What It Measures | How (in RAGAS) | Tool |
|--------|------------------|----------------|------|
| **Faithfulness** | Is response grounded in context? | LLM extracts claims → checks each against retrieved docs | `ragas.metrics.Faithfulness` |
| **Answer Relevancy** | Does answer address the question? | Inverse of LLM-generated “counterfactual” questions needed to recover answer | `ragas.metrics.AnswerRelevancy` |
| **Context Precision** | Are relevant docs ranked above noise? | Ground-truth relevant items ranked high → higher score | `ragas.metrics.ContextPrecision` (needs ground truth) |
| **Context Recall** | Did we retrieve what we need? | Overlap between answer-supporting context and retrieved context; or vs. reference | `ragas.metrics.ContextRecall` / `LLMContextRecall` |

**Practical RAGAS workflow:** Build a list of dicts with `user_input`, `retrieved_contexts`, `response`, and optionally `reference`. Load into `EvaluationDataset.from_list(dataset)`, then call `evaluate(dataset=..., metrics=[Faithfulness(), AnswerRelevancy(), ...], llm=evaluator_llm)`. Use a **different** LLM for evaluation than for generation to reduce self-consistency bias. See [RAGAS docs](https://docs.ragas.io/en/stable/getstarted/rag_eval/).

**Other tools:**

- **LangSmith** (LangChain): Predefined RAG evaluators (correctness, relevance, groundedness), dataset runs, human annotation queues, and online feedback. Use `client.run_evaluator` or the LangSmith UI to run evals on logged runs. Strong when your stack is already LangChain.
- **Giskard** (Python: `pip install giskard`): RAG Evaluation Toolkit (RAGET)—testset generation, knowledge-base–aware tests, and scalar metrics. Good for “test-suite” style regression and CI.
- **Arize Phoenix** (Python: `pip install arize-phoenix`): Open-source LLM tracing + evals. Phoenix Evals include **hallucination**, relevance, toxicity; they run over OpenTelemetry traces. Use for production monitoring and “eval on sampled traffic.”
- **Braintrust** (Python: `braintrust`): `Eval()` / `EvalAsync()` over datasets; you define **scorers** (functions that score outputs). Fits custom logic and proprietary benchmarks.
- **TruLens**: Focus on “RAG triad” (context relevance, grounding, relevance) with minimal config; integrates with LlamaIndex and other frameworks.

---

### Hallucination Detection: Approaches & Tools

| Approach | What It Does | Accuracy | Latency | Tools / How |
|----------|--------------|----------|---------|-------------|
| **Self-consistency** | Sample N answers, check agreement | Moderate | High (N× calls) | Custom loop or Braintrust/Phoenix over multiple runs |
| **NLI / cross-encoder** | Entailment model: premise = context, hypothesis = claim | High | +50–100 ms | Sentence-transformers NLI, or Phoenix “groundedness”–style evals |
| **LLM-as-Judge** | “Is this claim supported by the context?” | High | +100–200 ms | **RAGAS** `Faithfulness`, **LangSmith** groundedness, **Phoenix** hallucination template, **Braintrust** custom scorer |
| **Specialized faithfulness models** | Fine-tuned “faithfulness vs. hallucination” judge | Highest | ~+50 ms | **Vectara FaithJudge** ([GitHub](https://github.com/vectara/FaithJudge)): benchmark + model for RAG QA/summarization; use when you need max agreement with human judgment |

**Practical tip:** In production, run **fast** checks inline (format, length, toxicity if you have a small classifier), and push **faithfulness / hallucination** to async jobs on a sample (e.g. 5–10%) using RAGAS or Phoenix so cost and latency stay bounded.

---

### How to Run Evaluation in Practice

1. **Offline / batch (before release or in CI)**  
   - **Data:** List of `(query, retrieved_contexts, response)` or `(query, response)`; optional `reference` for reference-based metrics.  
   - **Run:** RAGAS `evaluate()` on a dataset; or LangSmith “evaluate dataset”; or Braintrust `Eval(dataset, scorers=...)`.  
   - **Use:** Regressions, A/B on prompts or retrievers, and calibration of thresholds.

2. **Online / production (sampled)**  
   - **Data:** Log requests and responses (and retrieved contexts if RAG) to **LangSmith**, **Phoenix**, or your own store.  
   - **Run:** Periodic jobs (e.g. cron or queue) that pull a sample (e.g. 5%), run RAGAS or Phoenix evals, and write scores to a dashboard or alerting.  
   - **Use:** Drift detection, “did we build the right thing?” in the wild.

3. **Human loop**  
   - **Data:** Subset of production or offline examples (e.g. 100–500) with labels (good/bad, error type, etc.).  
   - **Tools:** **LangSmith** annotation queue, Label Studio, or internal tooling.  
   - **Use:** Calibrate automated metrics (“at what faithfulness score do humans usually approve?”), build training data for task-specific judges, and categorize failure modes.

**Aha:** You don’t need gold labels for every request. **Reference-free** metrics (RAGAS faithfulness, answer relevancy, Phoenix hallucination) answer “is this grounded?” and “does this match the question?” without human annotations. Use them on a sample in production, then a **small human-labeled set** to set thresholds and sanity-check.

---

### Production Evaluation Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                 EVALUATION PIPELINE                             │
│                                                                 │
│   Request → LLM Response                                        │
│                │                                                │
│                ├──► Real-time checks (< 50ms budget)             │
│                │    • Toxicity (e.g. Perspective API, small      │
│                │      classifier, or rule-based filters)         │
│                │    • Format validation (schema, length)         │
│                │    • Length limits                             │
│                │    Tools: in-process code, light model or API  │
│                │                                                │
│                ├──► Async evaluation (sampled, e.g. 5–10%)      │
│                │    • Faithfulness / grounding → RAGAS, Phoenix  │
│                │    • Hallucination → Phoenix evals, FaithJudge │
│                │    • Task-specific metrics → Braintrust, custom │
│                │    Tools: RAGAS, Phoenix, LangSmith, Braintrust  │
│                │                                                │
│                └──► Human evaluation (subset of async or batch) │
│                     • Quality ratings, error taxonomy            │
│                     • Calibrate automated score thresholds       │
│                     Tools: LangSmith annotation, Label Studio   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Not every request gets every metric. Use **tiered evaluation**—cheap checks inline, expensive ones (RAGAS, hallucination, custom scorers) on a **sample** and/or async, so latency and cost stay under control.

---

### Tools Quick Reference

| Tool | What It Does | When to Use |
|------|----------------|-------------|
| **RAGAS** | Reference-free RAG metrics (faithfulness, relevancy, context precision/recall) | Batch RAG evals, CI, offline benchmarks; Python-first |
| **LangSmith** | Evaluators, datasets, runs, human annotation | LangChain-based apps; need UI + queues + feedback |
| **Phoenix** | Tracing + evals (hallucination, relevance, toxicity) over OTLP | Production monitoring, eval-on-sampled-traffic |
| **Giskard** | RAG test suite, testset generation, scalar metrics | Regression and “test suite” style RAG evaluation |
| **Braintrust** | Custom scorers, `Eval`/`EvalAsync`, experiments | Proprietary benchmarks, custom logic, experiments |
| **FaithJudge** (Vectara) | Faithfulness/hallucination benchmark + model | High-stakes RAG; max agreement with human judgment |

---

### Evaluation data pipeline at scale

The metrics and tools above assume you have prediction data to evaluate. At scale, you need a **data pipeline**: predictions flow from the LLM → event stream → stream processor → evaluation/metrics layer and time-series store → dashboards and alerting. This is the *evaluation* pipeline (log predictions, run quality/safety/cost metrics); the *training* pipeline (user interactions → fine-tuning data) is §6.

**Use case: Production LLM evaluation system**

**Requirements:** Evaluate model performance continuously; track 100+ metrics (accuracy, latency, cost, safety); process 1M predictions/day; alert on degradation; support A/B testing.

```
┌─────────────────────────────────────────────────────────────────┐
│                  EVAL DATA PIPELINE (at scale)                  │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │    LLM       │────►│ Event Stream │────►│   Stream     │   │
│   │ Predictions  │     │ Pub/Sub or   │     │ Processor    │   │
│   └──────────────┘     │ Kinesis      │     └──────┬───────┘   │
│                        └──────────────┘            │           │
│                    ┌───────────────────────────────┼───────┐    │
│                    ▼                               ▼       ▼    │
│              ┌───────────┐                   ┌───────────────┐ │
│              │ Evaluation│                   │  Time-Series   │ │
│              │ (RAGAS,   │                   │  DB → Dashboards│
│              │ Phoenix…) │                   │  Alerting, A/B │ │
│              └───────────┘                   └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Sampling:** Full (100%) = complete visibility but costly; sampled (e.g. 10%) = cheaper, may miss rare errors; **smart (100% errors + sample successes)** = recommended—capture all failures, sample successes for stats.

**Frequency:** Real-time for latency/errors (user-facing); batch (hourly/daily) for quality/cost (expensive metrics); **hybrid** for most production.

**What to track:** Quality (task accuracy, ROUGE/BLEU, human eval), latency (P50/P95/P99), cost (tokens, model tier), safety (toxicity, jailbreak, bias).

---

## 6. GenAI Data Pipeline Architecture

**In the big picture** (see [GenAI System: Big Picture](#genai-system-big-picture-frontend-to-backend)), this is the **training-data pipeline**: the path from "users interacted with the system" to "we have clean, formatted examples for fine-tuning." It is *distinct* from the evaluation pipeline (§5), which moves *prediction* data into metrics and alerts. Here we focus on **collecting user interactions** (prompts, responses, feedback), processing them at scale, and producing training-ready datasets.

**T-shaped summary:** User interactions → event stream (Pub/Sub, Kinesis) → stream processor (Dataflow, etc.) → data lake and optionally feature store → training data prep (filter, dedupe, validate, format for fine-tuning). Deep dive below.

---

### Use Case: Design a Training Data Pipeline for Fine-Tuning

**Requirements:**
- Collect user interactions (prompts, responses, feedback)
- Process 10M examples/day
- Clean and prepare data for fine-tuning
- Support continuous data collection

**High-Level Design:**

```
┌─────────────────────────────────────────────────────────────────┐
│                  TRAINING DATA PIPELINE                         │
│                                                                 │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │    User      │────►│    Event     │────►│    Data      │   │
│   │ Interactions │     │  Collection  │     │  Processing  │   │
│   │              │     │  Pub/Sub     │     │  Dataflow    │   │
│   └──────────────┘     └──────────────┘     └──────┬───────┘   │
│                                                     │           │
│                        ┌────────────────────────────┤           │
│                        │                            │           │
│                        ▼                            ▼           │
│                  ┌───────────┐              ┌───────────────┐   │
│                  │ Data Lake │              │ Feature Store │   │
│                  │   (GCS)   │              │               │   │
│                  └─────┬─────┘              └───────────────┘   │
│                        │                                        │
│                        ▼                                        │
│                  ┌───────────────────────────────────────────┐ │
│                  │          Training Data Prep               │ │
│                  │  Filter, dedupe, validate, format         │ │
│                  │         for fine-tuning                   │ │
│                  └───────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Service Comparison

| Component | Google Cloud | AWS |
|-----------|--------------|-----|
| **Event Streaming** | Pub/Sub | Kinesis Data Streams |
| **Stream Processing** | Dataflow | Kinesis Analytics |
| **Data Lake** | Cloud Storage | S3 |
| **Data Warehouse** | BigQuery | Redshift |
| **Feature Store** | Vertex AI Feature Store | SageMaker Feature Store |
| **Training** | Vertex AI Training | SageMaker Training |
| **Orchestration** | Vertex AI Pipelines | SageMaker Pipelines |

---

## 7. Cost Optimization for GenAI Systems

**In the big picture** (see [GenAI System: Big Picture](#genai-system-big-picture-frontend-to-backend)), this is **how we keep inference affordable**: cost scales with tokens (input + output) and model tier, so optimization is about **reducing spend per request**—shorter prompts, caching, model routing, quantization, and when relevant fine-tuning ROI. *Throughput* and *capacity* are in §8 Scalability; here we focus on *cost per request*.

**T-shaped summary:** Cost = f(tokens, model). Levers: prompt optimization, response/prompt caching, routing easy queries to smaller models, quantization, and continuous batching (better GPU use → same throughput with fewer machines). Deep dive below.

---

### Token-Based Cost Model

**Cost Components:**
- **Input tokens**: Tokens in prompt (including context)
- **Output tokens**: Generated tokens (typically 2-4x more expensive)
- **Model tier**: Different models have different costs

**Aha:** GenAI cost scales with **length**, not just request count. A 10× longer prompt or answer can mean ~10× cost per call. So trimming context, caching prefixes, and routing easy queries to smaller models all directly lower spend.

**Example Calculation:**

```
Model: Gemini Pro
Input: $0.000125 per 1K tokens
Output: $0.0005 per 1K tokens

Request:
- Input: 1,000 tokens
- Output: 500 tokens

Cost = (1,000 / 1,000) × $0.000125 + (500 / 1,000) × $0.0005
     = $0.000125 + $0.00025
     = $0.000375 per request

At 1M requests/day: $375/day = $11,250/month
```

### Optimization Strategies

**1. Prompt Optimization**

| Technique | Savings | Trade-off |
|-----------|---------|-----------|
| Shorter prompts | 20-40% input tokens | May lose context |
| Fewer examples | 50-200 tokens/example | May reduce quality |
| Prompt compression | Variable | Compression cost vs savings |

**Few-shot sweet spot**: 2-3 examples usually sufficient. Research shows diminishing returns after 3 examples—the model has learned the pattern.

**2. Caching Strategy**

| Strategy | Hit Rate | Savings | Best For |
|----------|----------|---------|----------|
| Prompt caching | High for prefixes | 2-5x speedup | System prompts |
| Response caching | 10-30% | 100% for hits | FAQ systems |
| Semantic caching | 30-50% | Varies | Q&A systems |

**3. Model Selection (Tiered Strategy)**

| Model | Cost | Quality | Use For |
|-------|------|---------|---------|
| **Large (GPT-4, Gemini Ultra)** | $0.03-0.06/1K output | Best | Complex reasoning |
| **Medium (GPT-3.5, Gemini Pro)** | ~$0.002/1K output | Good | Most production tasks |
| **Small (Gemini Flash)** | ~$0.001/1K output | Basic | Simple, high-volume |

**Model Routing Strategies:**

| Strategy | How It Works | Savings |
|----------|--------------|---------|
| **Routing** | Classify query → send to single optimal model | 40-60% |
| **Cascading** | Start small → escalate to larger if low confidence | 50-80% |
| **Cascade Routing** | Combines both: route + escalation | Best cost/quality |

```
Query → Classifier → Simple? → Small Model → Done
                         │
                         └──► Complex? → Large Model → Done

OR (Cascading):

Query → Small Model → Confident? → Return
              │
              └──► Low confidence → Large Model → Return
```

**Quality Estimation**: The key to routing—use a small classifier or confidence scores to predict which model can handle the query.

**Aha:** Routing and cascading both assume "hard" and "easy" queries. If you can **predict** hardness (e.g. by query length, intent, or a tiny classifier), you send easy ones to small/cheap models and reserve the big model for the rest. The leverage comes from that prediction being cheap and reasonably accurate.

**4. Fine-tuning ROI**

- **Upfront cost**: $100-1000s
- **Break-even**: If fine-tuning costs $1000 and saves $0.001 per request, break-even at 1M requests
- **Benefits**: Better quality for domain, can use smaller base model

**5. Quantization**

Reducing numerical precision shrinks model size and speeds inference. **FP32** (32-bit float), **FP16** (16-bit), **INT8** (8-bit integer), **INT4** (4-bit) are common levels.

| Precision | Memory Reduction | Quality Loss |
|-----------|-----------------|--------------|
| FP32 → FP16 | 2x | Minimal |
| FP16 → INT8 | 4x | Some |
| INT8 → INT4 | 8x | Significant |

**Why FP16 is safe**: Modern **GPUs** (graphics processing units) have Tensor Cores optimized for FP16. Quality loss is minimal (<1%) but memory/cost savings are significant.

**Aha:** Weights don’t need 32-bit precision for good answers; most signal lives in a smaller range. Quantization **compresses** that range (FP32→FP16→INT8→INT4). You trade a little quality for large memory and speed gains. FP16 is the first step almost everyone takes because hardware is built for it and the drop is tiny.

**6. Continuous Batching**

- Static batching: 40–60% GPU utilization
- Continuous batching: 80–95% GPU utilization
- **Result**: 2–3× higher throughput → fewer machines for the same load (cost and scale). Throughput/parallelism patterns (model parallelism, pipeline parallelism) are in §8.

---

## 8. Scalability Patterns for GenAI

**In the big picture** (see [GenAI System: Big Picture](#genai-system-big-picture-frontend-to-backend)), this is **how we serve more load**: the LLM layer is GPU-heavy and stateful (KV cache), so scaling is about **throughput and capacity**—horizontal replication, model/pipeline parallelism, and caching that increases effective req/s. *Cost per request* is in §7; here we focus on *requests per second* and *utilization*.

**T-shaped summary:** Levers: stateless serving (more replicas), model parallelism (split layers across GPUs), pipeline parallelism (different layers on different GPUs), and caching (KV cache for prefixes, response cache for identical/similar queries). Deep dive below.

---

### Horizontal Scaling

**Challenge**: LLM inference is GPU-intensive and stateful (KV cache).

**Solutions:**

| Pattern | Description | Trade-off |
|---------|-------------|-----------|
| **Stateless Serving** | Load balancer → Multiple LLM servers | Higher memory (each server has full model) |
| **Model Parallelism** | Split model across GPUs | Communication overhead |
| **Pipeline Parallelism** | Different GPUs handle different layers | Better utilization |

**Model Parallelism Visual:**

```
Input → GPU 1 (Layers 1-10) → GPU 2 (Layers 11-20) → GPU 3 (Layers 21-30) → Output
```

### Caching Strategies for Scale

*Cost* impact of caching is in §7; here we focus on **throughput** impact: same hardware serves more requests when prefixes or responses are reused.

| Strategy | Throughput / latency impact | Best For |
|----------|-----------------------------|----------|
| Prompt caching (KV cache) | 2–3× effective throughput for repeated prefixes | System prompts, long context |
| Response caching | Near-instant for cache hits; frees GPU for other requests | Identical or near-identical queries |
| Semantic caching | Higher hit rate → more requests served from cache | Similar queries (e.g. Q&A) |

---

## 9. Monitoring & Observability for GenAI

**In the big picture** (see [GenAI System: Big Picture](#genai-system-big-picture-frontend-to-backend)), this is **how we observe the system**: metrics, traces, and drift detection across the request path and the evaluation/training pipelines. Quality metrics and eval pipeline are in §5; here we focus on **what to track** and **which platform services** support it.

**T-shaped summary:** Track quality (task accuracy, safety), performance (latency, throughput), cost (tokens, model tier), reliability (errors, timeouts), and safety (toxicity, jailbreak). Use Cloud Monitoring / CloudWatch, logging, tracing (Trace / X-Ray), and model monitoring for drift. Deep dive below.

---

### Key Metrics to Track

| Category | Metrics |
|----------|---------|
| **Quality** | Task accuracy, ROUGE/BLEU, human evaluation |
| **Performance** | P50/P95/P99 latency, throughput, tokens/second |
| **Cost** | Cost per request, token usage, model tier breakdown |
| **Reliability** | Error rate, timeout rate, availability |
| **Safety** | Toxicity score, jailbreak attempts, bias detection |

### Platform Services

| Function | Google Cloud | AWS |
|----------|--------------|-----|
| **Metrics** | Cloud Monitoring, Vertex AI Monitoring | CloudWatch |
| **Logging** | Cloud Logging | CloudWatch Logs |
| **Tracing** | Cloud Trace | X-Ray |
| **Drift Detection** | Vertex AI Model Monitoring | SageMaker Model Monitor |

---

## 10. Security & Guardrails

**In the big picture** (see [GenAI System: Big Picture](#genai-system-big-picture-frontend-to-backend)), this is **how we protect the system**: inputs (prompt injection, jailbreak, PII), outputs (harmful content, PII leakage), and access (IAM, API keys). Guardrails sit *around* the request path—input checks before the LLM, output checks after—and work with HTTP-level protections (Cloud Armor, WAF) and data protection (DLP).

**T-shaped summary:** Threats: direct/indirect prompt injection, data leakage, jailbreaking, unauthorized access. Mitigations: input/output guardrails, spotlighting, least-privilege tools, Model Armor (or Bedrock Guardrails). Use defense-in-depth: gateway → guardrails → LLM → guardrails → response. Deep dive below.

---

### Key Security Concerns

**Aha:** LLMs take natural language as input, so **any** user text can be an attempt to override instructions ("Ignore previous instructions…"). Guardrails and defense-in-depth exist because you can't whitelist "good" prompts—you have to detect and constrain *malicious* or out-of-scope intent at the boundary.

| Threat | Risk | Mitigation |
|--------|------|------------|
| **Direct Prompt Injection** | User injects malicious instructions | Input validation, guardrails |
| **Indirect Prompt Injection** | Hidden instructions in external content | Content isolation, spotlighting |
| **Data Leakage** | Training data memorization, **PII** (personally identifiable information) in outputs | Output filtering, **DLP** (data loss prevention) |
| **Jailbreaking** | Bypassing safety controls | Multi-layer defense, red teaming |
| **Access Control** | Unauthorized model access | **IAM** (identity and access management), API keys, least privilege |

### Prompt Injection Defense-in-Depth

| Layer | Technique | Description |
|-------|-----------|-------------|
| **Input** | Spotlighting | Clearly delimit user input vs system prompt |
| **Input** | Input validation | Regex, blocklists, encoding detection |
| **Input** | Guardrails check | Detect injection attempts before LLM |
| **Processing** | Least privilege | Limit tools/data agent can access |
| **Output** | Guardrails check | Validate output aligns with user intent |
| **Output** | PII filtering | Detect/redact sensitive data |

### Guardrails Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   GUARDRAILS PIPELINE                           │
│                                                                 │
│   User Input                                                    │
│       │                                                         │
│       ▼                                                         │
│   ┌────────────────┐                                           │
│   │ INPUT GUARDRAIL│  • Prompt injection detection             │
│   │                │  • Jailbreak detection                    │
│   │                │  • PII detection                          │
│   │                │  • Content policy check                   │
│   └───────┬────────┘                                           │
│           │                                                     │
│     Block ├──► Return error                                    │
│           │                                                     │
│           ▼                                                     │
│   ┌────────────────┐                                           │
│   │      LLM       │                                           │
│   └───────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│   ┌────────────────┐                                           │
│   │OUTPUT GUARDRAIL│  • Hallucination check                    │
│   │                │  • Response relevancy                     │
│   │                │  • PII in output                          │
│   │                │  • Harmful content                        │
│   └───────┬────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│   User Response                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Tool Call Validation** (for agents):
- **Pre-flight**: Validate tool call aligns with user's request before execution
- **Post-flight**: Validate returned data before showing to user

### Model Armor (Google Cloud)

Model Armor is Google Cloud's service for real-time input/output filtering on LLM traffic. It addresses threats that traditional **WAFs** (web application firewalls) can't catch—specifically **prompt injection** and **sensitive data disclosure** at the semantic level.

**What Model Armor Catches vs Cloud Armor:**

| Threat | Cloud Armor | Model Armor |
|--------|-------------|-------------|
| SQL injection in HTTP | ✅ | ❌ (not its job) |
| DDoS / rate limiting | ✅ | ❌ |
| **Prompt injection** | ❌ | ✅ |
| **Jailbreak attempts** | ❌ | ✅ |
| **PII in LLM output** | ❌ | ✅ |

**Use both for production deployments—they protect different attack surfaces.**

### Defense Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                  SECURE AGENT ARCHITECTURE                      │
│                                                                 │
│   User Request                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌───────────────┐                                            │
│   │ Cloud Armor   │  HTTP-level: DDoS, rate limiting           │
│   └───────┬───────┘                                            │
│           │                                                     │
│           ▼                                                     │
│   ┌───────────────┐                                            │
│   │  API Gateway  │  Auth, authorization (IAM)                 │
│   └───────┬───────┘                                            │
│           │                                                     │
│           ▼                                                     │
│   ┌───────────────┐                                            │
│   │ Model Armor   │  Input: prompt injection, PII              │
│   │   (Input)     │                                            │
│   └───────┬───────┘                                            │
│           │                                                     │
│           ▼                                                     │
│   ┌───────────────┐                                            │
│   │  LLM / Agent  │                                            │
│   └───────┬───────┘                                            │
│           │                                                     │
│           ▼                                                     │
│   ┌───────────────┐                                            │
│   │ Model Armor   │  Output: harmful content, PII              │
│   │   (Output)    │                                            │
│   └───────┬───────┘                                            │
│           │                                                     │
│           ▼                                                     │
│   User Response                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Compliance Considerations

| Regulation | Key Requirements |
|------------|------------------|
| **GDPR** (General Data Protection Regulation) | Right to explanation, data deletion, privacy by design |
| **HIPAA** (Health Insurance Portability and Accountability Act) | Healthcare data protection, audit logging |
| **PCI-DSS** (Payment Card Industry Data Security Standard) | Payment data security, no storage of card numbers |

### Security Stack Summary

| Layer | Google Cloud | AWS |
|-------|--------------|-----|
| **LLM Security** | Model Armor | Bedrock Guardrails |
| **HTTP Security** | Cloud Armor | WAF (web application firewall) |
| **Data Protection** | Cloud DLP (data loss prevention) | Macie |
| **Secrets** | Secret Manager | Secrets Manager |
| **Network** | VPC (virtual private cloud) Service Controls | VPC |
| **Access** | IAM (identity and access management) | IAM |
| **Audit** | Cloud Audit Logs | CloudTrail |

---

## 11. Real-World Examples: Applying the Stack

This section comes **after** all core concepts (§1–§10) so you can apply them. Each example states the **problem**, the **concepts** from this guide that apply, and a **concrete solution** using specific stacks: **LangChain** / **LlamaIndex** (orchestration, RAG, agents), **Google (Vertex AI)** or **AWS (Bedrock)**, and **open source** (vLLM, RAGAS, Phoenix, etc.). Use these as blueprints for "how would I build this with real tools?"

---

### Example 1: Code Generation Assistant (like GitHub Copilot)

**Problem:** In-IDE completions that understand the codebase, respect privacy, and run with low latency.

**Concepts:** §1 (LLM serving / model routing), §2 (RAG for code context), §4 (single agent + tools), §7 (cost: smaller model for completions, routing by complexity).

**Concrete solution:**

- **Orchestration + RAG:** **LangChain** or **LlamaIndex** to build a "code context" pipeline: embed workspace chunks (or AST-based chunks), retrieve on cursor context, format as prefix for the model. Use **LlamaIndex** `CodeIndex` / doc split by language or **LangChain** `RecursiveCharacterTextSplitter` + vector store (e.g. Chroma, open source).
- **LLM:** **Vertex AI Codey** (Google) or **Amazon CodeWhisperer** / **Bedrock** (AWS) for code-native APIs; or **open source** (**CodeLlama**, **StarCoder**) behind **vLLM** for self-hosted, low-latency completion.
- **Evaluation:** **RAGAS** or **LangSmith** on a sample of (prompt, context, completion) for relevance and correctness; **Phoenix** for production traces and latency.
- **Guardrails:** Input/output length limits, optional PII/secret filters (e.g. **Guardrails AI**, **NeMo Guardrails**), or **Bedrock Guardrails** / **Model Armor** if on AWS/Google.

**Stack snapshot:** LangChain/LlamaIndex (RAG + routing) + Vertex Codey or Bedrock + vLLM (optional) + RAGAS/LangSmith/Phoenix (eval) + guardrails.

---

### Example 2: Customer Service Chatbot with RAG and Tools

**Problem:** Chat that answers from internal docs, checks orders/tickets via tools, and escalates to humans when needed.

**Concepts:** §2 (RAG: knowledge base), §4 (agent with tools, escalation as a "tool"), §5 (eval: faithfulness, relevancy), §10 (guardrails, PII).

**Concrete solution:**

- **Orchestration + agent:** **LangChain** `create_react_agent` or **LlamaIndex** `ReActAgent` with tools: RAG retriever (knowledge base), "check order" (API), "create ticket" (CRM API), "escalate" (handoff to human queue). Use **MCP** or custom tool schemas so the agent can call backend APIs.
- **RAG:** **Vertex AI RAG Engine** (Google) or **Bedrock Knowledge Bases** (AWS) for managed ingestion + retrieval; or **LangChain** + **Chroma** / **Pinecone** + **OpenAI** or **Cohere** embeddings (open / API). Apply chunking and reranking from §2.
- **LLM:** **Vertex AI** (Gemini) or **Bedrock** (Claude, Llama) for conversation and tool use.
- **Evaluation:** **RAGAS** (faithfulness, answer relevancy) on logged (query, context, response); **LangSmith** for dataset runs and human annotation queues.
- **Security:** **Bedrock Guardrails** or **Model Armor** for input/output; scope tools with IAM/least privilege; filter PII in tool *outputs* before they reach the model or user.

**Stack snapshot:** LangChain/LlamaIndex (agent + tools) + Vertex RAG Engine or Bedrock Knowledge Bases + Vertex/Bedrock LLM + RAGAS/LangSmith (eval) + Model Armor/Bedrock Guardrails.

---

### Example 3: Content Generation Platform (research → draft → grounding)

**Problem:** Multi-step content: research from web, generate draft, fact-check against sources, then SEO and multi-format output.

**Concepts:** §4 (sequential pipeline: research → generation → grounding → SEO), §2 (RAG/grounding for fact-check), §5 (faithfulness eval), §7 (cost: model routing for easy vs hard steps).

**Concrete solution:**

- **Orchestration:** **LangChain** `SequentialChain` or a custom DAG: (1) research step = tool to **Google Search** or **Tavily** (or Vertex Search); (2) generation = LLM with research as context; (3) grounding = LLM or **Vertex AI grounding** / **Bedrock** retrieval + NLI-style check; (4) SEO = templates or a small LLM call. This is the "sequential pipeline" from §4 Additional Patterns.
- **LLM:** **Vertex AI** (Gemini) or **Bedrock** (Claude). Use **routing** (§7): e.g. Gemini Flash for research/summary, Gemini Pro for final draft.
- **Grounding:** **Vertex AI grounding with Google Search** or **Bedrock** retrieval + cite-check; or **open source**: RAG pipeline + **RAGAS** faithfulness on (claim, source) samples.
- **Evaluation:** **RAGAS** faithfulness and relevancy on (brief, sources, draft); **LangSmith** or **Braintrust** for A/B on prompts and model choices.

**Stack snapshot:** LangChain (sequential pipeline + tools) + Vertex/Bedrock LLMs + Vertex grounding or RAG + RAGAS (eval) + optional Giskard for regression tests.

---

### Cross-example takeaways

| Concern | Tools to reach for |
|--------|--------------------|
| **Orchestration (RAG, agents, pipelines)** | LangChain, LlamaIndex |
| **Managed RAG / embeddings** | Vertex RAG Engine, Bedrock Knowledge Bases |
| **LLM hosting** | Vertex AI (Codey, Gemini), Bedrock (Claude, CodeWhisperer, etc.), or vLLM for self-hosted |
| **Evaluation (reference-free)** | RAGAS (batch), LangSmith (datasets + humans), Phoenix (traces + evals) |
| **Guardrails** | Model Armor (Google), Bedrock Guardrails (AWS), Guardrails AI / NeMo (open source) |

---

## Resources

### Books

- **Building LLM Applications for Production** by Huyen, Chip
- **Designing Machine Learning Systems** by Chip Huyen
- **Designing Data-Intensive Applications** by Martin Kleppmann

### Online

- [vLLM Documentation](https://docs.vllm.ai/) - High-throughput LLM serving
- [RAGAS Documentation](https://docs.ragas.io/) - Reference-free RAG evaluation (faithfulness, relevancy, context metrics)
- [LangSmith Evaluation](https://docs.smith.langchain.com/evaluation) - Evaluators, datasets, human annotation
- [Arize Phoenix](https://phoenix.arize.com/) - LLM tracing and evals (hallucination, relevance, toxicity)
- [Giskard RAG Toolkit](https://docs.giskard.ai/en/stable/reference/rag-toolset/) - RAG test suite and testset generation
- [Braintrust Evaluate](https://www.braintrust.dev/docs/evaluation) - Custom scorers and experiments
- [Vectara FaithJudge](https://github.com/vectara/FaithJudge) - Faithfulness/hallucination benchmark and model
- [LangChain Documentation](https://docs.langchain.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [OpenAI Guardrails](https://openai.github.io/openai-guardrails-python/)
- [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) - Standard for tools and context to LLMs
- [A2A (Agent-to-Agent Protocol)](https://google.github.io/A2A/) - Standard for agent-to-agent communication

### Google Cloud Documentation

- [Vertex AI Generative AI](https://cloud.google.com/vertex-ai/generative-ai/docs/overview)
- [Vertex AI Agent Builder](https://cloud.google.com/vertex-ai/docs/agent-builder/overview)
- [Vertex AI RAG Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/overview)
- [Model Armor](https://cloud.google.com/security/products/model-armor)

### AWS Documentation

- [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/)
- [Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/)
- [Bedrock Agents](https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html)
- [Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html)

### Practice

- Build real GenAI applications
- Experiment with different model sizes and costs
- Practice with RAG systems and agents

---

## Quick Reference

### What FAANG Interviewers Evaluate

| Dimension | What They Test |
|-----------|----------------|
| **LLM Awareness** | Token limits, context windows, model types, pricing models |
| **Architectural Reasoning** | How retrieval, prompt logic, post-processing connect |
| **Cost-Latency Tradeoffs** | Balancing inference cost, response latency, quality |
| **Safety & Governance** | Reliable outputs, guardrails, compliance |
| **Observability** | Handling non-deterministic outputs, failure modes |

### Interview Framework (45-min structure)

**1. Clarify Requirements (5-10 min)**
- Token budget and latency targets
- Quality requirements (hallucination tolerance)
- Cost constraints (per-token, monthly budget)
- Safety requirements (compliance, content filtering)

**2. High-Level Architecture (10-15 min)**
- Draw components: API gateway → orchestration → LLM → post-processing
- Show data flow and identify APIs
- Include: RAG, caching, model routing

**3. Deep Dive (15-20 min)**
- RAG design: chunking, embedding, retrieval, reranking
- Model selection and routing strategy
- Evaluation and observability approach
- Security layers (guardrails, Model Armor)

**4. Bottlenecks & Trade-offs (5-10 min)**
- KV cache memory management
- Quality vs cost (model size, routing)
- Latency vs throughput (batching)
- Single vs multi-agent complexity

### Key Trade-offs to Articulate

| Decision | Option A | Option B |
|----------|----------|----------|
| RAG vs Fine-tuning | Fresh data, per-query cost | Behavioral change, upfront cost |
| Large vs Small Model | Higher quality | Lower cost, faster |
| Dense vs Hybrid Search | Semantic matching | + Keyword precision |
| Single vs Multi-Agent | Simpler, faster | More capable, modular |
| Sync vs Async Eval | Immediate | Cost-effective |

---

*For foundational system design concepts, see [System Design Essentials](./system-design-essentials.md).*

*Last updated: January 2026*
