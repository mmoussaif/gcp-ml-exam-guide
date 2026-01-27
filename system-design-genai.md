# ML & GenAI System Design Guide

A comprehensive guide to designing Machine Learning and Generative AI systems at scale, covering LLM serving, RAG systems, agentic AI, MLOps pipelines, and production considerations.

---

## Prerequisites

This guide focuses specifically on **ML and GenAI system design**. For foundational system design concepts (databases, caching, load balancing, networking, CAP theorem, etc.), see:

📖 **[System Design Essentials](./system-design-essentials.md)** - Core system design knowledge applicable to all distributed systems.

---

## Table of Contents

- [Introduction](#introduction)
- [GenAI vs Traditional ML](#genai-vs-traditional-ml)
- [Using Models & Sampling Parameters](#using-models--sampling-parameters)
- [Google Generative AI Development Tools](#google-generative-ai-development-tools)
- [1. LLM Serving Architecture](#1-llm-serving-architecture-at-scale)
- [2. RAG Systems](#2-rag-retrieval-augmented-generation-system)
- [3. Agentic AI Systems](#3-agentic-ai-systems)
- [4. LLM Ops Data Pipeline](#4-llm-ops-data-pipeline-at-scale)
- [5. GenAI Data Pipeline](#5-genai-data-pipeline-architecture)
- [6. Cost Optimization](#6-cost-optimization-for-genai-systems)
- [7. Real-World Examples](#7-real-world-genai-system-examples)
- [8. Scalability Patterns](#8-scalability-patterns-for-genai)
- [9. Monitoring & Observability](#9-monitoring--observability-for-genai)
- [10. Security & Compliance](#10-security--compliance-for-genai)
- [Resources](#resources)

---

## Introduction

Generative AI applications introduce unique challenges that differ significantly from traditional software systems:

- **Token-by-token generation**: Sequential decoding (unlike batch predictions)
- **Variable latency**: Generation time depends on output length
- **High memory requirements**: KV cache for attention mechanisms
- **Cost optimization**: Balance between latency and throughput
- **Hallucination management**: Ensuring factual accuracy
- **Agent orchestration**: Multi-step reasoning and tool use

This guide covers how to design, build, and operate GenAI systems at scale.

---

## GenAI vs Traditional ML

Understanding the fundamental differences between traditional ML systems and GenAI/LLM systems is crucial for making the right architectural decisions.

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

**2. Top-p (Nucleus Sampling)**

Selects the smallest set of tokens whose cumulative probability mass reaches threshold *p*.

- **High Top-p (0.9-1.0)**: Allows for more diversity by extending to lower probability tokens.
- **Low Top-p (0.1-0.5)**: Leads to more focused responses.
- **Adaptive**: Unlike Top-K, adapts to the distribution's shape—in confident contexts, the "nucleus" is small.

**3. Top-K**

Restricts the model's choice to only the *k* most probable tokens at each step.

- Improves output stability by eliminating the "long tail" of extremely unlikely tokens.
- **Limitation**: Unlike Top-p, it is not adaptive to the distribution's shape.

**4. Maximum Length (Max New Tokens)**

Determines the maximum number of tokens to generate before stopping.

- Prevents runaway generation ("rambling") and controls compute costs.
- Models stop early if they hit an `<EOS>` token.

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
| **Limitations** | Usage limits (QPM, RPM, TPM); small-scale projects | Service charges based on usage; enterprise-grade quotas |
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

**3. KV Cache Management**

**What**: Cache attention key-value pairs to avoid recomputation.

**Why KV cache is needed**: In transformer attention, each token needs to attend to all previous tokens. Without caching, we'd recompute attention for all previous tokens at each step, leading to O(n²) complexity per token.

**How it works**: During generation, we compute K and V for each new token, but reuse cached K/V from previous tokens. This reduces complexity to O(n) per token.

**Challenge**: Memory grows linearly with sequence length. For a 32-layer model with 768-dim embeddings, each token requires ~50KB of cache. A 2000-token sequence needs ~100MB just for KV cache.

**Solution**: Paged attention (vLLM) uses non-contiguous memory pages for better utilization and longer sequences.

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
- Handle 1,000 QPS
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
- **Open Source**: sentence-transformers, BGE models

### Chunking Strategy Trade-offs

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| **Fixed-size (512 tokens)** | Simple, predictable | May split concepts | Uniform documents |
| **Semantic chunking** | Preserves coherence | Complex, variable sizes | Complex content |
| **Hybrid (fixed + overlap)** | Balanced | More storage | Most production systems |

**Why chunking matters**: LLMs have context windows. Documents often exceed this, so we must break them into chunks. Smaller chunks improve retrieval precision—a query about "Python loops" matches better to a 500-token chunk about loops than a 5000-token document about Python.

### Retrieval Strategy Trade-offs

| Strategy | Latency | Semantic | Keywords | Best For |
|----------|---------|----------|----------|----------|
| **Dense (Vector)** | 10-50ms | ✓ | ✗ | Conceptual queries |
| **Sparse (BM25)** | 1-5ms | ✗ | ✓ | Exact matches |
| **Hybrid** | 15-60ms | ✓ | ✓ | Production (recommended) |

**Why hybrid works**: Dense retrieval captures meaning ("iterate" ≈ "loop"), sparse captures exact keywords ("Python"). Combining both via Reciprocal Rank Fusion (RRF) gives best results.

### Reranking Trade-offs

**No Reranking**: Lower latency, simpler pipeline, but lower quality.

**Cross-Encoder Reranking**: Much higher accuracy because it processes query-document pairs together (sees interactions), but adds ~10ms per document.

**Best practice**: Retrieve K=20, rerank to top 5. The two-stage approach combines speed (bi-encoder retrieval) with accuracy (cross-encoder reranking).

---

## 3. Agentic AI Systems

### Use Case: Design a Customer Support Agent

**Requirements:**
- Handle customer inquiries autonomously
- Access multiple tools (CRM, knowledge base, order system)
- Support multi-turn conversations
- Escalate to human when needed
- Handle 10,000 conversations/day

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

### Agent Frameworks

| Platform | Google Cloud | AWS | Open Source |
|----------|--------------|-----|-------------|
| No-code | Vertex AI Agent Builder | Bedrock Agents | - |
| Programmatic | Agent Development Kit (ADK) | AgentCore | LangChain, LlamaIndex, AutoGen |

### Tool Types

| Tool Type | Execution | Description | Best For |
|-----------|-----------|-------------|----------|
| **Extensions (APIs)** | Agent-side | Standardized bridges to external APIs | Multi-service access |
| **Function Calling** | Client-side | Model outputs function name + args; your app executes | Security, audit, human-in-loop |
| **Data Stores** | Agent-side | Connect to vector DBs, knowledge bases | RAG, real-time info |
| **Plugins** | Agent-side | Pre-built integrations (calendar, CRM) | Rapid capability addition |

### Reasoning Frameworks

**Chain-of-Thought (CoT)**: Focuses on internal logic by generating intermediate reasoning steps ("think step-by-step").

**ReAct (Reason + Act)**: Combines reasoning with external tool use in a "Thought-Action-Observation" loop:

| Phase | What Happens |
|-------|--------------|
| **1. Reasoning** | Agent analyzes task, selects tools |
| **2. Acting** | Agent executes selected tool |
| **3. Observation** | Agent receives tool output |
| **4. Iteration** | Based on observation, agent reasons about next steps |

### Agent Design Patterns

**1. Single Agent Pattern**

- One LLM handles entire conversation with all tools
- ✅ Simple, low latency, easy to debug
- ❌ Limited capabilities, may struggle with complex tasks
- *Best for*: Simple use cases, single domain

**2. Multi-Agent Pattern**

- Multiple specialized agents, each with specific tools
- ✅ Better performance (specialists), parallel execution, modular
- ❌ Coordination complexity, higher latency
- *Best for*: Complex domains, multiple expertise areas

**3. Hierarchical Pattern (Supervisor/Manager)**

- Supervisor agent delegates to specialist agents
- ✅ Scalable, organized, handles complex workflows
- ❌ Higher latency, more complex
- *Best for*: Enterprise applications, complex workflows

**4. Additional Patterns**

| Pattern | Architecture | Use Case |
|---------|--------------|----------|
| **Sequential Pipeline** | A → B → C | Content creation workflows |
| **Parallel Fan-out** | Query → [A, B, C] → Aggregate | Research, multi-perspective analysis |
| **Debate/Adversarial** | Pro vs Con → Judge | High-stakes decisions, red teaming |

### Context Engineering

**The Problem**: As agents run longer, context (chat history, tool outputs, documents) **explodes**. Simply using larger context windows is not a scaling strategy.

**The Three-Way Pressure on Context:**

| Pressure | Problem |
|----------|---------|
| **Cost & latency spirals** | Cost and time-to-first-token grow with context size |
| **Signal degradation** | Irrelevant logs distract the model ("lost in the middle") |
| **Physical limits** | RAG results and traces eventually overflow even largest windows |

**The Solution: Tiered Context Model**

| Layer | Purpose | Lifecycle |
|-------|---------|-----------|
| **Working Context** | Immediate prompt for this call | Ephemeral |
| **Session** | Durable log of events | Per-conversation |
| **Memory** | Long-lived searchable knowledge | Cross-session |
| **Artifacts** | Large files | Addressed by name, not pasted |

**Multi-Agent Context Scoping:**

| Pattern | Description |
|---------|-------------|
| **Agents as Tools** | Sub-agent sees only specific instructions |
| **Agent Transfer** | Sub-agent inherits configurable view over Session |

---

## 4. LLM Ops Data Pipeline at Scale

### Use Case: Design a Production LLM Evaluation System

**Requirements:**
- Evaluate model performance continuously
- Track 100+ metrics (accuracy, latency, cost, safety)
- Process 1M predictions/day
- Alert on degradation
- Support A/B testing

**High-Level Design:**

```
┌─────────────────────────────────────────────────────────────────┐
│                  LLM OPS PIPELINE                               │
│                                                                 │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │    LLM       │────►│ Event Stream │────►│   Stream     │   │
│   │ Predictions  │     │ Pub/Sub or   │     │ Processor    │   │
│   │              │     │ Kinesis      │     │              │   │
│   └──────────────┘     └──────────────┘     └──────┬───────┘   │
│                                                     │           │
│                    ┌────────────────────────────────┼───────┐   │
│                    │                                │       │   │
│                    ▼                                ▼       ▼   │
│              ┌───────────┐                   ┌───────────────┐ │
│              │ Evaluation│                   │  Time-Series  │ │
│              │  Metrics  │                   │      DB       │ │
│              │           │                   │               │ │
│              │• Quality  │                   │ • Dashboards  │ │
│              │• Latency  │                   │ • Alerting    │ │
│              │• Cost     │                   │ • A/B Testing │ │
│              │• Safety   │                   │               │ │
│              └───────────┘                   └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Sampling Strategy Trade-offs

| Strategy | Pros | Cons | Cost Estimate |
|----------|------|------|---------------|
| **Full (100%)** | Complete visibility, no bias | Very high cost, privacy concerns | $500-2000/month for 1M/day |
| **Sampled (10%)** | 10x cost reduction | May miss rare errors | $50-200/month |
| **Smart (100% errors + sample successes)** | Captures all failures, cost-effective | More complex | Recommended |

**Why smart sampling works**: Errors are rare but critical—missing one could mean missing a production issue. Successes are common—sampling gives statistical representation without cost.

### Evaluation Frequency Trade-offs

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| **Real-time** | Immediate alerts | High cost, +10-50ms latency | Critical systems, safety |
| **Batch (hourly/daily)** | 10-100x cheaper | Delayed detection | Analytics, reporting |
| **Hybrid** | Balanced | More complex | Most production systems |

**Recommended**: Real-time for latency/errors (user-facing), batch for quality/cost analysis (expensive metrics).

### Key Metrics to Track

- **Quality**: Task-specific accuracy, ROUGE, BLEU, human evaluation
- **Latency**: P50, P95, P99 response times
- **Cost**: Tokens used, cost per request, model tier breakdown
- **Safety**: Toxicity score, jailbreak attempts, bias detection

---

## 5. GenAI Data Pipeline Architecture

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

## 6. Cost Optimization for GenAI Systems

### Token-Based Cost Model

**Cost Components:**
- **Input tokens**: Tokens in prompt (including context)
- **Output tokens**: Generated tokens (typically 2-4x more expensive)
- **Model tier**: Different models have different costs

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

**Routing strategy**: Route complex queries to large model, use small model with fallback to large if confidence is low. Savings: 50-80%.

**4. Fine-tuning ROI**

- **Upfront cost**: $100-1000s
- **Break-even**: If fine-tuning costs $1000 and saves $0.001 per request, break-even at 1M requests
- **Benefits**: Better quality for domain, can use smaller base model

**5. Quantization**

| Precision | Memory Reduction | Quality Loss |
|-----------|-----------------|--------------|
| FP32 → FP16 | 2x | Minimal |
| FP16 → INT8 | 4x | Some |
| INT8 → INT4 | 8x | Significant |

**Why FP16 is safe**: Modern GPUs have Tensor Cores optimized for FP16. Quality loss is minimal (<1%) but memory/cost savings are significant.

**6. Continuous Batching**

- Static batching: 40-60% GPU utilization
- Continuous batching: 80-95% GPU utilization
- **Result**: 2-3x higher throughput

---

## 7. Real-World GenAI System Examples

### Example 1: Code Generation Assistant (like GitHub Copilot)

```
Developer → IDE Extension → API Gateway → Code Generation Service
                                              │
                                    ├──► LLM (Code Model)
                                    ├──► Context Retrieval (RAG)
                                    └──► Code Validation
```

**Key Features:**
- Context-aware (understands codebase)
- Multi-file support
- Real-time generation
- Privacy (code stays private)

**Services**: Vertex AI Codey API, Amazon CodeWhisperer, CodeLlama

### Example 2: Customer Service Chatbot with RAG

```
Customer → Chat Interface → Agent Orchestrator
                                │
                      ├──► RAG System (Knowledge Base)
                      ├──► CRM Integration (Tool)
                      ├──► Order System (Tool)
                      └──► Escalation Logic
```

**Key Features:**
- Knowledge retrieval from company docs
- Tool use (check orders, create tickets)
- Human escalation when needed
- Multi-language support

**Services**: Vertex AI Agent Builder + RAG Engine, Bedrock Agents + Knowledge Bases

### Example 3: Content Generation Platform

```
User Request → Content Pipeline
                    │
          ├──► Research (Web Search)
          ├──► Content Generation (LLM)
          ├──► Fact-Checking (Grounding)
          ├──► SEO Optimization
          └──► Multi-format Output
```

**Key Features:**
- Multi-step generation
- Fact grounding against sources
- Format adaptation (blog, social, email)
- Brand voice consistency

---

## 8. Scalability Patterns for GenAI

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

| Strategy | Speedup | Best For |
|----------|---------|----------|
| Prompt caching (KV cache) | 2-3x | Repeated prefixes |
| Response caching | Instant | Identical requests |
| Semantic caching | Higher hit rate | Similar queries |

---

## 9. Monitoring & Observability for GenAI

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

## 10. Security & Compliance for GenAI

### Key Security Concerns

| Threat | Risk | Mitigation |
|--------|------|------------|
| **Prompt Injection** | Malicious prompts override system instructions | Input validation, Model Armor |
| **Data Leakage** | Training data memorization, PII in outputs | Output filtering, DLP |
| **Access Control** | Unauthorized model access | IAM, API keys, least privilege |

### Model Armor (Google Cloud)

Model Armor is Google Cloud's service for real-time input/output filtering on LLM traffic. It addresses threats that traditional WAFs can't catch—specifically **prompt injection** and **sensitive data disclosure** at the semantic level.

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
| **GDPR** | Right to explanation, data deletion, privacy by design |
| **HIPAA** | Healthcare data protection, audit logging |
| **PCI-DSS** | Payment data security, no storage of card numbers |

### Security Stack Summary

| Layer | Google Cloud | AWS |
|-------|--------------|-----|
| **LLM Security** | Model Armor | Bedrock Guardrails |
| **HTTP Security** | Cloud Armor | WAF |
| **Data Protection** | Cloud DLP | Macie |
| **Secrets** | Secret Manager | Secrets Manager |
| **Network** | VPC Service Controls | VPC |
| **Access** | IAM | IAM |
| **Audit** | Cloud Audit Logs | CloudTrail |

---

## Resources

### Books

- **Building LLM Applications for Production** by Huyen, Chip
- **Designing Machine Learning Systems** by Chip Huyen
- **Designing Data-Intensive Applications** by Martin Kleppmann

### Online

- [LLM Ops Guide](https://llm-ops.com/)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [LangChain Documentation](https://docs.langchain.com/)

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

### Interview Tips for ML/GenAI System Design

1. **Clarify the GenAI-specific requirements**:
   - Expected latency (streaming vs batch?)
   - Quality requirements (accuracy, hallucination tolerance)
   - Cost constraints (per-token, budget caps)
   - Safety requirements (content filtering, compliance)

2. **Consider GenAI-specific components**:
   - Model selection (size vs quality vs cost)
   - Batching strategy (continuous for throughput)
   - Caching (prompt, response, semantic)
   - RAG vs fine-tuning decision

3. **Address unique challenges**:
   - KV cache memory management
   - Token-based cost optimization
   - Prompt injection security
   - Hallucination mitigation

4. **Discuss trade-offs specific to GenAI**:
   - Quality vs cost (model size)
   - Latency vs throughput (batching)
   - Context length vs accuracy (RAG chunking)
   - Complexity vs reliability (single vs multi-agent)

---

*For foundational system design concepts, see [System Design Essentials](./system-design-essentials.md).*

*Last updated: January 2026*
