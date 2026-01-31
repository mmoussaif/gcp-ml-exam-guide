# ML & GenAI System Design Guide

A comprehensive guide to designing **ML (Machine Learning)** and **GenAI (Generative AI)** systems at scale, covering **LLM (Large Language Model)** serving, **RAG** (retrieval-augmented generation) systems, agentic AI, **MLOps** (ML operations) pipelines, and production considerations.

---

## Prerequisites

This guide focuses specifically on **ML and GenAI system design**. For foundational system design concepts (databases, caching, load balancing, networking, CAP theorem, etc.), see:

📖 **[System Design Essentials](./system-design-essentials.md)** - Core system design knowledge applicable to all distributed systems.

---

## Table of Contents

Use this numbered list to track your progress. Check off sections as you complete them.

### Part A: Getting Started

| # | Section | Description | Status |
|---|---------|-------------|--------|
| A.1 | [Introduction](#a1-introduction) | Why GenAI is different; how to use this guide | ☐ |
| A.2 | [Visual Guide Map](#a2-visual-guide-map) | Diagram showing how sections connect | ☐ |
| A.3 | [Glossary](#a3-glossary) | 80+ terms organized by category | ☐ |

### Part B: System Overview

| # | Section | Description | Status |
|---|---------|-------------|--------|
| B.1 | [GenAI System: Big Picture](#b1-genai-system-big-picture-frontend-to-backend) | End-to-end request path and supporting systems | ☐ |
| B.2 | [GenAI vs Traditional ML](#b2-genai-vs-traditional-ml) | Key differences in architecture and operations | ☐ |

### Part C: Generative Models (theory)

| # | Section | Description | Status |
|---|---------|-------------|--------|
| C.1 | [Text-to-Video Generation](#c1-text-to-video-generation) | LDM, temporal layers, DiT, video evaluation | ☐ |
| C.2 | [Multimodal & Vision-Language](#c2-multimodal--vision-language-models) | CLIP, image captioning, visual Q&A | ☐ |

### Part D: LLM Fundamentals

| # | Section | Description | Status |
|---|---------|-------------|--------|
| D.1 | [Using Models & Sampling](#d1-using-models--sampling-parameters) | Temperature, top-p, top-k, when to use each | ☐ |
| D.2 | [Google GenAI Tools](#d2-google-generative-ai-development-tools) | AI Studio, Vertex AI, ADK quick start | ☐ |
| D.3 | [Text Tokenization](#d3-text-tokenization-strategies) | BPE, SentencePiece, WordPiece | ☐ |
| D.4 | [Transformer Architectures](#d4-transformer-architectures) | Attention, encoder-decoder, decoder-only | ☐ |
| D.5 | [ML Objectives for Pretraining](#d5-ml-objectives-for-pretraining) | Next-token prediction, masked LM | ☐ |
| D.6 | [Two-Stage Training](#d6-two-stage-training-pretraining--finetuning) | Pretraining + finetuning pipeline | ☐ |
| D.7 | [Three-Stage Training (Chatbots)](#d7-three-stage-training-for-chatbots-pretraining--sft--rlhf) | Pretraining → SFT → RLHF | ☐ |
| D.8 | [Sampling Strategies](#d8-sampling-strategies-for-text-generation) | Greedy, beam search, nucleus sampling | ☐ |
| D.9 | [Text Generation Evaluation](#d9-text-generation-evaluation-metrics) | Perplexity, BLEU, ROUGE, BERTScore | ☐ |
| D.10 | [Chatbot Inference Pipeline](#d10-chatbot-inference-pipeline-components) | Components from prompt to response | ☐ |

### Part E: Core System Design (the main content)

| # | Section | Description | Status |
|---|---------|-------------|--------|
| E.1 | [LLM Serving Architecture](#e1-llm-serving-architecture-at-scale) | Inference, batching, KV cache, vLLM, parallelism | ☐ |
| E.2 | [RAG Systems](#e2-rag-retrieval-augmented-generation-system) | Chunking, embeddings, vector DB, reranking | ☐ |
| E.3 | [RAG vs Fine-Tuning](#e3-rag-vs-fine-tuning-decision-framework) | When to use each; LoRA, PEFT, decision tree | ☐ |
| E.4 | [Agentic AI Systems](#e4-agentic-ai-systems) | ReAct, tools, multi-agent, ADK, orchestration | ☐ |
| E.5 | [LLM Evaluation & Quality](#e5-llm-evaluation--quality) | RAGAS, LLM-as-judge, human eval, A/B testing | ☐ |
| E.6 | [GenAI Data Pipeline](#e6-genai-data-pipeline-architecture) | Events, labeling, training data, feedback loops | ☐ |
| E.7 | [Cost Optimization](#e7-cost-optimization-for-genai-systems) | Token economics, model routing, caching | ☐ |
| E.8 | [Scalability Patterns](#e8-scalability-patterns-for-genai) | Batching, parallelism, quantization, autoscaling | ☐ |
| E.9 | [Monitoring & Observability](#e9-monitoring--observability-for-genai) | Traces, metrics, drift detection, alerting | ☐ |
| E.10 | [Security & Guardrails](#e10-security--guardrails) | Model Armor, prompt injection, PII, filters | ☐ |

### Part F: Real-World Examples

| # | Section | Description | Status |
|---|---------|-------------|--------|
| F.1 | [Real-World Examples](#f1-real-world-examples-applying-the-stack) | Complete system designs with estimations | ☐ |
| F.1.1 | — LLM Service | High-throughput inference at scale | ☐ |
| F.1.2 | — Support Chatbot | RAG + agents for customer service | ☐ |
| F.1.3 | — Code Assistant | IDE integration, code generation | ☐ |
| F.1.4 | — RAG Pipeline | Document Q&A with evaluation | ☐ |
| F.1.5 | — Gmail Smart Compose | Real-time text prediction | ☐ |
| F.1.6 | — Google Translate | Seq2seq, attention, serving at scale | ☐ |
| F.1.7 | — ChatGPT Clone | Personal assistant architecture | ☐ |
| F.1.8 | — Image Captioning | Vision + language models | ☐ |
| F.1.9 | — RAG Deep Dive | Advanced retrieval patterns | ☐ |
| F.1.10 | — Text-to-Image | Diffusion, CLIP, CFG, evaluation | ☐ |
| F.1.11 | — Text-to-Video | LDM, temporal layers, FVD | ☐ |

### Part G: Reference & Interview Prep

| # | Section | Description | Status |
|---|---------|-------------|--------|
| G.1 | [Strategy & Planning](#g1-strategy-and-planning-for-integration-and-impact) | ROI, metrics, change management | ☐ |
| G.2 | [Quick Reference](#g2-quick-reference) | 45-min interview framework | ☐ |
| G.3 | [Interview Checklist](#what-faang-interviewers-evaluate) | What interviewers look for | ☐ |
| G.4 | [End-to-End Solutioning](#end-to-end-solutioning-and-rrk-skills) | Scope → Design → Deploy → Communicate | ☐ |
| G.5 | [Resources](#g5-resources) | Books, docs, links | ☐ |

---

**Total: 7 parts, 40+ sections** — See [How to Use This Guide](#how-to-use-this-guide) in the Introduction for reading paths.

---

## A.1 Introduction

### Why This Guide Exists

You're building with LLMs. Maybe it's a chatbot, a RAG system, an agent that calls APIs, or a pipeline that generates images. The technology is powerful—but designing reliable, cost-effective GenAI systems is hard.

**Traditional software is deterministic.** Same input → same output. You can test it, cache it, reason about it.

**GenAI is different.** Same prompt → different response every time. Responses can be wrong (hallucinations), expensive (every token costs money), slow (seconds, not milliseconds), and unpredictable (agents can loop forever).

This guide teaches you how to design for these realities.

---

### The Six Challenges of GenAI Systems

| Challenge | The Problem | What You'll Learn |
| --------- | ----------- | ----------------- |
| **Non-determinism** | Same prompt yields different outputs; hard to test and debug | Evaluation strategies (E.5), guardrails (E.10) |
| **Token economics** | Cost and latency scale with input + output length | Cost optimization (E.7), caching, model routing |
| **Memory pressure** | KV cache grows with context; long prompts exhaust GPU memory | Serving architecture (E.1), quantization (E.8) |
| **Hallucinations** | Model confidently states false information | RAG for grounding (E.2), evaluation (E.5) |
| **Orchestration complexity** | Agents need tools, retrieval, and multi-step reasoning | Agentic systems (E.4), ADK patterns |
| **Scale unpredictability** | Variable output length makes capacity planning hard | Scalability patterns (E.8), continuous batching |

---

### What You'll Get From This Guide

| Layer | What's Covered | Sections |
| ----- | -------------- | -------- |
| **Theory** | How LLMs, RAG, agents, diffusion models, and training pipelines actually work | Parts C, D |
| **System Design** | Architecture patterns for serving, retrieval, agents, evaluation, and operations | E.1–E.10 |
| **Practice** | Real stacks, cost estimations, and complete system designs | F.1 Examples |
| **Interview Prep** | 45-minute framework, FAANG evaluation criteria, end-to-end solutioning | Part G |

---

### How to Use This Guide

**If you're new to GenAI systems:**
1. Start with the **Visual Guide Map** (next section) to see the big picture
2. Read **Part B: System Overview** to understand the request path
3. Work through **Part D: LLM Fundamentals** for theory
4. Then dive into **Part E** (E.1–E.10) for system design patterns

**If you're preparing for interviews:**
1. Skim the **Glossary** to ensure you know the terminology
2. Read **E.1–E.4** (Serving, RAG, Fine-tuning, Agents) deeply
3. Study 3–4 examples from **F.1** and practice explaining them
4. Use **Part G: Quick Reference** for the interview framework

**If you're building a system now:**
1. Find the closest example in **F.1**
2. Read the relevant deep-dive sections (E.1–E.10)
3. Use the **Glossary** and **Resources** as needed

---

### Legend

Throughout this guide, you'll see these markers:

| Symbol | Meaning |
| ------ | ------- |
| 💡 | Key insight—something that clicks once you understand it |
| 📊 | Estimation—rough numbers for capacity planning and cost |
| 🛠️ | Stack snapshot—concrete tools and technologies |
| ✅ | Best practice—what works in production |
| 🔷 | End-to-end phase—part of a complete workflow |

---

### The Mental Model

GenAI system design comes down to three things:

```
1. NON-DETERMINISM     →  How do you evaluate and control probabilistic outputs?
2. TOKEN ECONOMICS     →  How do you manage cost and latency that scale with length?
3. ORCHESTRATION       →  How do you combine models, retrieval, and tools?
```

Every section in this guide addresses one or more of these. By the end, you'll be able to:

- **Design** a complete GenAI system (serving → RAG → agents → evaluation)
- **Estimate** costs, latency, and capacity requirements
- **Articulate** trade-offs clearly in interviews or architecture reviews
- **Build** with real tools (Vertex AI, Bedrock, vLLM, LangChain, ADK)

Let's start with the big picture.

---

## A.2 Visual Guide Map

This map shows how the guide fits together. Follow **Parts A → G** in order, or jump to what you need.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           GenAI SYSTEM DESIGN GUIDE                                 │
└─────────────────────────────────────────────────────────────────────────────────────┘

  START HERE
      │
      ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PART A    │     │   PART B    │     │   PART C    │
│  GETTING    │────▶│   SYSTEM    │────▶│ GENERATIVE  │
│  STARTED    │     │  OVERVIEW   │     │   MODELS    │
├─────────────┤     ├─────────────┤     ├─────────────┤
│ A.1 Intro   │     │ B.1 Big     │     │ C.1 Text-to │
│ A.2 Map     │     │     Picture │     │     -Video  │
│ A.3 Glossary│     │ B.2 GenAI   │     │ C.2 Multi-  │
│             │     │     vs ML   │     │     modal   │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
      ┌───────────────────────────────────────┘
      ▼
┌─────────────┐
│   PART D    │
│    LLM      │
│FUNDAMENTALS │
├─────────────┤
│ D.1  Models & Sampling      │
│ D.2  Google GenAI Tools     │
│ D.3  Tokenization           │
│ D.4  Transformers           │
│ D.5  Pretraining Objectives │
│ D.6  Two-Stage Training     │
│ D.7  Three-Stage (RLHF)     │
│ D.8  Sampling Strategies    │
│ D.9  Evaluation Metrics     │
│ D.10 Inference Pipeline     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            PART E: CORE SYSTEM DESIGN                               │
│                              (the main content)                                     │
├────────────────┬────────────────┬────────────────┬──────────────────────────────────┤
│   SERVING      │   KNOWLEDGE    │   QUALITY      │          OPERATIONS              │
├────────────────┼────────────────┼────────────────┼──────────────────────────────────┤
│ E.1 LLM        │ E.2 RAG        │ E.5 Evaluation │ E.6 Data Pipeline                │
│     Serving    │     System     │     & Quality  │ E.7 Cost Optimization            │
│   - Batching   │   - Chunking   │   - RAGAS      │ E.8 Scalability                  │
│   - KV Cache   │   - Embeddings │   - Human Eval │ E.9 Monitoring                   │
│   - vLLM       │   - Vector DB  │   - A/B Test   │                                  │
│                │   - Reranking  │                │                                  │
│                │                │ E.10 Security  │                                  │
│                │ E.3 RAG vs     │    & Guardrails│                                  │
│                │     Fine-tune  │   - Model Armor│                                  │
│                │   - LoRA/PEFT  │   - PII Filter │                                  │
│                │                │                │                                  │
│                │ E.4 Agentic AI │                │                                  │
│                │   - ReAct      │                │                                  │
│                │   - Tools      │                │                                  │
│                │   - Multi-agent│                │                                  │
│                │   - ADK        │                │                                  │
└────────────────┴────────────────┴────────────────┴──────────────────────────────────┘
              │
              ▼
┌─────────────┐     ┌─────────────┐
│   PART F    │     │   PART G    │
│  EXAMPLES   │────▶│  REFERENCE  │
│  (Apply it) │     │ & INTERVIEW │
├─────────────┤     ├─────────────┤
│ F.1 Real-   │     │ G.1 Strategy│
│   World     │     │ G.2 Quick   │
│   Examples  │     │     Ref     │
│  - LLM Svc  │     │ G.3 What    │
│  - Chatbot  │     │   FAANG     │
│  - Code Ast │     │   Evaluates │
│  - RAG      │     │ G.4 End-to- │
│  - Translate│     │   End       │
│  - ChatGPT  │     │ G.5 Resources
│  - T2I/T2V  │     │             │
└─────────────┘     └─────────────┘
```


---

## A.3 Glossary

Quick reference for key terms. Organized by category for easier navigation. **Start here if you're new** — the Fundamentals section explains basic computing terms.

### Fundamentals (Start Here)

| Term | Definition | Why it matters |
| ---- | ---------- | -------------- |
| **AI** | Artificial Intelligence. Computers that can do tasks normally requiring human intelligence — recognizing images, understanding language, making decisions. | The broad field. ML and GenAI are types of AI. |
| **ML** | Machine Learning. Teaching computers to learn patterns from data instead of explicit programming. Show it 1000 cat photos → it learns to recognize cats. | The foundation for all modern AI. Models learn from examples. |
| **Neural Network** | A computer system inspired by the human brain. Layers of connected "neurons" that process information and learn patterns. Deep = many layers. | The architecture behind LLMs, image generators, and most modern AI. |
| **GPU** | Graphics Processing Unit. A chip originally designed for video games that's very good at doing many calculations in parallel. Essential for AI training and inference. | AI needs GPUs. A single GPU can do 1000× more parallel math than a CPU. Training LLMs requires 100s-1000s of GPUs. |
| **CPU** | Central Processing Unit. The main "brain" of a computer. Good at sequential tasks but slow for AI workloads compared to GPUs. | CPUs run your computer; GPUs run AI. |
| **API** | Application Programming Interface. A way for programs to talk to each other. You send a request, you get a response. Like a waiter taking your order to the kitchen. | How you use AI services. Send prompt to OpenAI API → get response back. |
| **JSON** | JavaScript Object Notation. A simple text format for structured data: `{"name": "Alice", "age": 30}`. Both humans and computers can read it. | The standard format for API requests/responses and LLM tool calls. |
| **NLP** | Natural Language Processing. Teaching computers to understand and generate human language — the field that led to ChatGPT. | LLMs are NLP models. Understanding NLP history helps understand LLM design. |
| **Open Source** | Software whose code is freely available for anyone to use, modify, and share. Examples: Linux, LLaMA, Stable Diffusion. | Many AI tools are open source. You can run them yourself instead of paying API fees. |
| **Hyperparameter** | A setting you choose before training (learning rate, batch size, etc.). The model doesn't learn these — you set them. | Tuning hyperparameters is how you optimize training. Different from model parameters (weights). |
| **Epoch** | One complete pass through all training data. Training for 3 epochs = seeing every example 3 times. | More epochs = model sees data more times. Too many = overfitting (memorizing instead of learning). |
| **Weights / Parameters** | The numbers inside a neural network that determine its behavior. A 7B model has 7 billion parameters. Training = adjusting these numbers. | Model size is measured in parameters. More parameters = more capable but more expensive to run. |
| **Training** | Teaching a model by showing it examples and adjusting its weights to reduce errors. Requires lots of data and compute. | Training GPT-4 cost ~$100M. Most users don't train — they use pre-trained models or fine-tune. |
| **Inference** | Running a trained model to get predictions. For LLMs: sending a prompt and getting a response. | What you pay for when using APIs. Most of your AI costs are inference, not training. |
| **Loss Function** | Measures how wrong the model's predictions are. Training tries to minimize loss. Lower loss = better predictions. | The "score" during training. Model adjusts weights to reduce loss. Different tasks use different loss functions. |
| **Attention** | A mechanism that lets models focus on relevant parts of the input. "What words should I pay attention to when predicting the next word?" | The key innovation in Transformers. Why LLMs can understand context and relationships between words. |
| **Transformer** | The neural network architecture behind LLMs. Uses attention to process all words in parallel instead of one-by-one. | Invented in 2017 ("Attention Is All You Need" paper). Powers GPT, BERT, Gemini, Claude, and all modern LLMs. |
| **Encoder** | Processes input and creates a representation (embedding). Reads and understands. BERT is encoder-only. | Good for classification, embeddings, understanding. Not for generation. |
| **Decoder** | Generates output token by token. GPT and most chat models are decoder-only. | Good for text generation, chat, code. The architecture behind ChatGPT. |
| **Cross-Attention** | Attention between two different sequences (e.g., text prompt and image). Lets one sequence "look at" the other. | How text guides image generation in diffusion models. Text embeddings cross-attend with image features. |

### Core Concepts

| Term | Definition | Why it matters |
| ---- | ---------- | -------------- |
| **LLM** | Large Language Model. A neural network with billions of parameters trained on massive text corpora to predict the next token. Examples: GPT-4, Gemini, Claude, LLaMA. | The foundation of modern GenAI. Understanding how LLMs work (attention, tokens, context) is essential for system design. |
| **T5** | Text-to-Text Transfer Transformer. Google's encoder-decoder model that frames all NLP tasks as text-to-text. Input: "translate English to German: Hello" → Output: "Hallo". | Versatile architecture for translation, summarization, Q&A. Used as text encoder in diffusion models (Imagen, Stable Diffusion 3). |
| **GenAI** | Generative AI. Models that create new content—text, images, video, audio, code—rather than just classifying or predicting. | Broader than LLMs: includes diffusion models (images), video generators, music models. |
| **Token** | The smallest unit of text the model processes. Roughly 4 characters or 0.75 words in English. "Hello world" ≈ 2 tokens. Models charge and limit by tokens. | Tokens determine cost, latency, and context limits. A 100K token context costs 100× more than 1K. |
| **Context Window** | Maximum number of tokens an LLM can see in one request (prompt + response combined). GPT-4: 128K, Gemini 1.5: 2M, Claude 3: 200K. | Larger context = more information per request, but higher cost and latency. Design retrieval to fit within limits. |
| **Inference** | Running a trained model to get predictions. For LLMs: turning a prompt into a response. | Most of your GenAI costs come from inference, not training. Optimize inference = save money. |
| **Latency** | Time from request to response. For LLMs: TTFT (first token) + generation time. Typically 100ms–10s depending on model and output length. | Users notice latency >2s. Streaming helps perception. Trade off latency vs cost vs quality. |

### Tokens & Generation

| Term | Definition | Why it matters |
| ---- | ---------- | -------------- |
| **Tokenization** | Converting text to tokens. Different models use different tokenizers (BPE, SentencePiece, WordPiece). "unhappily" → ["un", "happy", "ly"]. | Same text = different token counts on different models. Affects cost calculations and context limits. |
| **Autoregressive** | Generating one token at a time, using previous tokens to predict the next. LLMs generate left-to-right, token by token. | Explains why LLM latency scales with output length. 1000 tokens takes ~10× longer than 100 tokens. |
| **Temperature** | Controls randomness in token selection. 0 = always pick most likely token (deterministic). 1 = sample according to probabilities. >1 = more random/creative. | Low temp for factual tasks (0–0.3). High temp for creative tasks (0.7–1.0). Critical parameter for quality. |
| **Top-p (Nucleus)** | Only consider tokens whose cumulative probability ≤ p. Top-p=0.9 means pick from tokens covering 90% of probability mass. | Alternative to temperature. Often used together. Prevents very unlikely tokens from being selected. |
| **Top-k** | Only consider the k most likely next tokens. Top-k=50 means choose from top 50 candidates only. | Simpler than top-p. Can combine with temperature. Prevents rare/weird token selection. |
| **Sampling** | The process of selecting the next token from the probability distribution. Greedy (always max) vs random (sample from distribution). | Greedy = deterministic but repetitive. Random sampling with temp/top-p/top-k = more varied outputs. |

### Memory & Caching

| Term | Definition | Why it matters |
| ---- | ---------- | -------------- |
| **KV Cache** | Key-Value cache. Stores the computed attention keys and values for all previous tokens so they don't need to be recomputed for each new token. | Without KV cache, generating token N would require O(N²) computation. KV cache makes it O(N). But it uses memory that grows with sequence length. |
| **PagedAttention** | Memory management technique (from vLLM) that stores KV cache in non-contiguous memory pages, like virtual memory in operating systems. | Enables much higher throughput by reducing memory fragmentation. Can serve 2-4× more concurrent requests. |
| **Semantic Cache** | Cache LLM responses by embedding similarity rather than exact string match. Similar questions get cached answers. | Can reduce costs 30-50% for repetitive queries. But risk of returning stale or slightly wrong cached answers. |
| **Prompt Cache** | Cache the KV computations for common prompt prefixes (system prompts, few-shot examples). Reuse across requests. | System prompts are often identical across requests. Caching saves compute and reduces TTFT. |

### RAG (Retrieval-Augmented Generation)

| Term | Definition | Why it matters |
| ---- | ---------- | -------------- |
| **RAG** | Retrieval-Augmented Generation. Pattern: (1) embed user query, (2) retrieve relevant documents from vector DB, (3) inject documents into prompt, (4) generate response grounded in retrieved context. | The standard way to give LLMs access to private/current data without fine-tuning. Cheaper, more flexible, data stays fresh. |
| **Embedding** | A dense vector (e.g., 768 or 1536 dimensions) representing the semantic meaning of text. Similar meanings → vectors that are close together in vector space. | Embeddings enable semantic search: "car" and "automobile" are close even though strings are different. |
| **Embedding Model** | Model that converts text → embedding vector. Examples: OpenAI text-embedding-3, Cohere embed, Vertex AI textembedding-gecko, open-source e5/bge. | Different models have different dimensions, quality, and cost. Choose based on your retrieval quality needs. |
| **Vector Database** | Database optimized for storing embeddings and finding nearest neighbors. Examples: Pinecone, Weaviate, Milvus, Qdrant, pgvector (Postgres extension), Vertex AI Vector Search. | Regular databases can't efficiently search by vector similarity. Vector DBs use specialized indexes (HNSW, IVF). |
| **Chunking** | Splitting documents into smaller pieces (chunks) for embedding and retrieval. Typically 200-1000 tokens per chunk. | Too small = lose context. Too large = irrelevant content dilutes signal. Chunk size affects retrieval quality. |
| **Overlap** | When chunking, include some text from the previous chunk (e.g., 50-100 tokens). Helps preserve context across chunk boundaries. | Without overlap, sentences split across chunks lose meaning. Overlap trades storage for better retrieval. |
| **Reranking** | After initial retrieval (e.g., top 20 chunks by embedding similarity), use a more expensive cross-encoder model to re-score and reorder by true relevance. | Embedding similarity is fast but approximate. Reranking is slower but more accurate. Typical flow: retrieve 20 → rerank → use top 5. |
| **Bi-Encoder** | Embeds query and documents *separately*, then compares with dot product. Like judging if two puzzle pieces fit by looking at photos of each piece alone. | Fast (embed once, compare many). But misses how query and document interact. Used for initial retrieval. |
| **Cross-Encoder** | Processes query and document *together* in one pass, seeing how they relate. Like actually trying to fit two puzzle pieces together to see if they match. | Slow (one forward pass per pair) but much more accurate. Used for reranking top results from bi-encoder. |
| **Hybrid Search** | Combine vector similarity search with keyword search (BM25). Merges results using reciprocal rank fusion or similar. | Vector search misses exact matches; keyword search misses synonyms. Hybrid gets both. Often 10-20% better retrieval than either alone. |
| **BM25** | Best Match 25. A keyword search algorithm that ranks documents by term frequency. Finds exact word matches. The "traditional" search before vector search. | Fast and good for exact matches ("error code 404"). Use with vector search for best results. |
| **FAISS** | Facebook AI Similarity Search. Open-source library for fast vector similarity search. Implements HNSW, IVF, and other ANN algorithms. | The most popular vector search library. Used standalone or inside vector databases. |
| **ANN** | Approximate Nearest Neighbor. Algorithms that find similar vectors quickly by trading exactness for speed. Exact search is O(n); ANN is O(log n). | Essential for RAG at scale. 1M vectors with exact search = seconds. ANN = milliseconds. Recall vs speed trade-off. |
| **HNSW** | Hierarchical Navigable Small World. Graph-based ANN algorithm. Builds a multi-layer graph where upper layers connect distant nodes. | Best recall-latency trade-off for high-dimensional vectors. Default choice for most vector DBs (Pinecone, Weaviate, FAISS). |
| **IVF** | Inverted File Index. Clustering-based ANN. Clusters vectors into groups (e.g., 100 clusters), then only searches relevant clusters at query time. | Uses less memory than HNSW. Good when index size is a constraint. Lower recall than HNSW; requires tuning nprobe. |
| **nprobe** | Number of clusters to search in IVF. If you have 100 clusters and nprobe=10, you search the 10 closest clusters. Like checking 10 filing cabinets instead of all 100. | Higher nprobe = better recall (find more matches) but slower. Lower nprobe = faster but might miss results. Typical: 10-50. |
| **Grounding** | Anchoring LLM responses to specific retrieved sources. Model should cite where information came from. | Without grounding, LLMs confidently hallucinate. Grounding makes responses verifiable and trustworthy. |
| **Context Stuffing** | Putting as much retrieved context as possible into the prompt, up to context window limit. | More context = more information, but also more noise and higher cost. Quality of retrieval matters more than quantity. |
| **OCR** | Optical Character Recognition. Extracts text from images or scanned documents. Modern OCR uses neural networks for accuracy. | Required for RAG on PDFs, scans, or images with text. Quality varies—test on your document types. |

### Fine-Tuning

| Term | Definition | Why it matters |
| ---- | ---------- | -------------- |
| **Fine-tuning** | Training a pretrained model on your task-specific data. Model learns your domain, style, or capabilities. | Changes model behavior permanently. More expensive than RAG but can improve quality for specific tasks. |
| **SFT** | Supervised Fine-Tuning. Train on (input, output) pairs. Model learns to produce the expected output for each input. | The standard fine-tuning approach. Need 100s–1000s of high-quality examples. |
| **LoRA** | Low-Rank Adaptation. Instead of updating all model weights, train small "adapter" matrices (rank 8-64) that modify the frozen base model. | 10-100× cheaper than full fine-tuning. Adapters are small (MBs vs GBs). Can swap adapters at inference. |
| **QLoRA** | Quantized LoRA. Combine LoRA with 4-bit quantization of base model. Train adapters on quantized model. | Even cheaper than LoRA. Can fine-tune 70B models on a single GPU. Some quality loss from quantization. |
| **PEFT** | Parameter-Efficient Fine-Tuning. Umbrella term for LoRA, QLoRA, adapters, prefix tuning—any method that trains only a small subset of parameters. | Full fine-tuning is expensive and requires storing full model copies. PEFT makes fine-tuning practical. |
| **RLHF** | Reinforcement Learning from Human Feedback. Train a reward model on human preferences, then use RL to optimize the LLM to get higher rewards. | How ChatGPT was trained to be helpful/harmless. Complex pipeline: need preference data, reward model, RL training. |
| **DPO** | Direct Preference Optimization. Simpler alternative to RLHF that directly optimizes on preference pairs without a separate reward model. | Easier to implement than RLHF. Becoming the preferred approach for alignment fine-tuning. |
| **Instruction Tuning** | Fine-tuning on (instruction, response) pairs to make model better at following instructions. | Why base models become chat models. Instruct-tuned models follow prompts better than base models. |

### Agents & Tools

| Term | Definition | Why it matters |
| ---- | ---------- | -------------- |
| **Agent** | An LLM that can use tools and reason in a loop. Perceive state → decide action → execute tool → observe result → repeat until done. | Enables LLMs to take actions in the world: query databases, call APIs, run code, browse web. |
| **Tool / Function Calling** | LLM outputs structured JSON specifying which function to call with what arguments. System executes the function and returns result to LLM. | The mechanism for agents to interact with external systems. Most modern LLMs support native function calling. |
| **ReAct** | Reasoning + Acting. Agent pattern: Thought (reasoning about what to do) → Action (tool call) → Observation (tool result) → repeat. | Popular agent framework. Interleaving reasoning with actions makes agent behavior more interpretable. |
| **Multi-Agent** | System with multiple specialized agents that collaborate. E.g., researcher agent + writer agent + reviewer agent. | Complex tasks benefit from specialization. Agents can have different tools, prompts, or even different LLMs. |
| **ADK** | Agent Development Kit. Google's open-source framework for building agents. Supports workflow agents (Sequential, Parallel, Loop), tools, multi-agent orchestration. | The recommended way to build agents on GCP. Integrates with Vertex AI Agent Engine for deployment. |
| **MCP** | Model Context Protocol. Open standard for exposing tools and context to LLMs. Defines how to describe tools, call them, and return results. | Standardizes tool integration. Tools written for MCP work with any MCP-compatible agent framework. |
| **A2A** | Agent-to-Agent Protocol. Standard for how agents communicate and delegate tasks to each other. | Enables interoperable multi-agent systems. Agent A can delegate to Agent B even if built with different frameworks. |
| **Orchestration** | The layer that manages LLM calls, tool execution, retrieval, and control flow. Examples: LangChain, LlamaIndex, ADK. | Glue code between LLM, tools, and your application. Handles retries, routing, state management. |

### Prompting

| Term | Definition | Why it matters |
| ---- | ---------- | -------------- |
| **Prompt Engineering** | Designing prompts to get better outputs. Includes system prompts, few-shot examples, chain-of-thought, output format specifications. | Good prompts can improve quality 2-3× without changing the model. Often the highest-ROI optimization. |
| **System Prompt** | Instructions at the start of the prompt that set persona, constraints, and behavior. Persists across the conversation. | "You are a helpful assistant that..." Sets the tone and rules. Most production apps have carefully crafted system prompts. |
| **Few-shot** | Including examples in the prompt: "Input: X → Output: Y. Input: A → Output: B. Input: [user query] → Output:" | Shows the model the desired format and style. Often 3-5 examples. More examples = better but uses more tokens. |
| **Zero-shot** | Prompting without examples. Just the instruction and the query. | Simpler and cheaper. Works well for capable models on common tasks. Try zero-shot first, add few-shot if needed. |
| **Chain-of-Thought (CoT)** | Prompting the model to reason step-by-step before giving the final answer. "Let's think through this step by step..." | Dramatically improves reasoning and math. Makes the model "show its work." Can add 2-3× to output length. |
| **Output Formatting** | Specifying the desired output structure. "Respond in JSON with fields: answer, confidence, sources." | Makes outputs parseable and consistent. Essential for production systems that need structured data. |

### Serving & Performance

| Term | Definition | Why it matters |
| ---- | ---------- | -------------- |
| **TTFT** | Time To First Token. Latency from sending request to receiving the first token of the response. | Users perceive responsiveness from TTFT. Optimize TTFT for interactive applications. Streaming helps. |
| **TPS** | Tokens Per Second. How fast the model generates tokens after the first one. Typical: 30-100 TPS depending on model and hardware. | Affects total response time. 100 tokens at 50 TPS = 2 seconds of generation time. |
| **Throughput** | Total tokens per second across all concurrent requests. A serving system's capacity. | More throughput = serve more users. Trade off throughput vs latency (batching helps throughput, hurts latency). |
| **Batching** | Processing multiple requests together. Static batching waits for batch to fill; continuous batching adds/removes requests dynamically. | Batching improves GPU utilization. Continuous batching (vLLM, TGI) is state-of-the-art for LLM serving. |
| **Continuous Batching** | Dynamically add new requests to a running batch as slots free up (when requests complete). No waiting for batch boundaries. | Much higher throughput than static batching. Standard in modern LLM serving (vLLM, TGI, TensorRT-LLM). |
| **Quantization** | Reducing model precision from FP32/FP16 to INT8/INT4. Model is smaller and faster, with some quality loss. | Can reduce memory 2-4× and improve speed 1.5-2×. Quality loss is often acceptable. Essential for deploying large models. |
| **FP32/FP16** | Floating Point 32-bit / 16-bit. How precisely numbers are stored. FP32 = very precise but uses more memory. FP16 = half the memory, slightly less precise. | FP16 is standard for inference. Same quality, half the memory. FP32 sometimes needed for training stability. |
| **INT8/INT4** | Integer 8-bit / 4-bit. Even lower precision than FP16. Numbers rounded to integers. | Aggressive compression. INT8 = 2× smaller than FP16. INT4 = 4× smaller. Some quality loss but often acceptable. |
| **vLLM** | Open-source LLM serving engine with PagedAttention, continuous batching, and high throughput. The most popular OSS option. | 2-4× better throughput than naive serving. Production-ready. Supports most open models. |
| **TGI** | Text Generation Inference. Hugging Face's LLM serving solution. Similar capabilities to vLLM. | Good Hugging Face integration. Used by Inference Endpoints. Alternative to vLLM. |
| **TensorRT** | Tensor Runtime. NVIDIA's library that optimizes neural networks for faster inference on NVIDIA GPUs. Fuses layers, reduces precision, optimizes memory. | Can speed up inference 2-5×. TensorRT-LLM is the LLM-specific version with batching and KV cache optimizations. |
| **Speculative Decoding** | Use a small "draft" model to predict multiple tokens, then verify with the large model in parallel. Faster if draft model is accurate. | Can speed up generation 2-3× for some model pairs. Works best when draft model is good at predicting the large model. |

### Parallelism & Scaling

| Term | Definition | Why it matters |
| ---- | ---------- | -------------- |
| **Tensor Parallelism** | Split each layer's weights across multiple GPUs. Each GPU computes part of each layer, then they communicate. | Required when model doesn't fit on one GPU. Llama 70B needs 4+ GPUs with tensor parallelism. |
| **Pipeline Parallelism** | Split model into stages (groups of layers), each stage on different GPU. Requests flow through pipeline. | Alternative to tensor parallelism. Less communication but more complex scheduling. Often combined with tensor parallelism. |
| **Data Parallelism** | Same model on multiple GPUs, each processes different data. For training: gradients are averaged. | Standard for training. For serving, more about replication than parallelism—multiple model copies. |
| **Model Parallelism** | Umbrella term for tensor and pipeline parallelism—any technique that splits the model across GPUs. | Essential for large models. A 70B model with FP16 needs ~140GB, far exceeding single GPU memory. |
| **FSDP** | Fully Sharded Data Parallel. Distributed training technique that shards model parameters, gradients, and optimizer states across GPUs. Each GPU holds only a fraction; gathers on-demand. | Enables training models too large for one GPU. PyTorch native. Combine with gradient checkpointing and mixed precision for 70B+ models. |
| **RoPE** | Rotary Position Embedding. A way to encode token positions in LLMs that supports any sequence length. Rotates embeddings based on position. | Enables long context windows (100K+ tokens). Used in LLaMA, Gemini, and most modern LLMs. Better than absolute position embeddings. |
| **Softmax** | A function that converts a list of numbers into probabilities that sum to 1. Used in attention and for picking the next token. | The final step in LLM generation. Turns "raw scores" into "probability that this is the next token." |
| **Cosine Similarity** | Measures how similar two vectors are (0 = unrelated, 1 = identical direction). Used to compare embeddings. | How vector search works. Query embedding vs document embeddings → rank by cosine similarity. |

### Image & Video Generation

| Term | Definition | Why it matters |
| ---- | ---------- | -------------- |
| **Diffusion Model** | Generative model trained to reverse a noising process. Learns to denoise: given noisy image, predict the noise to remove. Generation: start from pure noise, iteratively denoise. | The dominant approach for image generation (Stable Diffusion, DALL-E 3, Imagen). Also used for video and audio. |
| **Latent Space** | A compressed "summary" of data. Think of it like a ZIP file for images — smaller but contains the essential information to recreate the original. A 512×512 image (262,144 pixels) becomes a 64×64 latent (4,096 numbers). Similar images have similar latents, like how similar books have similar summaries. | Working in latent space is 64× cheaper than pixel space. It's like editing a thumbnail instead of a 4K photo — much faster, and you can upscale later. |
| **Latent Diffusion (LDM)** | Generate images in compressed latent space, then decompress to full resolution. Like sketching on a small notepad first (fast, easy to erase/edit), then enlarging to a full canvas at the end. | Stable Diffusion uses this: compress 512×512 → work in 64×64 → decompress back. 64× less computation. That's why it runs on consumer GPUs. |
| **DiT** | Diffusion Transformer. Uses Transformer architecture instead of U-Net for the denoising network. Patches image like ViT, applies attention. | Scales better than U-Net. Used in Sora, newer video models. More compute but better quality at scale. |
| **U-Net** | A neural network shaped like a "U". Downsamples image → processes → upsamples back. Used in Stable Diffusion to predict noise. | The original architecture for diffusion denoising. Being replaced by DiT in newer models. |
| **ViT** | Vision Transformer. Applies Transformer to images by splitting into patches (e.g., 16×16), treating each patch as a token. | Foundation for modern vision models. CLIP, DINO, and many image encoders use ViT architecture. |
| **VAE** | Variational Autoencoder. Encoder compresses image to latent, decoder reconstructs image from latent. Used in latent diffusion. | The compression step that makes latent diffusion efficient. Trained separately from the diffusion model. |
| **CLIP** | Contrastive Language-Image Pretraining. Model trained to align images and text in a shared embedding space. | Enables text-to-image: encode text with CLIP, use embedding to guide diffusion. Also used for evaluation (CLIPScore). |
| **CFG** | Classifier-Free Guidance. Technique to improve prompt adherence in diffusion. Generate with and without prompt, amplify the difference. CFG scale controls strength. | Higher CFG = more prompt-adherent but less diverse. Typical values: 7-15. Critical parameter for image quality. |
| **DDPM** | Denoising Diffusion Probabilistic Models. Original diffusion sampling method. 1000 steps, each step predicts and removes a small amount of noise. | High quality but very slow (~minutes per image). The theoretical foundation for diffusion models. |
| **DDIM** | Denoising Diffusion Implicit Models. Faster sampling that skips steps (1000 → 20-50) while maintaining quality. Deterministic given same seed. | Standard for production. 20-50 steps = 1-3 seconds per image. Trade-off: fewer steps = faster but lower quality. |
| **Negative Prompt** | Text describing what you don't want in the image. "blurry, low quality, watermark". Diffusion model steers away from it. | Often as important as the positive prompt. Standard practice in image generation. |
| **FID** | Fréchet Inception Distance. Metric comparing distribution of generated images to real images using Inception network features. Lower = better. | Standard metric for image generation quality. Measures both quality and diversity. |
| **FVD** | Fréchet Video Distance. Like FID but for video, using I3D features. Measures both frame quality and temporal consistency. | The main automated metric for video generation. Captures motion quality, not just frame quality. |
| **CLIPScore** | Cosine similarity between CLIP embeddings of image and text prompt. Higher = better text-image alignment. | Measures if the image matches the prompt. FID measures quality; CLIPScore measures relevance. Need both. |
| **Temporal Consistency** | Whether video frames transition smoothly and objects maintain identity across frames. | The hard part of video generation. Individual frames can look good but motion can be jittery or objects can morph. |

### Evaluation & Quality

| Term | Definition | Why it matters |
| ---- | ---------- | -------------- |
| **Hallucination** | Model generates plausible-sounding but factually incorrect information. Confidently states false things. | The core reliability problem with LLMs. RAG, grounding, and guardrails help but don't eliminate. |
| **Faithfulness** | Whether the response accurately reflects the retrieved/provided context. Did the model use the sources correctly? | Key metric for RAG. Model might have sources but still make things up or misrepresent them. |
| **Relevancy** | Whether the response actually answers the question. Model might be faithful to context but not address the query. | Different from faithfulness. Response can be grounded but off-topic. Measure both. |
| **RAGAS** | Reference-free RAG evaluation framework. Computes faithfulness, answer relevancy, context relevancy without ground truth. Uses LLM-as-judge. | Enables automated RAG evaluation at scale without labeled data. Industry standard for RAG metrics. |
| **LLM-as-Judge** | Using an LLM to evaluate another LLM's outputs. Prompt: "Rate this response for accuracy 1-5 and explain why." | Scalable evaluation. Not perfect (LLMs have biases) but correlates with human judgment. Use strong models as judges. |
| **Human Evaluation** | Human raters assess quality, usually on Likert scales or A/B preferences. Gold standard but expensive and slow. | Required for high-stakes applications. Use for calibration and final validation. Automate what you can, human-eval the rest. |
| **A/B Testing** | Show different model versions to different users, measure which performs better on business metrics. | The ultimate evaluation: does it work in production? Requires sufficient traffic and clear metrics. |
| **Guardrails** | Safety filters that check inputs and outputs for policy violations: toxicity, PII, jailbreaks, harmful content. | Required for production. Check inputs (block malicious prompts) and outputs (block harmful responses). |
| **Model Armor** | Google Cloud's guardrail service. Detects prompt injection, jailbreaks, and harmful content. | Managed guardrails—don't build from scratch. Integrates with Vertex AI. |

### Infrastructure & Deployment

| Term | Definition | Why it matters |
| ---- | ---------- | -------------- |
| **Vertex AI** | Google Cloud's ML platform. Includes model hosting, fine-tuning, RAG Engine, Agent Engine, evaluation tools. | The GCP way to deploy GenAI. Managed infrastructure, enterprise security, Gemini access. |
| **Bedrock** | AWS's managed GenAI service. Access to Claude, Llama, and others. Includes agents, knowledge bases, guardrails. | The AWS way to deploy GenAI. Similar capabilities to Vertex AI. |
| **Cloud Run** | Google Cloud's serverless container platform. Pay per request, auto-scales to zero. Good for bursty GenAI workloads. | Simple deployment for orchestration layers. Not for running LLMs (use GPUs), but good for the API/RAG layer. |
| **GKE** | Google Kubernetes Engine. Managed Kubernetes. Use for complex deployments that need more control than serverless. | Run vLLM or TGI on GKE with GPUs. More control than managed services, more ops burden. |

### Costs

| Term | Definition | Why it matters |
| ---- | ---------- | -------------- |
| **Per-token pricing** | LLM APIs charge by input + output tokens. Gemini 1.5 Flash: ~$0.075/1M input, ~$0.30/1M output. GPT-4o: ~$2.50/1M input, ~$10/1M output. | Output tokens cost 3-4× more than input. Long responses are expensive. Prompt engineering to reduce output saves money. |
| **Model Routing** | Sending easy requests to cheap/fast models, hard requests to expensive/capable models. E.g., simple FAQ → Flash, complex reasoning → Pro. | Can reduce costs 50-70% with minimal quality loss. Classify difficulty first, then route. |
| **Token Budget** | Maximum tokens you're willing to spend per request or per user session. Enforce limits to control costs. | Without budgets, runaway agents or verbose prompts can explode costs. Set and monitor token budgets. |

---

## B.1 GenAI System: Big Picture (Frontend to Backend)

This is the end-to-end shape of a GenAI system. Every request follows this path:

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
  │  Orchestration  │  RAG retrieval, agent logic, tool calls (E.2, E.4)
  │  (Agent / RAG)  │
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  LLM(s)         │  Inference, model routing (E.1)
  └────────┬────────┘
           │
           ▼
  Response (→ user, or → tools, then back into orchestration)
```

The remaining Part E sections (E.5–E.10) are cross-cutting concerns that surround this path: evaluation, data pipelines, cost, scale, monitoring, and security.

---

## B.2 GenAI vs Traditional ML

Understanding the fundamental differences between traditional ML systems and **GenAI** / **LLM (Large Language Model)** systems is crucial for making the right architectural decisions.

| Aspect         | Traditional ML       | GenAI/LLM                                |
| -------------- | -------------------- | ---------------------------------------- |
| **Prediction** | Single forward pass  | Token-by-token generation                |
| **Latency**    | Fixed (milliseconds) | Variable (seconds to minutes)            |
| **Memory**     | Model weights        | Model + KV cache (grows with sequence)   |
| **Batching**   | Static batches       | Dynamic/continuous batching              |
| **Cost**       | Per-request          | Per-token (input + output)               |
| **Control**    | Fixed weights        | Sampling parameters (temp, top-p, top-k) |

**Why these differences matter:**

- **Token-by-token generation** means you can't predict exact response time—a 10-token response is much faster than a 1000-token response.
- **KV cache growth** means memory requirements increase with context length, limiting how many concurrent requests you can serve.
- **Per-token pricing** means prompt engineering and response length directly impact costs.

> [!TIP]
> 💡 **Aha:** Traditional ML is "one input → one prediction." GenAI is "one prompt → a stream of tokens, each depending on the last." That shifts bottlenecks from GPU compute to memory (KV cache), latency (time-to-first-token vs total time), and cost (every token billed).

### Generative Algorithm Classes

Modern GenAI uses four main algorithm classes. Each has different strengths:

| Algorithm | How it works | Strengths | Weaknesses | Best for |
| --------- | ------------ | --------- | ---------- | -------- |
| **VAE** (Variational Autoencoder) | Encode to latent space → decode back | Fast sampling, smooth latent space | Blurry outputs | Latent representations, simple generation |
| **GAN** (Generative Adversarial Network) | Generator vs discriminator compete | Sharp, realistic outputs | Training instability, mode collapse | Face generation, image-to-image |
| **Diffusion** | Learn to reverse noise → image | Highest quality, stable training | Slow sampling (many steps) | Text-to-image (DALL-E, Stable Diffusion, Imagen) |
| **Autoregressive** | Predict next token given previous | Handles sequences, scales well | Sequential = slow; can't "look ahead" | LLMs (GPT, Gemini, Claude), text generation |

> [!TIP]
> 💡 **Aha:** In interviews, when asked "design a text-to-image system," diffusion is the default choice (quality). For LLMs/chatbots, autoregressive Transformers are the default. GANs are rarely used for new systems due to training instability; VAEs are used for latent representations (e.g., Stable Diffusion's VAE encoder).

### GAN Architecture Deep Dive

GANs have two competing networks trained together:

**Generator Architecture:**
```
Noise Vector (100-dim) → Reshape → [Upsampling Blocks] → Output Image
                                          ↓
                         ConvTranspose2D → BatchNorm → ReLU (repeat)
                                          ↓
                         Final: ConvTranspose2D → Tanh (-1 to 1)
```

**Discriminator Architecture:**
```
Input Image → [Downsampling Blocks] → Classification Head → Probability (real/fake)
                    ↓                          ↓
         Conv2D → BatchNorm → LeakyReLU    Fully Connected → Sigmoid
```

| Component | Generator | Discriminator |
| --------- | --------- | ------------- |
| **Purpose** | Transform noise → image | Classify real vs fake |
| **Convolution** | Transposed Conv (upsample) | Standard Conv (downsample) |
| **Final activation** | Tanh (output in [-1, 1]) | Sigmoid (probability) |

### Normalization Layers

Normalization stabilizes training by ensuring consistent distributions across layers.

| Type | Normalizes across | Best for | Notes |
| ---- | ----------------- | -------- | ----- |
| **Batch Norm (BN)** | Batch dimension | CNNs, GANs | Standard choice; needs decent batch size |
| **Layer Norm (LN)** | Feature dimension | Transformers, RNNs | Batch-size independent |
| **Instance Norm (IN)** | Each feature map individually | Style transfer | Removes style information |
| **Group Norm (GN)** | Groups of channels | Small batch sizes | Balance between BN and LN |

### GAN Training: Adversarial Loss

**Minimax objective:** Discriminator maximizes, Generator minimizes:

```
L = E[log D(x)] + E[log(1 - D(G(z)))]
     ↑ real          ↑ fake
```

**Training loop:**
1. Train discriminator for k steps (generator frozen)
2. Train generator for 1 step (discriminator frozen)
3. Repeat until convergence

### GAN Training Challenges & Mitigations

| Challenge | What happens | Mitigations |
| --------- | ------------ | ----------- |
| **Vanishing gradients** | Discriminator too good → generator gets tiny gradients | Modified loss: maximize log(D(G(z))) instead of minimize log(1-D(G(z))); Wasserstein loss |
| **Mode collapse** | Generator produces only 1–2 image types | Wasserstein loss; Unrolled GAN; minibatch discrimination |
| **Failure to converge** | Discriminator/generator oscillate, never stabilize | Different learning rates; spectral normalization; gradient penalty |

**Wasserstein GAN (WGAN):**
- Discriminator (critic) outputs score, not probability
- Critic loss = D(real) - D(fake) (maximize)
- Generator loss = -D(G(z)) (minimize)
- More stable gradients; reduces mode collapse

### GAN Latent Space & Sampling

**Latent space:** The generator learns a mapping from noise vectors to images. Points in this space represent potential images; nearby points → similar images.

| Sampling method | How it works | Trade-off |
| --------------- | ------------ | --------- |
| **Random** | Sample from N(0,1) | Maximum diversity; may include outliers |
| **Truncated** | Reject samples beyond threshold | Higher quality; less diversity |

**StyleGAN** extends this with style-based generation:
- Separate "style" vectors control different attributes (age, hair, expression)
- Enables **attribute manipulation**: change age without changing identity
- Used for face generation, editing, and deepfakes

### Image Generation Metrics

| Metric | What it measures | How it works | Interpretation |
| ------ | ---------------- | ------------ | -------------- |
| **Inception Score (IS)** | Quality + diversity | Run images through Inception v3; measure class probability sharpness and diversity | Higher = better (quality: sharp predictions; diversity: spread across classes) |
| **FID** (Fréchet Inception Distance) | Similarity to real images | Compare feature statistics (mean, covariance) of generated vs real images | Lower = better (distributions closer) |
| **KID** (Kernel Inception Distance) | Like FID, unbiased | Uses kernel methods instead of Gaussian assumption | Lower = better |
| **CLIP Score** | Image-text alignment | Cosine similarity between CLIP embeddings | Higher = better match to prompt |

**FID calculation:**
1. Generate large set of images
2. Extract features from Inception v3 (both real and generated)
3. Compute mean and covariance for each set
4. Calculate Fréchet distance between distributions

> [!TIP]
> 💡 **Aha:** FID and IS use ImageNet-trained Inception, which can introduce artifacts. **CLIP-based metrics** (e.g., CLIP-FID) often align better with human judgment. For face generation, **human evaluation** (pairwise comparison: "which looks more real?") is still the gold standard.

### Diffusion Model Architecture

Diffusion models iteratively denoise images. Two main architectures:

**U-Net Architecture:**
```
Noisy Image → [Downsampling Blocks] → Bottleneck → [Upsampling Blocks] → Predicted Noise
                      ↓                                    ↑
              Conv2D → BatchNorm → ReLU            ConvTranspose2D → BatchNorm → ReLU
              → MaxPool → Cross-Attention          → Cross-Attention (to text)
```

**DiT (Diffusion Transformer) Architecture:**
```
Noisy Image → Patchify → Positional Encoding → Transformer Blocks → Unpatchify → Predicted Noise
                                    ↑
                              Text Conditioning
```

| Architecture | How it works | Pros | Cons | Examples |
| ------------ | ------------ | ---- | ---- | -------- |
| **U-Net** | CNN-based; downsampling + upsampling with skip connections | Proven; efficient for images | Limited to fixed resolution | Stable Diffusion, Imagen |
| **DiT** | Transformer-based; patches like ViT | Scales better; flexible | More compute | Sora, newer models |

**Cross-attention in diffusion:** Queries from image features; keys/values from text embeddings. Allows text to guide noise prediction at each step.

### Diffusion Training Process

**Forward process (noise addition):**
```
x_0 (clean) → x_1 → x_2 → ... → x_T (pure noise)
```
- Add Gaussian noise at each step according to **noise schedule** (β₁ < β₂ < ... < βₜ)
- Can compute x_t directly from x_0: `x_t = √(α'_t) * x_0 + √(1-α'_t) * ε`

**Backward process (denoising):**
```
x_T (noise) → x_{T-1} → ... → x_1 → x_0 (clean)
```
- Model predicts noise ε at each step
- Subtract predicted noise to get cleaner image

**Loss function:** MSE between true noise and predicted noise:
```
L = E[||ε - ε_θ(x_t, t, text)||²]
```

| Component | Purpose |
| --------- | ------- |
| **Noise schedule** | Controls how much noise at each timestep (typically 1000 steps) |
| **Timestep embedding** | Tells model current noise level |
| **Text conditioning** | CLIP or T5 encodes prompt; cross-attention injects into model |

### Diffusion Sampling Techniques

| Technique | What it does | Benefit |
| --------- | ------------ | ------- |
| **DDPM** (original) | 1000 steps, predict noise at each | High quality; very slow |
| **DDIM** | Deterministic; skip steps (1000 → 20–50) | Much faster; slight quality loss |
| **Classifier-Free Guidance (CFG)** | Blend conditioned and unconditioned predictions | Better text alignment |

**CFG formula:**
```
ε_guided = ε_uncond + w * (ε_cond - ε_uncond)
```
- w = guidance scale (typically 7–15)
- Higher w = stronger text adherence, less diversity
- w = 1 = no guidance; w > 1 = amplify text condition

> [!TIP]
> 💡 **Aha:** **CFG** is why "a cat on a skateboard" actually shows a cat on a skateboard. Without it, diffusion models often ignore parts of the prompt. The guidance scale w trades off **text fidelity** (high w) vs **image diversity** (low w).

### Diffusion Training Challenges & Mitigations

| Challenge | Problem | Mitigations |
| --------- | ------- | ----------- |
| **High memory** | Billions of params + high-res images | Mixed precision (FP16/BF16); gradient checkpointing |
| **Slow training** | Many timesteps; large models | Data/model parallelism (FSDP, DeepSpeed) |
| **Slow sampling** | 1000 steps per image | DDIM (20–50 steps); consistency models; distillation |
| **High-res generation** | Directly training at 1024² is expensive | **Latent diffusion**: train in VAE latent space, then decode |

**Latent Diffusion (Stable Diffusion approach):**
1. Train VAE to compress images to latent space (64×64 instead of 512×512)
2. Train diffusion model in latent space (much cheaper)
3. Decode latent → high-res image

**Super-resolution cascade:**
```
Prompt → Diffusion (64×64) → SR Model #1 (256×256) → SR Model #2 (1024×1024)
```
- Train base model at low resolution
- Train separate super-resolution models to upscale
- Faster training; easier to scale

### Text-to-Image Inference Pipeline

```
User Prompt → [Prompt Safety] → [Prompt Enhancement] → [Text Encoder (CLIP/T5)]
                    ↓                                           ↓
               Reject if unsafe                          Text Embeddings
                                                                ↓
                                            [Diffusion Model + CFG] → [Harm Detection]
                                                                           ↓
                                                                  [Super-Resolution]
                                                                           ↓
                                                                     Final Image
```

| Component | Purpose |
| --------- | ------- |
| **Prompt auto-complete** | Suggest completions as user types |
| **Prompt safety** | Text classifier rejects violence, NSFW, etc. |
| **Prompt enhancement** | LLM expands "a dog" → "a golden retriever sitting on grass, sunny day..." |
| **Text encoder** | CLIP or T5 converts text to embeddings |
| **Diffusion model** | Generates image from noise + text embeddings |
| **Harm detection** | Image classifier catches unsafe outputs |
| **Super-resolution** | Upscales low-res output to target resolution |

### CLIPScore for Image-Text Alignment

**CLIP** (Contrastive Language-Image Pretraining):
- Dual encoder: text encoder + image encoder
- Trained to bring matching (image, text) pairs close in embedding space

**CLIPScore:**
```
CLIPScore = cosine_similarity(CLIP_text(prompt), CLIP_image(generated_image))
```
- Higher = better alignment between generated image and prompt
- Reference-free (no ground-truth image needed)
- Standard metric for text-to-image evaluation

| Evaluation aspect | Metric |
| ----------------- | ------ |
| **Image quality** | FID, IS, human eval |
| **Image diversity** | IS (class spread), FID |
| **Text alignment** | CLIPScore, human eval |

> [!TIP]
> 💡 **Aha:** For text-to-image, you need **both** quality metrics (FID) **and** alignment metrics (CLIPScore). A model could generate beautiful images that ignore the prompt (low CLIPScore, good FID) or follow the prompt but look bad (high CLIPScore, poor FID).

---

## C.1 Text-to-Video Generation

Text-to-video extends text-to-image by generating sequences of temporally consistent frames.

### Latent Diffusion Models (LDM)

**Problem:** Video is expensive. A 5-second 720p video at 24 FPS = 120 frames × 1280×720 pixels = ~110M pixels.

**Solution:** Train diffusion model in **latent space** instead of pixel space.

```
Original Video → VAE Encoder → Latent Representation (compressed) → Diffusion Model → Denoised Latent → VAE Decoder → Generated Video
```

**Compression network (VAE):**
- **Visual Encoder**: Video pixels → lower-dimensional latent representation
- **Visual Decoder**: Latent representation → reconstructed video

**Compression ratio example:**
- Input: 120 frames × 1280×720 = 110M pixels
- With 8× temporal + 8× spatial compression: 15 frames × 160×90 = 216K
- **512× smaller** → much cheaper training and inference

| Approach | Operates in | Training cost | Examples |
| -------- | ----------- | ------------- | -------- |
| **Pixel diffusion** | Full resolution pixels | Very expensive | Imagen Video |
| **Latent diffusion** | Compressed latent space | Much cheaper | Stable Diffusion, Sora, Movie Gen |

### Extending DiT to Video

**Image DiT:** 2D patches (spatial only)
**Video DiT:** 3D patches (spatial + temporal)

```
Video → 3D Patchify → Positional Encoding (3D: x, y, t) → Transformer + Temporal Layers → Unpatchify → Predicted Noise
```

**Temporal layers added to architecture:**

| Layer | How it works | Purpose |
| ----- | ------------ | ------- |
| **Temporal Attention** | Each feature attends across frames | Capture motion, ensure consistency |
| **Temporal Convolution** | 3D conv across frames | Local temporal patterns |

**U-Net for video:** Inject temporal attention + temporal convolution into each downsampling/upsampling block.

**DiT for video:** Use 3D patches; Transformer naturally handles sequences; RoPE for 3D positional encoding.

### Video Training Challenges

| Challenge | Problem | Mitigations |
| --------- | ------- | ----------- |
| **Limited video-text data** | Much less paired video-text than image-text | Train on both images (as 1-frame videos) + videos; pretrain on images, finetune on videos |
| **High compute cost** | 120 frames vs 1 frame | Latent diffusion; precompute latents; spatial/temporal super-resolution |
| **High-res generation** | 720p+ is expensive | Generate at lower res (e.g., 360p), upscale with super-resolution models |
| **Long videos** | More frames = more memory | Generate short clips, stitch; temporal super-resolution |

**Two training strategies:**
1. **Joint training**: Train DiT on mixed image-text + video-text data (treat images as 1-frame videos)
2. **Two-stage**: Pretrain on images → finetune on videos

**Super-resolution cascade for video:**
```
LDM (40×23 @ 8 fps) → Visual Decoder → Spatial SR (320×180) → Temporal SR (24 fps) → Final Video (1280×720 @ 24 fps)
```

### Video Evaluation Metrics

| Metric | What it measures | How it works |
| ------ | ---------------- | ------------ |
| **FID (per-frame)** | Frame quality | Average FID across all frames |
| **IS (per-frame)** | Frame quality + diversity | Average Inception Score |
| **FVD** (Fréchet Video Distance) | Quality + temporal consistency | Like FID, but uses I3D features (action recognition model) |
| **CLIP similarity** | Video-text alignment | Average CLIP score across frames |

**FVD calculation:**
1. Extract features from I3D model (trained for action recognition)
2. Compute mean + covariance for generated and real videos
3. Calculate Fréchet distance between distributions
4. Lower FVD = better (quality + temporal consistency)

**Benchmarks:** VBench, Movie Gen Bench

> [!TIP]
> 💡 **Aha:** FID only measures individual frame quality—a video could have great frames but terrible motion. **FVD** captures **temporal consistency** by using an action recognition model (I3D) that understands motion. Always report FVD for video generation.

### Video Inference Pipeline

```
Prompt → Safety → Enhancement → Text Encoder → LDM (latent space)
                                                      ↓
                                               Visual Decoder → Spatial SR → Temporal SR → Harm Detection → Final Video
```

| Component | Purpose |
| --------- | ------- |
| **LDM** | Generate video in latent space (cheaper than pixel space) |
| **Visual Decoder** | Convert latent → pixel space |
| **Spatial Super-Resolution** | Upscale resolution (e.g., 360p → 720p) |
| **Temporal Super-Resolution** | Interpolate frames (e.g., 8 fps → 24 fps) |

**Models to know:**
- **Sora** (OpenAI): DiT-based; variable duration/resolution; "world simulator"
- **Movie Gen** (Meta): DiT + LDM; 16s videos at 768p
- **Stable Video Diffusion** (Stability AI): U-Net based; image-to-video
- **Runway Gen-3**: Commercial; fast iteration
- **Imagen Video** (Google): Pixel-space cascade; high quality

---

### Model Capacity: Parameters vs FLOPs

**Model capacity** determines how much a model can learn. Two measures:

| Measure | What it means | Example |
| ------- | ------------- | ------- |
| **Parameters** | Learnable weights in the model | GPT-4: ~1.8T params; Llama 3: 405B params; Gemini Ultra: ~1T params |
| **FLOPs** | Floating-point operations per forward pass | Measures computational complexity, not just size |

**Why this matters for interviews:** Larger models generally perform better but cost more to train and serve. Training cost scales with **FLOPs** (compute); serving cost scales with **parameters** (memory) and tokens.

### Scaling Laws

**Scaling laws** predict model performance from compute, data, and parameters—critical for planning large training runs.

**OpenAI (2020):** Performance improves predictably with scale. Loss follows a power law:
- More compute → lower loss
- More data → lower loss
- More parameters → lower loss

**DeepMind Chinchilla (2022):** Many LLMs were **undertrained**. Optimal scaling: **data should scale linearly with model size**. A 70B model trained on 1.4T tokens outperforms a 280B model trained on 300B tokens.

| Insight | Implication |
| ------- | ----------- |
| Scale matters more than architecture tweaks | Focus on data + compute, not micro-optimizations |
| Data and parameters should scale together | Don't just make models bigger—feed them more data |
| Compute-optimal training | Given a compute budget, there's an optimal (N, D) pair |

**Inference-time scaling (2024+):** With models like GPT o1, researchers are exploring scaling at inference time (e.g., chain-of-thought, repeated sampling) to improve reasoning.

> [!TIP]
> 💡 **Aha:** When an interviewer asks "how would you improve this model?", scaling laws say: **more data and compute** often beat architecture changes. But for deployment, you often want **smaller models** (distillation, quantization) to reduce cost.

---

## C.2 Multimodal & Vision-Language Models

Many GenAI tasks involve **multiple modalities** (text + image, text + video, etc.). Key architectures:

### Image Encoders

| Architecture | How it works | Pros | Cons | Examples |
| ------------ | ------------ | ---- | ---- | -------- |
| **CNN-based** | Convolutional filters detect patterns; output = feature grid | Fast; good for local patterns | Weak on long-range dependencies | ResNet, EfficientNet |
| **Transformer-based (ViT)** | Patchify image → positional encoding → Transformer | Global attention; scales well | More compute; needs more data | ViT, CLIP, DINOv2 |

**ViT (Vision Transformer) Process:**
1. **Patchify**: Divide image into fixed patches (e.g., 256×256 → 16 patches of 64×64)
2. **Flatten + Project**: Each patch → linear projection to embedding vector
3. **Positional Encoding**: Add position info (1D or 2D)
4. **Transformer**: Self-attention across patches → sequence of embeddings

**Positional Encoding for Images:**

| Type | How it works | Use case |
| ---- | ------------ | -------- |
| **1D** | Position in flattened sequence | Simple; may lose 2D spatial info |
| **2D** | Row + column position | Preserves spatial structure |
| **Learnable** | Learned during training | Task-optimized; may overfit |
| **Fixed** (sine-cosine) | Computed from position | Generalizes to new sizes |

### Encoder Output: Single Token vs Sequence

| Output | Description | Pros | Cons |
| ------ | ----------- | ---- | ---- |
| **Single token** | Entire image compressed to one vector | Simple; less compute | Loses local details; generic captions |
| **Sequence of tokens** | Each token = patch/region of image | Rich detail; works with cross-attention | More tokens; more memory |

> [!TIP]
> 💡 **Aha:** For **image captioning** and **VQA**, use **sequence output** (one embedding per patch). Cross-attention in the decoder can then focus on relevant image regions for each output word. Single-token outputs work for simple classification but lose detail for generation.

### Vision-Language Models

| Model | Architecture | Use Cases |
| ----- | ------------ | --------- |
| **CLIP** | Dual encoder (image + text); contrastive learning | Image-text similarity, zero-shot classification, filtering |
| **ViT** | Image encoder (patches → Transformer) | Feature extraction, image classification |
| **BLIP-2/BLIP-3** | Frozen image encoder + LLM + Q-Former bridge | Image captioning, VQA, multimodal chat |
| **LLaVA** | ViT encoder + LLM decoder | Multimodal chat, image understanding |
| **Gemini** | Native multimodal (image, text, audio, video) | General-purpose multimodal |

### Image Captioning Architecture

```
Input Image → Image Encoder (ViT) → Sequence of Embeddings → Text Decoder (GPT-style) → Caption
                                              ↓
                                    Cross-Attention (decoder attends to image)
```

**Key components:**
1. **Image Encoder**: ViT or CLIP encoder → sequence of patch embeddings
2. **Text Decoder**: Decoder-only Transformer (GPT-2, LLaMA)
3. **Cross-Attention**: Decoder attends to image embeddings at each generation step

**Training:**
1. **Pretrain encoder** (CLIP, ViT) on image classification or contrastive learning
2. **Pretrain decoder** (GPT) on text
3. **Finetune together** on image-caption pairs (next-token prediction, cross-entropy loss)

### CIDEr Metric (Image Captioning)

CIDEr (Consensus-based Image Description Evaluation) is designed specifically for image captioning:

1. **TF-IDF representation**: Convert captions to vectors based on word importance
2. **Cosine similarity**: Compare generated caption to each reference caption
3. **Average**: Final score = mean similarity across all references

| Metric | Focus | Best For |
| ------ | ----- | -------- |
| **BLEU** | N-gram precision | Translation, short text |
| **ROUGE** | N-gram recall | Summarization |
| **METEOR** | Precision + recall + synonyms | Translation with paraphrasing |
| **CIDEr** | Consensus across multiple references | Image captioning |

> [!TIP]
> 💡 **Aha:** CIDEr rewards captions that match **multiple** reference captions (consensus). BLEU/ROUGE only compare to one reference at a time. For image captioning with 3–5 reference captions per image, CIDEr is the standard metric.

---

## D.1 Using Models & Sampling Parameters

Generative AI agents are powered by models that act as the "brains" of the operation. While models are pre-trained, their behavior during inference can be customized using **sampling parameters**—the "knobs and dials" of the model.

### Common Sampling Parameters

**1. Temperature**

Controls the "creativity" or randomness of the output by rescaling logits before softmax.

- **High Temperature (T > 1)**: Flattens the distribution, making output more random, diverse, and unpredictable. Increases risk of incoherence.
- **Low Temperature (T < 1)**: Sharpens the distribution, making it more focused, deterministic, and repeatable.
- **Extreme (T → 0)**: Collapses into greedy decoding (always picks the highest probability token).

_Use low temperature (0.1-0.3) for factual tasks, higher (0.7-1.0) for creative tasks._

> [!TIP]
> 💡 **Aha:** Temperature rescales logits before sampling. Low T makes the top token dominate (nearly deterministic); high T flattens the distribution so unlikely tokens get a real chance. You're tuning "how much to trust the model's confidence."

**2. Top-p (Nucleus Sampling)**

Selects the smallest set of tokens whose cumulative probability mass reaches threshold _p_.

- **High Top-p (0.9-1.0)**: Allows for more diversity by extending to lower probability tokens.
- **Low Top-p (0.1-0.5)**: Leads to more focused responses.
- **Adaptive**: Unlike Top-K, adapts to the distribution's shape—in confident contexts, the "nucleus" is small.

> [!TIP]
> 💡 **Aha:** Top-p says "consider only tokens that together account for probability mass _p_." When the model is sure, that might be 2–3 tokens; when unsure, many more. So Top-p scales with confidence; Top-K does not.

**3. Top-K**

Restricts the model's choice to only the _k_ most probable tokens at each step.

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

## D.2 Google Generative AI Development Tools

Google provides two primary environments for experimenting with and deploying Gemini models:

| Attribute        | Google AI Studio                                                                                   | Vertex AI Studio                                                          |
| :--------------- | :------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------ |
| **Focus**        | Streamlined, easy-to-use interface for rapid prototyping                                           | Comprehensive environment for building, training, and deploying ML models |
| **Target Users** | Beginners, hobbyists, initial project stages                                                       | Professionals, researchers, enterprise developers                         |
| **Access**       | Standard Google Account login                                                                      | Google Cloud Console (Enterprise account)                                 |
| **Limitations**  | Usage limits (**QPM** queries/min, **RPM** requests/min, **TPM** tokens/min); small-scale projects | Service charges based on usage; enterprise-grade quotas                   |
| **Advantages**   | Simplified interface; easy to get started                                                          | Enterprise-grade security, compliance, flexible quotas                    |

**Key Takeaway**: Use **Google AI Studio** for fast, small-scale prototyping. Transition to **Vertex AI Studio** for large-scale, production-ready enterprise applications.

### Agent Development Kit (ADK)

**ADK** is Google's open-source framework for building AI agents. It's the recommended way to build multi-agent systems on GCP.

| Tool | What it does | When to use |
| ---- | ------------ | ----------- |
| **Google AI Studio** | Prompt playground; quick prototyping | Experimenting with prompts |
| **Vertex AI Studio** | Enterprise model access; fine-tuning; evaluation | Production workloads |
| **ADK** | Agent framework; multi-agent orchestration | Building agents with tools, workflows, multi-agent coordination |
| **Vertex AI Agent Engine** | Managed agent hosting | Deploying ADK agents at scale |

**ADK installation:**
```bash
pip install google-adk   # Python
npm install @google/adk  # TypeScript
```

**Quick start:**
```bash
adk create my_agent   # Scaffold project
adk run my_agent      # Run locally
adk web               # Local dev UI
```

See **E.4 Agentic AI Systems** for full ADK coverage with code examples.

---

## D.3 Text Tokenization Strategies

Tokenization converts raw text into numerical tokens the model can process. The choice of tokenization affects vocabulary size, model performance, and handling of unseen words.

### Tokenization Levels

| Level | How it works | Vocabulary Size | Pros | Cons |
| ----- | ------------ | --------------- | ---- | ---- |
| **Character** | Split into individual characters | ~100 | Small vocab; handles any word | Hard to learn semantics; slow (many tokens) |
| **Word** | Split on whitespace/punctuation | ~300,000+ | Easy semantics; fewer tokens | Huge vocab; can't handle unseen words |
| **Subword** | Frequent words stay whole; rare words split into subwords | ~50,000–150,000 | Best of both; handles unseen words | More complex algorithms |

### Subword Algorithms (Industry Standard)

| Algorithm | Used By | How it works |
| --------- | ------- | ------------ |
| **BPE** (Byte-Pair Encoding) | GPT-4, LLaMA | Iteratively merge most frequent character pairs |
| **SentencePiece** | Gemini, T5 | Language-agnostic; works directly on raw text |
| **WordPiece** | BERT | Similar to BPE; maximizes likelihood of training data |

> [!TIP]
> 💡 **Aha:** Subword tokenization solves two problems: (1) vocabulary explosion from word-level, and (2) semantic loss from character-level. "unhappily" becomes ["un", "happy", "ly"]—each subword has meaning the model can learn.

---

## D.4 Transformer Architectures

The Transformer architecture has three variations, each suited for different tasks:

| Variation | How it works | Best For | Examples |
| --------- | ------------ | -------- | -------- |
| **Encoder-only** | Processes entire input; outputs understanding/classification | Sentiment analysis, NER, classification | BERT, RoBERTa |
| **Decoder-only** | Generates output token-by-token autoregressively | Text generation, chatbots, code completion | GPT-4, Gemini, LLaMA, Claude |
| **Encoder-Decoder** | Encoder processes input; decoder generates transformed output | Translation, summarization | T5, BART |

**Key Components of a Decoder-only Transformer:**

1. **Text Embedding**: Converts token IDs to dense vectors (learned during training). Captures semantic similarity—"happy" and "joyful" are close in embedding space.

2. **Positional Encoding**: Adds position information since attention is permutation-invariant.
   - **Fixed** (sine-cosine): No extra parameters; generalizes to longer sequences
   - **Learned**: Optimized for task; may overfit to training sequence lengths

3. **Multi-Head Self-Attention**: Each token attends to all previous tokens (in decoder) or all tokens (in encoder). Multiple "heads" capture different relationship types.

4. **Feed-Forward Network**: Two linear layers with ReLU; applied independently to each position.

5. **Prediction Head**: Maps final embeddings to vocabulary probabilities for next-token prediction.

> [!TIP]
> 💡 **Aha:** For **generation tasks** (chatbots, code completion, Smart Compose), use **decoder-only**. For **understanding tasks** (classification, entity extraction), use **encoder-only**. For **transformation tasks** (translation, summarization), use **encoder-decoder**.

### Encoder-Decoder Architecture (for Seq2Seq)

For tasks where input is **transformed** into output (translation, summarization), encoder-decoder is preferred:

**Why encoder-decoder for translation?**
1. **Separation of concerns**: Encoder specializes in understanding source language; decoder generates target language
2. **Bidirectional encoding**: Encoder processes full input with bidirectional attention before generation starts
3. **Cross-attention**: Decoder can focus on relevant parts of input during each output step
4. **Variable-length I/O**: Naturally handles input/output of different lengths

**Key difference: Cross-Attention**

In encoder-decoder models, the decoder has an additional **cross-attention** layer that attends to encoder outputs:

```
Encoder: Input → Self-Attention → Encoder Output (context vectors)
                                         ↓
Decoder: Previous Output → Self-Attention → Cross-Attention (to encoder) → Prediction
```

- **Self-attention in encoder**: Each token attends to ALL tokens (bidirectional)
- **Self-attention in decoder**: Each token attends only to PREVIOUS tokens (causal/masked)
- **Cross-attention**: Each decoder token attends to ALL encoder outputs

> [!TIP]
> 💡 **Aha:** Cross-attention is the "bridge" between encoder and decoder. It lets the decoder ask "which parts of the input should I focus on for this output token?" For translation, generating "bonjour" attends heavily to "hello" in the encoder output.

---

## D.5 ML Objectives for Pretraining

Different architectures use different pretraining objectives:

| Architecture | Pretraining Objective | How it works |
| ------------ | --------------------- | ------------ |
| **Decoder-only** | Next-token prediction | Predict `x_i` given `x_1...x_{i-1}` |
| **Encoder-only** | Masked Language Modeling (MLM) | Predict [MASK] tokens given surrounding context |
| **Encoder-decoder** | MLM or Span Corruption | Mask spans in input; decoder predicts masked spans |

### Masked Language Modeling (MLM)

Used by BERT and encoder-decoder models (T5, BART). Randomly mask 15% of tokens; model predicts the originals.

**Why MLM for encoder-decoder?**
- Next-token prediction would let encoder "cheat" by encoding the answer
- MLM forces encoder to build deep understanding without seeing the masked tokens
- Decoder learns to generate based on incomplete information

**Example:**
```
Input:  "Thank [MASK] for inviting [MASK]"
Target: "you", "me"
```

**Span Corruption (T5 style):**
```
Input:  "Thank <X> inviting <Y>"  (masked spans)
Target: "<X> you for <Y> me"
```

> [!TIP]
> 💡 **Aha:** For **decoder-only** (GPT, Gemini), use **next-token prediction**. For **encoder-only** (BERT), use **MLM**. For **encoder-decoder** (T5), use **span corruption**. The objective shapes what the model learns.

---

## D.6 Two-Stage Training: Pretraining + Finetuning

Training LLMs directly on task-specific data is inefficient. Instead, use two stages:

| Stage | Data | Purpose | Compute |
| ----- | ---- | ------- | ------- |
| **Pretraining** | Massive general data (web, books) | Learn language structure, grammar, world knowledge | Very expensive (weeks on 1000s of GPUs) |
| **Finetuning** | Task-specific data (emails, code, medical) | Adapt to specific task, style, domain | Cheaper (hours to days on fewer GPUs) |

**Benefits of Two-Stage Training:**

- **Transfer learning**: Knowledge from pretraining transfers to finetuning
- **Data efficiency**: Performs well even with limited task-specific data
- **Reduced overfitting**: Pretraining acts as regularization
- **Resource optimization**: Pretrain once, finetune for many tasks
- **Fast adaptation**: Finetuning is much faster than training from scratch

**ML Objective**: Next-token prediction (predict `x_i` given `x_1, ..., x_{i-1}`)

**Loss Function**: Cross-entropy loss between predicted and actual next token

> [!TIP]
> 💡 **Aha:** You almost never train an LLM from scratch. You take a **pretrained base model** (GPT, LLaMA, Gemini) and **finetune** it on your domain data. This is why foundation models are so valuable—they encode billions of dollars of pretraining compute.

---

## D.7 Three-Stage Training for Chatbots (Pretraining → SFT → RLHF)

For **chatbots** (ChatGPT, Gemini, Claude), two stages aren't enough. A third stage aligns the model to human preferences:

| Stage | Data | Purpose | Compute | Outcome |
| ----- | ---- | ------- | ------- | ------- |
| **1. Pretraining** | Web, books (trillions of tokens) | Learn language, world knowledge | Very expensive (months, 1000s GPUs) | Base model (continues text) |
| **2. SFT** (Supervised Finetuning) | (prompt, response) pairs (10K–100K) | Learn to respond to prompts, not just continue | Cheaper (days, 10–100 GPUs) | SFT model (answers prompts) |
| **3. RLHF** (Reinforcement Learning from Human Feedback) | Human preference rankings | Align to human preferences (helpful, harmless) | Moderate (days, 10–100 GPUs) | Final chatbot |

### Stage 2: Supervised Finetuning (SFT)

**Demonstration data**: High-quality (prompt, response) pairs created by educated humans (often 30%+ with master's degrees for accuracy).

| Dataset | Size | Notes |
| ------- | ---- | ----- |
| InstructGPT | ~14,500 | OpenAI's original instruction dataset |
| Alpaca | 52,000 | Stanford; GPT-generated |
| Dolly-15K | ~15,000 | Databricks; open-source |
| FLAN 2022 | ~104,000 | Google; multi-task |

**ML Objective**: Same as pretraining—next-token prediction, cross-entropy loss. But now on (prompt, response) format.

**Outcome**: SFT model responds to prompts instead of just continuing text. But responses may not be optimal—just plausible.

### Stage 3: RLHF (Alignment)

The SFT model produces plausible responses, but not necessarily the **best** response. RLHF aligns the model to human preferences.

**Step 3.1: Train a Reward Model**

1. **Generate responses**: SFT model generates multiple responses per prompt
2. **Human ranking**: Contractors rank responses (easier than scoring)
3. **Create preference pairs**: (prompt, winning response, losing response)
4. **Train reward model**: Predicts score for (prompt, response); trained to maximize `S_win - S_lose`

**Loss function (margin ranking):**
```
L = max(0, margin - (S_win - S_lose))
```
If the gap between winning and losing scores is less than the margin, the loss is positive → model updates.

**Step 3.2: Optimize SFT Model with RL**

1. **Generate responses**: SFT model generates responses
2. **Score with reward model**: Get helpfulness score
3. **Update with PPO**: Reinforce responses that get high reward scores

**Common RL algorithms:** PPO (Proximal Policy Optimization), DPO (Direct Preference Optimization)

> [!TIP]
> 💡 **Aha:** RLHF is why ChatGPT feels "helpful" and "safe" compared to raw GPT-3. The base model knows a lot but doesn't know what humans want. RLHF teaches it to prefer helpful, harmless responses.

### Rotary Positional Encoding (RoPE)

For long-context chatbots (4K+ tokens), traditional positional encodings struggle. **RoPE** (used by LLaMA, Gemini) encodes position as rotation in embedding space:

**Absolute vs Relative Positional Encoding:**

| Type | How it works | Limitation |
| ---- | ------------ | ---------- |
| **Absolute** (sinusoidal, learned) | Each position has unique vector added to embedding | Doesn't capture relative distances; struggles to generalize to longer sequences |
| **Relative** (T5, DeBERTa) | Encodes distance between tokens, not absolute position | More complex; can't use efficient linear attention |
| **RoPE** | Rotates embeddings by position angle; relative distance = angle difference | Best of both; efficient; generalizes well |

**Why RoPE is better:**
- **Translational invariance**: Same relative distance = same angle, regardless of absolute position
- **Generalizes to unseen lengths**: Rotation maintains consistent relationships
- **Efficient**: Can use standard attention implementations

---

## D.8 Sampling Strategies for Text Generation

After training, **sampling** generates new text from the model. Two main categories:

### Deterministic vs Stochastic

| Type | How it works | Pros | Cons | Best For |
| ---- | ------------ | ---- | ---- | -------- |
| **Deterministic** | Always pick highest probability token(s) | Consistent, reproducible | Repetitive, lacks diversity | Email completion, code, factual Q&A |
| **Stochastic** | Sample from probability distribution | Diverse, creative | Inconsistent, may produce nonsense | Creative writing, dialogue, brainstorming |

### Deterministic Methods

| Method | How it works | Pros | Cons |
| ------ | ------------ | ---- | ---- |
| **Greedy Search** | Always pick the single highest-probability token | Simple, fast | Often repetitive; misses better sequences |
| **Beam Search** | Track top-k sequences simultaneously; prune at each step | Better quality than greedy; finds coherent sequences | Computationally expensive; limited diversity |

**Beam Search Example (beam width = 3):**
1. Start with input → get top 3 next tokens
2. For each of 3 sequences → get top 3 next tokens (9 candidates)
3. Keep only top 3 sequences by cumulative probability
4. Repeat until `<EOS>` or max length
5. Return sequence with highest cumulative probability

### Stochastic Methods

| Method | How it works | Use Case |
| ------ | ------------ | -------- |
| **Random Sampling** | Sample according to full probability distribution | Maximum diversity |
| **Top-K Sampling** | Sample only from top K tokens | Balance diversity and quality |
| **Top-p (Nucleus)** | Sample from smallest set with cumulative probability ≥ p | Adaptive diversity |
| **Temperature Scaling** | Adjust distribution sharpness before sampling | Control creativity |

> [!TIP]
> 💡 **Aha:** For **autocomplete** (Smart Compose, code completion), use **beam search** (deterministic, consistent). For **chatbots** and **creative generation**, use **Top-p + Temperature** (stochastic, diverse). The choice depends on whether users expect the same answer every time.

---

## D.9 Text Generation Evaluation Metrics

### Offline Metrics

| Metric | What it measures | Formula/Method | Lower/Higher is better |
| ------ | ---------------- | -------------- | ---------------------- |
| **Perplexity** | How "surprised" the model is by the test data | `exp(-1/N * Σ log P(x_i | x_{1:i-1}))` | **Lower** = better |
| **ExactMatch@N** | % of N-word predictions that exactly match ground truth | `(correct N-word matches) / (total predictions)` | **Higher** = better |
| **BLEU** | N-gram precision vs reference text | Geometric mean of n-gram precisions | **Higher** = better |
| **ROUGE-N** | N-gram recall vs reference text | `(matching n-grams) / (reference n-grams)` | **Higher** = better |
| **ROUGE-L** | Longest common subsequence with reference | LCS-based F1 score | **Higher** = better |
| **METEOR** | Precision + recall with synonyms/stemming | Weighted harmonic mean with synonym matching | **Higher** = better |

### Translation Metrics Deep Dive

**BLEU (BiLingual Evaluation Understudy)** — Precision-focused

`BLEU = BP × exp(Σ wn × log(pn))`

- **pn** = n-gram precision (how many candidate n-grams appear in reference)
- **BP** = Brevity Penalty (penalizes short translations)
- **wn** = weight for each n-gram size (usually 1/N each)

| Pros | Cons |
| ---- | ---- |
| Simple, fast to compute | Penalizes correct but different wording |
| Widely used benchmark | No semantic understanding |
| Correlates reasonably with human judgment | Exact match only |

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** — Recall-focused

`ROUGE-N Recall = (matching n-grams) / (n-grams in reference)`

- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest Common Subsequence

| Pros | Cons |
| ---- | ---- |
| Captures coverage of reference | No semantic understanding |
| Good for summarization | Exact match only |

**METEOR (Metric for Evaluation of Translation with Explicit ORdering)** — Semantic-aware

- Considers **synonyms** (via WordNet) and **stemming** (run ≈ running)
- Combines precision and recall with weighted harmonic mean
- Penalizes fragmented matches

| Pros | Cons |
| ---- | ---- |
| Semantic understanding | Computationally expensive |
| Better correlation with human judgment | Requires linguistic resources |
| Handles paraphrasing | Language-dependent resources |

> [!TIP]
> 💡 **Aha:** Use **BLEU** for quick benchmarking (translation). Use **ROUGE** for summarization. Use **METEOR** when you need semantic matching but can afford the compute. In production, **human evaluation** is still the gold standard.

### Online Metrics

| Category | Metric | What it measures |
| -------- | ------ | ---------------- |
| **User Engagement** | Acceptance Rate | % of suggestions accepted by users |
| | Usage Rate | % of sessions using the feature |
| **Effectiveness** | Avg Completion Time | Time to complete task (with vs without feature) |
| **Latency** | Response Time | Time for suggestion to appear after keystroke |
| **Quality** | Feedback Rate | Rate of explicit user feedback |
| | Human Evaluation | Expert ratings of output quality |

> [!TIP]
> 💡 **Aha:** **Perplexity** tells you how well the model predicts test data, but doesn't tell you if outputs are useful. **Online metrics** (acceptance rate, completion time) tell you if users actually benefit. Always measure both.

### LLM Evaluation Benchmarks (Task-Specific)

For chatbots, perplexity isn't enough. Evaluate across diverse tasks:

| Category | Benchmarks | What it tests |
| -------- | ---------- | ------------- |
| **Common-Sense Reasoning** | PIQA, HellaSwag, WinoGrande, CommonsenseQA | Everyday logic, cause-effect, idioms |
| **World Knowledge** | TriviaQA, Natural Questions, SQuAD | Factual recall, question answering |
| **Reading Comprehension** | SQuAD, QuAC, BoolQ | Understanding and extracting from text |
| **Mathematical Reasoning** | GSM8K (grade school), MATH (competition) | Problem-solving, multi-step reasoning |
| **Code Generation** | HumanEval (Python), MBPP (multi-language) | Syntax, correctness, functionality |
| **Composite** | MMLU, AGIEval, MMMU (multilingual) | Multi-domain, multi-task assessment |

### Safety Evaluation Benchmarks

Critical for production chatbots:

| Category | Benchmarks | What it tests |
| -------- | ---------- | ------------- |
| **Toxicity** | RealToxicityPrompts, ToxiGen, HateCheck | Harmful content generation, hate speech |
| **Bias & Fairness** | CrowS-Pairs, BBQ, BOLD | Gender, racial, socioeconomic bias |
| **Truthfulness** | TruthfulQA | Factual accuracy, avoiding falsehoods |
| **Privacy** | PrivacyQA | Data leakage, PII exposure |
| **Adversarial Robustness** | AdvGLUE, TextFooler, AdvBench | Resistance to adversarial inputs |

### Online LLM Evaluation

| Metric | What it measures |
| ------ | ---------------- |
| **User Feedback/Ratings** | Direct satisfaction (thumbs up/down) |
| **Engagement** | Queries per session, session duration, return rate |
| **Conversion Rate** | % users who subscribe/pay after interaction |
| **Online Leaderboards** | LMSYS Chatbot Arena (800K+ human comparisons) |

> [!TIP]
> 💡 **Aha:** Task-specific benchmarks tell you **what** the model can do. Safety benchmarks tell you **what it shouldn't do**. Human eval (LMSYS Arena) tells you **what users prefer**. A production chatbot needs all three.

---

## D.10 Chatbot Inference Pipeline Components

Beyond the model itself, production chatbots need:

```
User Prompt → Safety Filter → Prompt Enhancer → Response Generator (LLM + Top-p) → Response Safety Evaluator → Output
                   ↓                                                                        ↓
            Rejection Response ←────────────────────────────────────────────────── Rejection Response
```

| Component | Purpose |
| --------- | ------- |
| **Safety Filter** | Block harmful/inappropriate prompts before model sees them |
| **Prompt Enhancer** | Expand acronyms, fix typos, add context for better responses |
| **Response Generator** | LLM + top-p sampling; may generate multiple and select best |
| **Response Safety Evaluator** | Check generated response for harmful content before showing user |
| **Rejection Response Generator** | Polite explanation when request can't be fulfilled |
| **Session Management** | Track conversation history for multi-turn context |

**Session Management for Multi-Turn:**
- Feed previous turns into context: `[Turn 1] ... [Turn 2] ... [Current Prompt]`
- Track within context window limit (4K, 8K, 128K tokens)
- May summarize old turns if context exceeds limit

---

### Google's Generative AI APIs

Google's generative AI APIs offer pre-trained foundation models that can be fine-tuned for specific tasks:

- **Text Completion**: Generating long-form content or completing snippets
- **Multi-turn Chat**: Maintaining state across several turns of conversation
- **Code Generation**: Specialized models for writing and debugging code
- **Image Generation**: Using the Imagen API to create and customize images

---

## E.1 LLM Serving Architecture at Scale

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

| Option                                | Pros                                                        | Cons                                                             | Best For                                     |
| ------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------- |
| **Managed (Vertex AI / SageMaker)**   | Zero infra management, auto-scaling, built-in monitoring    | Less optimization control, vendor lock-in, higher costs at scale | Startups, rapid prototyping, small ops teams |
| **Self-hosted (vLLM / TensorRT-LLM)** | Full control, better cost efficiency at scale, customizable | Requires ML infra expertise, GPU management complexity           | High volume (millions/day), cost-sensitive   |

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

> [!TIP]
> 💡 **Aha:** With static batching, one long answer blocks the whole batch. Continuous batching **refills** the batch as soon as any request completes, so the GPU rarely idles. The "aha" is: treat the batch as a **queue**, not a fixed group.

**3. KV Cache Management**

**What**: Store the **Key** and **Value** matrices produced by each attention head so they are not recomputed. In standard attention, the score matrix has shape `[batch, heads, sequence_length, sequence_length]`; each new token would require recomputing scores over all previous tokens.

**Why KV cache is needed**: Autoregressive decoding feeds all prior tokens into the next step. Without caching, every generation step recomputes keys and values for the entire prefix, giving O(n²) work per token. Caching lets you compute K and V only for the new token and reuse the rest, reducing to O(n) per token. Reported speedups from KV caching are on the order of ~30–40% in standard implementations.

**How it works**: For each new token, compute and store its K and V; look up cached K/V for all previous positions when computing attention. Only the new token’s key/value are written each step.

**Challenge**: Cache size grows linearly with sequence length (and with layers × heads × head_dim). For a 32-layer model with 768-dim embeddings, each token can use on the order of ~50KB of cache; a 2K-token sequence can need ~100MB of KV cache. Long contexts and many concurrent requests make this the main memory bottleneck.

**Solution — PagedAttention (vLLM)**: Inspired by OS virtual memory and paging. The KV cache is split into **fixed-size blocks** and stored in non-contiguous memory. That reduces fragmentation and allows sharing (e.g. shared prompt prefix across requests). vLLM reports near-zero wasted KV memory and roughly **2–4× throughput** versus non-paged systems on long sequences and large models.

**5. Speculative Decoding**

**Problem**: Token-by-token autoregressive generation is slow because each new token requires a full forward pass of the large model.

**Solution**: A small **draft** model proposes several candidate tokens in a row. The **target** (large) model does a single forward pass over the whole candidate sequence and accepts tokens that match its predictions; the first mismatch stops the run and the rest are discarded. Accepted tokens advance the sequence without extra target-model steps. Typical reported speedups are **2–2.5×**; variants (multiple draft models, tree-based decoding) can reach ~3–4× or more at the cost of extra memory and complexity.

| Technique                | Speedup                  | Trade-off                                    |
| ------------------------ | ------------------------ | -------------------------------------------- |
| **Standard Speculative** | 2–2.5× (often up to ~3×) | Needs a separate draft model                 |
| **Self-Speculative**     | ~2.5×                    | Uses smaller/quantized version of same model |
| **Tree-based**           | Up to ~4–6×              | More memory and logic for tree search        |

**Why it works**: The target model verifies **N** candidates in one forward pass (over a sequence of length N). That cost is similar to generating a single token, so you effectively get several tokens per large-model step when the draft is accurate. **Draft latency** (how fast the draft runs) usually matters more for end-to-end speedup than the draft’s raw language quality.

**4. Caching Strategy**

| Strategy             | Hit Rate                | Latency          | Best For                           |
| -------------------- | ----------------------- | ---------------- | ---------------------------------- |
| **Prompt caching**   | High for system prompts | 2-5x speedup     | Common prefixes, few-shot examples |
| **Response caching** | 10-30%                  | Instant          | Identical requests                 |
| **Semantic caching** | 30-50%                  | +5-10ms overhead | Paraphrased queries                |

---

## E.2 RAG (Retrieval-Augmented Generation) System

**Why this comes next:** E.1 gave you **LLM serving** (how to run the model at scale). When the model **lacks knowledge** about your domain (docs, KB, policies) or that knowledge **changes often**, you add **retrieval** at query time—that's **RAG**. Same request path (gateway → orchestration → LLM), but orchestration now includes "retrieve relevant chunks, then generate."

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

> [!TIP]
> 💡 **Aha:** RAG doesn't cram everything into the model's weights. It keeps the LLM general and **fetches** relevant knowledge at query time. That gives you updatable knowledge, smaller models, and citations—but you must design retrieval and chunking well or the model "makes it up."

### Key Components

**1. Document Ingestion Pipeline**

| Service       | Google Cloud            | AWS                     |
| ------------- | ----------------------- | ----------------------- |
| RAG Engine    | Vertex AI RAG Engine    | Bedrock Knowledge Bases |
| Vector Search | Vertex AI Vector Search | OpenSearch Serverless   |
| Processing    | Dataflow                | Glue/EMR                |

**2. Vector Database Options**

- **Managed**: Vertex AI Vector Search, Amazon OpenSearch
- **Self-hosted**: Pinecone, Weaviate, Qdrant, Milvus

**3. Embedding Models**

- **Google**: text-embedding-004 (Vertex AI)
- **AWS**: Amazon Titan Embeddings (Bedrock)
- **Open Source**: sentence-transformers, **BGE** (BAAI General Embeddings)—embedding models from BAAI (Beijing Academy of Artificial Intelligence), e.g. bge-base, BGE-M3 for multilingual

### Search as RAG: the power of search agents

**Why search matters in system design:** Search is how users navigate digital information—products, docs, internal knowledge. Good search means **relevance** (results that match what they want) and **speed**. Users also expect search to "get" intent: understand what they _mean_, not just the keywords they type. For businesses, poor search means lost customers, wasted time in internal docs, and users leaving for another platform.

**Search = RAG + optional GenAI.** A "search agent" in this sense is: connect to your data (structured in BigQuery, unstructured in GCS, or both) → observe the user's query (environment) → **act** by retrieving or recommending (data stores as tools) → return the right information (or an LLM-generated answer grounded in that data). That loop is exactly **RAG**: retrieve first, then optionally generate. **Grounding**—feeding the LLM with your first-party data, curated third-party data, or even a knowledge graph (e.g. **Grounding with Google Search**)—reduces hallucinations and keeps answers trustworthy.

**Vertex AI Search** is Google's managed offering for this. It provides:

| Capability          | What it does                                                                                                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data connection** | Index structured (BigQuery) and unstructured (GCS) data; same RAG idea: your data as the source of truth.                                                                             |
| **Grounding**       | Ground LLM responses in your data and optionally **Google's knowledge graph** (Grounding with Google Search) for public facts.                                                        |
| **Search variants** | **Document search** (docs, media, healthcare); **Search for commerce** (e-commerce catalog, product discovery, product attributes, complex product queries).                          |
| **Recommendations** | General-purpose recommendation engine (similar content, user behavior); media and retail recommendations.                                                                             |
| **GenAI on top**    | **Search summaries** (concise overview of results, doc summary, product comparison); **Answers and follow-ups** (natural-language Q&A over search results, with follow-up questions). |
| **Enterprise**      | Access controls, analytics (search trends, user behavior), scalable APIs/SDKs for customer-facing search or internal knowledge bases.                                                 |

> [!TIP]
> 💡 **Aha:** When an interviewer says "design search for our site" or "smart search for our catalog," they often mean RAG: connect data → retrieve (and optionally rerank) → optionally add an LLM answer grounded in retrieved results. Vertex AI Search (and AWS equivalents) package this as a managed "search agent"; you can also build it from RAG Engine + Vector Search + an LLM yourself.

### Document Parsing

Before chunking, PDFs and other documents must be **parsed** to extract text, tables, and images into a structured format the LLM can understand.

| Approach | How it works | Pros | Cons | Tools |
| -------- | ------------ | ---- | ---- | ----- |
| **Rule-based** | Predefined rules based on layout patterns | Simple; fast | Brittle; fails on varied layouts | PyMuPDF, pdfplumber |
| **AI-based** | Object detection + OCR to identify regions | Handles complex layouts; robust | Slower; needs more compute | Layout-Parser, Dedoc, Nougat |

**AI-based parsing pipeline:**
1. **Layout detection**: Object detection model identifies regions (paragraphs, tables, figures, headers)
2. **Text extraction**: OCR extracts text from each region with correct reading order
3. **Structured output**: Text blocks (coordinates, text, order) + non-text blocks (figure coordinates)

**Managed services:** Google Cloud Document AI, Amazon Textract, PDF.co

> [!TIP]
> 💡 **Aha:** If your PDFs have **consistent templates** (e.g., invoices, forms), rule-based is faster and cheaper. If layouts **vary widely** (wiki pages, reports, mixed formats), use AI-based parsing—it's worth the extra compute.

### Chunking Strategy Trade-offs

| Strategy                     | Pros                | Cons                    | Best For                |
| ---------------------------- | ------------------- | ----------------------- | ----------------------- |
| **Fixed-size (512 tokens)**  | Simple, predictable | May split concepts      | Uniform documents       |
| **Semantic chunking**        | Preserves coherence | Complex, variable sizes | Complex content         |
| **Hybrid (fixed + overlap)** | Balanced            | More storage            | Most production systems |

**Chunking methods in practice:**

| Method | Description | Tool example |
| ------ | ----------- | ------------ |
| **Length-based** | Split by character/token count | LangChain `CharacterTextSplitter` |
| **Recursive** | Split by separators (paragraphs → sentences → words) with overlap | LangChain `RecursiveCharacterTextSplitter` |
| **Regex-based** | Split on punctuation (., ?, !) for sentence-level chunks | Custom regex splitters |
| **Structure-aware** | Split at element boundaries (headers, list items, code blocks) | `MarkdownHeaderTextSplitter`, `HTMLHeaderTextSplitter` |

**Why chunking matters**: LLMs have context windows. Documents often exceed this, so we must break them into chunks. Smaller chunks improve retrieval precision—a query about "Python loops" matches better to a 500-token chunk about loops than a 5000-token document about Python.

> [!TIP]
> 💡 **Aha:** Chunk size is a **precision vs context** trade-off. Too small → you retrieve the right idea but maybe miss surrounding explanation. Too large → you get more context but dilute relevance. Overlap and semantic boundaries help keep "one concept per chunk."

### Retrieval Strategy Trade-offs

| Strategy           | Latency | Semantic | Keywords | Best For                 |
| ------------------ | ------- | -------- | -------- | ------------------------ |
| **Dense (Vector)** | 10-50ms | ✓        | ✗        | Conceptual queries       |
| **Sparse (BM25)**  | 1-5ms   | ✗        | ✓        | Exact matches            |
| **Hybrid**         | 15-60ms | ✓        | ✓        | Production (recommended) |

**BM25** = keyword-based ranking using term frequency and inverse document frequency; no embeddings, just lexical match.

**Why hybrid works**: Dense retrieval captures meaning ("iterate" ≈ "loop"), sparse captures exact keywords ("Python"). Combining both via **RRF (Reciprocal Rank Fusion)** gives best results.

> [!TIP]
> 💡 **Aha:** **Dense** = "these two _mean_ the same thing" (embedding similarity). **Sparse** = "these two _contain_ the same words" (e.g. BM25). Queries need both: "how do I loop in Python?" benefits from semantic match on "loop" and exact match on "Python." Hybrid + RRF merges the two rank lists without a single embedding doing everything.

### Reranking Trade-offs

**No Reranking**: Lower latency, simpler pipeline, but lower quality.

**Cross-Encoder Reranking**: Much higher accuracy because it processes query-document pairs together (sees interactions), but adds ~10ms per document.

✅ **Best practice:** Retrieve K=20, rerank to top 5. The two-stage approach combines speed (bi-encoder retrieval) with accuracy (cross-encoder reranking).

> [!TIP]
> 💡 **Aha:** **Bi-encoder** = query and doc are embedded _separately_; similarity is dot product. Fast (one pass each) but the model never sees "query + doc together." **Cross-encoder** = one forward pass with "[query] [doc]"; the model sees the _pair_ and scores relevance directly. Slower, but much more accurate. So: retrieve broadly with bi-encoder, then rerank the top K with a cross-encoder.

### Approximate Nearest Neighbor (ANN) Algorithms

At scale (millions of chunks), **exact nearest neighbor search** (O(N×D)) is too slow. ANN algorithms trade a small accuracy loss for sublinear search time (O(log N × D) or better).

| Category | How it works | Pros | Cons | Examples |
| -------- | ------------ | ---- | ---- | -------- |
| **Tree-based** | Partition space by feature values; search relevant regions | Fast for low dimensions | Degrades in high dimensions (>20) | k-d tree, Annoy |
| **LSH** (Locality-Sensitive Hashing) | Hash similar points to same bucket | Simple; fast | Lower recall; many hash tables needed | LSH, MinHash |
| **Clustering-based** | Group vectors into clusters; search by centroid, then within cluster | Balances speed and recall | Cluster quality matters | IVF (FAISS), ScaNN |
| **Graph-based** | Build proximity graph; navigate from coarse to fine levels | Best recall at high scale | More memory; complex index build | HNSW (Hierarchical Navigable Small World) |

**Clustering-based retrieval (two-step):**
1. **Inter-cluster**: Compare query to cluster centroids → select closest clusters
2. **Intra-cluster**: Search only within selected clusters

**HNSW (graph-based):**
- Nodes = data points; edges = proximity links
- Hierarchical layers: start at coarse top layer, descend to fine bottom layer
- Navigate by following edges to closer nodes at each level

**Frameworks:**
- **FAISS** (Meta): IVF, PQ, HNSW; production-ready, GPU support
- **ScaNN** (Google): Optimized quantization + HNSW
- **Annoy** (Spotify): Tree-based; simple, read-only index
- **Elasticsearch**: Vector similarity search with HNSW
- **Managed**: Vertex AI Vector Search, Amazon OpenSearch, Pinecone, Weaviate, Qdrant

> [!TIP]
> 💡 **Aha:** For RAG at scale, **HNSW** (graph-based) is the default choice—best recall-latency trade-off. **IVF** (clustering) is good when you need to control index size. **Tree-based** and **LSH** are faster to build but lower recall for high-dimensional embeddings.

### Query Expansion

**Problem:** User queries are often short, ambiguous, or misspelled. Raw query embedding may not match relevant documents.

**Solution:** Expand the query before embedding to improve retrieval.

| Technique | How it works | When to use |
| --------- | ------------ | ----------- |
| **Query rewriting** | LLM rewrites query for clarity, fixes typos, expands acronyms | Always (cheap preprocessing) |
| **HyDE (Hypothetical Document Embedding)** | LLM generates a hypothetical answer; embed that instead of raw query | Short queries; "what is X" questions |
| **Query2Doc** | LLM generates pseudo-document with relevant terms | Conceptual queries; improve keyword coverage |
| **Multi-query** | Generate N query variants; retrieve for each; merge results | High-stakes retrieval; cover more angles |

**Query expansion pipeline:**
```
User Query → LLM (rewrite/expand) → Expanded Query → Embedding → Vector Search
```

> [!TIP]
> 💡 **Aha:** **HyDE** is counterintuitive: instead of embedding the question "What is RAG?", you embed an LLM-generated answer "RAG is a technique that combines retrieval with generation..." The answer's embedding is often closer to relevant documents than the question's embedding.

### Advanced RAG Techniques

These techniques improve retrieval when plain “embed query → top‑k chunks” is not enough: when answers span multiple hops, when queries vary in difficulty, or when user wording doesn’t match document wording.

---

**1. Graph RAG**

**What it is:** You build a **knowledge graph** from your corpus (entities as nodes, relations as edges) and combine it with vector search. Retrieval can follow _links_ (e.g. “this person → worked at → this company”) as well as semantic similarity.

**How it helps:** Many questions need **multi-hop** reasoning: “Who was the CEO of the company that acquired X?” requires (X → acquired by → company → CEO → person). Flat vector search often returns only one hop. Graph RAG retrieves **subgraphs** (e.g. k-hop neighborhoods) so the LLM sees not just similar text but explicit _who–what–where_ structure.

**When to use:** Strong fit for domains rich in **entities and relations** (people, orgs, products, events) and questions that chain them. Overkill for unstructured long-form text with few named relations.

> [!TIP]
> 💡 **Aha:** Vector search answers “what text is similar?” Graph RAG adds “how are these things _connected_?” so the model can follow paths, not only similarity.

---

**2. Adaptive Retrieval**

**What it is:** Instead of always retrieving the same number of documents (e.g. k=10), you **change k per query**. Simple factoid questions get fewer docs; broad or multi-fact questions get more.

**How it helps:** With a **fixed k**, easy questions get unnecessary context (wasted tokens, more noise) and hard questions may get too few (missing evidence). Adaptive retrieval uses a small classifier, heuristics (e.g. query length, question type), or the **shape of similarity scores** (e.g. “biggest drop” between consecutive docs) to choose k. Some methods need no extra model—e.g. set k at the largest score gap in the ranked list.

**When to use:** When your traffic mixes **simple lookups** and **complex / multi-document** questions. Saves tokens and latency on easy queries and improves recall on hard ones.

> [!TIP]
> 💡 **Aha:** One size doesn’t fit all: “What is the capital of France?” needs 1–2 chunks; “Compare the economic policies of France and Germany in the 1980s” needs many. Adaptive k tunes retrieval to each question.

---

**3. Query Decomposition**

**What it is:** Before retrieval, an LLM **splits** the user question into 2–5 **sub-questions** that are answered by different parts of the corpus. You run retrieval once per sub-question, then merge and deduplicate the chunks and pass that combined context to the final answer model.

**How it helps:** Questions like “How does X differ from Y?” or “Which of A, B, C had the highest Z?” don’t match one passage—they need **several**. One query embedding often misses some of them. Decomposing into “What is X?”, “What is Y?”, “How do they differ?” (or “What is Z for A?”, “What is Z for B?”, …) yields focused sub-queries and better coverage.

**When to use:** **Multi-part** or **comparison** questions, and whenever a single embedding tends to retrieve only one “side” of the answer. Adds latency (one LLM call to decompose, then multiple retrievals) but can significantly improve accuracy.

> [!TIP]
> 💡 **Aha:** One query → one vector → one retrieval set often undersamples. Decomposing “How does A differ from B?” into “What is A?” and “What is B?” (and optionally “How do they differ?”) pulls in the right evidence for each piece, then the model synthesizes.

---

**4. HyDE (Hypothetical Document Embeddings)**

**What it is:** You **don’t** embed the user query directly. Instead, you ask an LLM: “Write a short passage that would answer this question.” You get 1–5 such **hypothetical** passages, embed _those_, and (often) **average** their vectors. That single vector is used to search the real document index.

**How it helps:** Query and documents often use **different words** for the same idea (e.g. user: “loop,” docs: “iteration construct”). The query embedding can sit in a different region of the embedding space than the best-matching docs. Hypothetical answers “translate” the question into **passage-like** text, so their embeddings sit closer to real relevant passages. Averaging smooths noise from any one generation.

**When to use:** When **vocabulary mismatch** hurts recall (e.g. lay users vs technical docs, or one language vs translated corpus) and when you can afford one extra LLM call before retrieval. Less useful when queries already look like document sentences.

> [!TIP]
> 💡 **Aha:** You’re searching with “what an answer would look like” instead of “what the question looks like.” The hypothetical doc is in the same “language” as your corpus, so similarity search works better.

---

**Quick reference**

| Technique               | Main idea                                                                               | Best for                                         |
| ----------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **Graph RAG**           | Vector search + graph structure (entities, relations); retrieve subgraphs for multi-hop | Entity-heavy domains, “who/what/where” chains    |
| **Adaptive Retrieval**  | Vary number of retrieved docs (k) by query complexity                                   | Mix of simple and complex questions              |
| **Query Decomposition** | Split question into sub-questions; retrieve per sub-question; merge context             | Multi-part, comparison, “A vs B” style questions |
| **HyDE**                | Generate hypothetical answer(s), embed those, search with that vector                   | Vocabulary mismatch between user and corpus      |

---

### RAFT: Retrieval-Augmented Fine-Tuning

**Problem:** In RAG, retrieval isn't perfect—irrelevant documents (distractors) get included. Standard LLMs may be misled by these distractors and generate incorrect responses.

**Solution:** **RAFT** (Retrieval-Augmented Fine-Tuning) trains the LLM to distinguish relevant ("golden") documents from distractors.

**How it works:**
1. **Document labeling**: Retrieved documents are labeled as relevant (golden) or irrelevant (distractors)
2. **Joint training**: Finetune LLM on (query, mixed context, answer) where context includes both golden docs and distractors
3. **Result**: Model learns to prioritize relevant content and ignore noise

**Training data format:**
```
Query: "What year was the company founded?"
Context: [Golden doc: "Acme Corp was founded in 1995..."] + [Distractor 1] + [Distractor 2]
Answer: "The company was founded in 1995."
```

| Approach | Training data | LLM sees distractors | Performance |
| -------- | ------------- | -------------------- | ----------- |
| **Standard RAG** | None (use pretrained) | At inference only | Baseline |
| **Golden-only FT** | Only relevant docs | No | Better on clean retrieval |
| **RAFT** | Mix of golden + distractors | Yes (during training) | Best on noisy retrieval |

**When to use RAFT:**
- Retrieval quality is imperfect (often true in production)
- Domain has many similar-looking documents that confuse the LLM
- You can afford to finetune (need training data with relevance labels)

> [!TIP]
> 💡 **Aha:** RAFT is like training a student with "open-book exams" where some pages are irrelevant. The student learns to **find and use** the right pages while ignoring distractions. Standard finetuning is like "closed-book"—the student memorizes everything. RAFT produces LLMs that are robust to real-world noisy retrieval.

---

### RAG Evaluation: The Triad

RAG evaluation has three dimensions—retrieval quality, generation faithfulness, and answer quality:

```
                    Query
                   /     \
                  /       \
    Context Relevance    Answer Relevance
          |                    |
          v                    v
      Retrieved ─────────── Generated
       Context   Faithfulness   Response
```

| Metric | What it measures | How to evaluate |
| ------ | ---------------- | --------------- |
| **Context Relevance** | Did retrieval fetch the right documents? | Hit rate, MRR, NDCG, Precision@k |
| **Faithfulness** | Is the response grounded in retrieved context? | LLM-as-judge ("is this claim in the context?") |
| **Answer Relevance** | Does the response address the question? | LLM-as-judge ("does this answer the question?") |
| **Answer Correctness** | Is the response factually correct? | BLEU/ROUGE vs reference; or human eval |

**Faithfulness detection methods:**

| Method | How it works | Accuracy | Latency |
| ------ | ------------ | -------- | ------- |
| **Self-consistency** | Sample N answers, check agreement | Moderate | High (N× calls) |
| **NLI (entailment)** | Check if context entails each claim | High | +50–100ms |
| **LLM-as-Judge** | "Is this claim supported by context?" | High | +100–200ms |
| **Specialized models** | Fine-tuned faithfulness classifier | Highest | ~+50ms |

**Tools:** RAGAS (faithfulness, answer relevancy, context precision/recall), TruLens (RAG triad), LangSmith (groundedness), Phoenix (hallucination evals), Vectara FaithJudge (specialized model).

> [!TIP]
> 💡 **Aha:** A RAG system can fail in three ways: (1) **retrieval failure** (didn't fetch relevant docs), (2) **grounding failure** (LLM made things up), (3) **relevance failure** (answered a different question). Evaluate all three—RAGAS metrics cover them in one framework.

---

## E.3 RAG vs Fine-Tuning Decision Framework

**Why this comes next:** E.2 gave you **RAG** (retrieve, then generate). When do you **also**—or **instead**—**fine-tune**? This section is the decision framework so you choose the right lever for the problem.

**Key insight:** This is not a binary choice. Think of it as a **spectrum of adaptation**: RAG and fine-tuning solve different problems and are often used **together**. The right question is not "RAG or fine-tuning?" but "What does the model lack—**knowledge** or **behavior**?"

- **"The model doesn't _know_ X"** → Add knowledge via RAG (or long context, or caching).
- **"The model doesn't _behave_ like Y"** → Change behavior via fine-tuning (tone, format, schema, jargon).
- **"We need both fresh facts and consistent style"** → Use both: RAG for what to say, fine-tuning for how to say it.

---

### When to Use RAG

**What RAG fixes:** Gaps in **knowledge** and **freshness**. The model is good at reasoning and language but hasn't seen your data (policies, tickets, docs, logs). RAG injects that at query time: you retrieve relevant chunks and put them in the prompt, so the model "reads" your corpus on demand.

**Use RAG when:** The model **lacks knowledge** about your domain (e.g. internal docs, product specs, support history). Your **data changes often** (e.g. daily reports, new releases, tickets)—RAG lets you update the index without retraining. You want to **reduce hallucinations** by **grounding** answers in retrieved text and to **cite sources** (chunk or doc IDs).

**RAG does _not_ fix:** Tone, format, or jargon. If the base model is too informal or ignores your schema, RAG alone won't change that—you need behavior change (prompts or fine-tuning).

---

### When to Use Fine-Tuning

**What fine-tuning fixes:** **Behavior** and **style**. The model "knows" enough from pretraining, but its outputs don't match how you want it to answer: tone (formal vs casual), structure (e.g. JSON with fixed keys), or vocabulary (your domain terms). Fine-tuning adjusts the model's weights so it reliably produces that style.

**Use fine-tuning when:** You need a **specific tone or voice** (e.g. brand guidelines, compliance-friendly wording). You need **strict output format** (e.g. JSON, bullet lists, section headings)—fine-tuning helps the model adhere to schemas. The model **misuses or avoids domain jargon**; training on in-domain examples teaches it to use your terms correctly.

**Fine-tuning does _not_ fix:** Missing or outdated facts. Weights are fixed until the next train run. For fast-changing knowledge, use RAG (or both).

---

### When to Use Both

**Use RAG + fine-tuning when** you need **accurate, up-to-date content** _and_ **consistent presentation**: RAG supplies the **facts** (from docs, KB, logs); fine-tuning shapes **how** those facts are expressed (tone, format, terminology). Example: A support bot that answers from your knowledge base (RAG) but must always respond in a compliant, on-brand style (fine-tuned). Or a report generator that pulls from live data (RAG) and always outputs the same JSON schema (fine-tuned).

---

### Scenario Cheat Sheet

| Scenario                                              | RAG | Fine-Tuning | Both |
| ----------------------------------------------------- | :-: | :---------: | :--: |
| Model lacks knowledge about your domain               | ✅  |     ❌      |      |
| Data changes frequently (docs, tickets, metrics)      | ✅  |     ❌      |      |
| Need specific tone, style, or brand voice             | ❌  |     ✅      |      |
| Domain-specific jargon or terminology                 | ❌  |     ✅      |      |
| Reduce hallucinations by grounding in retrieved text  | ✅  |             |      |
| Change output format or schema (e.g. JSON, sections)  | ❌  |     ✅      |      |
| High accuracy _and_ fresh data _and_ consistent style |     |             |  ✅  |

### Cost Comparison

Cost structure is different, not just "cheaper vs more expensive":

| Approach                    | Cost model          | What you pay for                                                      | Example ballpark                                                                                  |
| --------------------------- | ------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **RAG**                     | **Per query**       | Retrieval (embeddings, vector search) + LLM tokens (context + answer) | ≈$0.01-0.05 per query; 1M queries/month ≈ $10-50K                                                 |
| **Fine-tuning (e.g. LoRA)** | **One-time**        | Training compute + data prep; then inference cost as usual            | ≈$500-2,000 for **LoRA** (Low-Rank Adaptation) on 7-70B model; amortizes over all future requests |
| **Full fine-tune**          | **One-time, large** | Full training run on your data                                        | $10K-100K+ depending on model size and data                                                       |

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

> [!TIP]
> 💡 **Aha:** RAG = **external memory** you can change without retraining (add docs, edit, delete). Fine-tuning = **internalized behavior** (tone, format, jargon) that’s fixed until the next train run. Use RAG when the world changes; use fine-tuning when you want the model itself to change how it answers.

---

## E.4 Agentic AI Systems

**Why this comes next:** E.2–E.3 gave you **RAG** and **fine-tuning** (retrieval + behavior). When do you need **tools** and **multi-step** reasoning—e.g. look up an order, call an API, then decide what to say? That's **agents**: the same request path (gateway → orchestration → LLM) but with a loop and tools.

### What Is an Agent? Why Do We Need One?

📖 **Definition:** An **agent** is an LLM that **repeatedly** decides, acts, and observes until a task is done. It has access to **tools** (APIs, databases, search, code) and runs in a **loop**: perceive the current state → decide the next step → call a tool → observe the result → repeat. That loop is what makes it an agent, not "one prompt → one answer."

**Why we need agents:** A single LLM call is stateless and one-shot. It can't look up live data, call your CRM, or run multi-step workflows. **RAG** adds retrieval at query time but still produces one answer from one retrieved context—no tool calls, no iterative refinement. **Agents** add the ability to _use the world_: query systems, run code, search, then decide what to do next from the results. So you need an agent when the task requires **multiple steps**, **live data** (orders, DB, APIs), or **decisions that depend on tool outputs** (e.g. "if order status is X, do Y").

**When to use agents vs. not:**

| Use an agent when…                                                                              | Use a single call or RAG when…                                                |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| The task needs **multiple tool calls** or steps (e.g. check order → update CRM → create ticket) | The task is **one question → one answer** (e.g. "what is our return policy?") |
| The **next step depends on live results** (e.g. "if refund approved, then…")                    | The pipeline is **fixed** (e.g. embed query → retrieve → generate)            |
| You need **orchestration across systems** (APIs, DBs, search)                                   | You only need **retrieval + generation** (RAG) or pure generation             |
| Decisions are **context-sensitive** and hard to encode as rules                                 | The flow is **deterministic** and easy to script                              |

> [!TIP]
> 💡 **Aha:** Start with the simplest thing that works (single call, or RAG). Add an agent only when you need **loop + tools**—when the model must _use_ external systems and _iterate_ based on what it sees.

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

**Why an agent fits here:** Support often needs _multi-step_ actions (look up order → check policy → create ticket or escalate) and _live data_ (order status, account history). One LLM call or RAG-only can't do that; you need a loop + tools.

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

> [!TIP]
> 💡 **Aha:** An agent is an LLM in a **loop** with tools. The model doesn’t just answer once; it _reasons → acts (calls a tool) → observes (gets result) → reasons again_ until it can respond. That turns the LLM into a controller over APIs, DBs, and search—so the "aha" is: the value is in the **loop + tools**, not in a bigger model.

### Customer engagement & contact center (Google Customer Engagement Suite)

**Why engagement beyond search:** Customers don’t always want to search—they want to **connect directly** for answers and support. Those touchpoints are pain points when they fail but also critical: positive engagement can make or break a company. Google’s **Customer Engagement Suite** is built for this: conversational AI (including generative AI) for chatbots, live-agent support, and analytics, often on top of a **Contact Center as a Service (CCaaS)** platform.

**Conversational agents (chatbots):** Two main approaches—**deterministic** and **generative**. **Deterministic** = rule-based, explicit logic (e.g. “if user presses 1, route to billing”); everything must be defined; low–medium code, can feel rigid. **Generative** = LLM-driven, flexible, but can lack structure. A **hybrid** approach is common in production: rules and flows for common paths, GenAI for open-ended questions and natural language. In Google Cloud you can build simple agents with natural-language instructions (GenAI) or complex hybrid agents with custom rules and logic.

**Agent Assist:** When bots aren’t enough or a human touch is needed, **live agents** take over—but they need support too. **Agent Assist** gives live agents **in-the-moment assistance**: AI-generated **suggested responses**, **knowledge-base recommendations** to solve the customer’s issue, **real-time transcription and translation**, **conversation summarization**, and coaching. That’s GenAI in the loop with the human: the system suggests; the agent decides. In system design terms, “escalate to human” is a tool; Agent Assist is the layer that makes that handoff effective.

**Conversational Insights:** All customer interactions (chatbot and human) are data. **Conversational Insights** analyzes that data to give contact center leaders **data-driven insights**: agent and caller **sentiment**, **entity identification**, **call topics**, and automatic flagging of interactions that need review. **Generative FAQ** in Insights surfaces the **common questions** customers ask and how they’re answered—so you can find **FAQ gaps**, **trending questions**, and improve responses. Useful for evaluation (E.5) and for improving your RAG/knowledge base.

**CCaaS (Contact Center as a Service):** A full contact center needs 24/7 multichannel (phone, text, email), security and privacy, CRM integration, and **omnichannel** experience (consistent across web, app, phone, text). CCaaS provides the infrastructure: **simultaneous multichannel** communication, **channel switching**, **multimodal** interactions (text, voice, images), and **agent routing**. It integrates with **Conversational Agents** (automated support), **Agent Assist** (live-agent guidance), and **Conversational Insights** (analytics). When an interviewer asks “design a contact center” or “support voice and chat,” CCaaS + agents + Agent Assist + Insights is the product landscape to reference.

> [!TIP]
> 💡 **Aha:** “Customer support” in system design often means: **conversational agent** (deterministic + generative hybrid) for self-service, **escalate-to-human** as a tool, and **Agent Assist** + **Insights** for when humans are in the loop. Full contact center = **CCaaS** plus these pieces.

### Enterprise knowledge workers (Gemini Enterprise)

**Why internal knowledge workers matter:** Transforming the organization often happens by supporting **internal** employees, not only external customers. Employees search across many internal sources—analytics, productivity, content, CRM, communications, portfolio, supply chain, enterprise planning. Making that information **discoverable and actionable** is a core use case for GenAI agents.

**Gemini Enterprise** is designed for this: it helps teams use company information more effectively by creating **AI agents** that **access and understand data from various sources**, regardless of where data is stored. These agents can be integrated into **internal websites or dashboards**—like personal research assistants for work. In system design terms: **unified search** across connected business systems + **agents** that plan, retrieve, and synthesize.

**Pattern: plan-then-verify-then-execute (deep research).** For complex, well-sourced outputs (e.g. advisor report on a trending political topic impacting markets):

1. **Limit data sources** to trusted, curated repositories (e.g. government reports, internal research).
2. **Agent generates a research plan**; the **human verifies** the plan before execution.
3. **Agent executes** the plan: searches thousands of sources, asks new questions, iterates until satisfied.
4. **Output:** detailed report with **source links** and optional **audio summary** for quick consumption.

This is the same "research → draft → grounding" pipeline (F.1 Example 3) but with **human-in-the-loop at the plan stage** and **curated sources only**. Useful when the domain is sensitive (e.g. financial, legal) and you need auditability and control.

**Gemini Enterprise vs NotebookLM Enterprise:** **NotebookLM Enterprise** is a **document-focused** tool: upload specific documents and web sources, then ask questions, summarize, and create content _from those sources only_. **Gemini Enterprise** is a **comprehensive enterprise AI assistant**: it uses **agents** and **unified search** to automate tasks and find information **across all connected business systems**, not just uploaded documents. Gemini Enterprise can **connect to** NotebookLM Enterprise (e.g. attach "Client Notes" for personalized advice); the two serve different roles—deep dive into a corpus vs. search and automate across the enterprise.

**Use case snapshot (advisor):** Retrieve and compare latest investment reports → attach **NotebookLM** client notes for tailored advice → agent evaluates research against client notes (e.g. finds portfolio lacks diversification) → upload spreadsheet, run through company risk calculator → Gemini drafts final client email. Combines **unified search**, **agent reasoning**, **tool use** (risk calculator), and **personalized context** (NotebookLM).

> [!TIP]
> 💡 **Aha:** For "design support for internal knowledge workers," think **Gemini Enterprise**-style: agents + unified search across connected systems, **plan-verify-execute** for high-stakes research, **trusted sources only**, output = report + sources + optional audio. For "deep dive into this set of documents," think **NotebookLM Enterprise**.

### Agent Frameworks

Choose **no-code** (Vertex AI Agent Builder, Bedrock Agents) when you want to configure agents in a UI with minimal code. Choose **programmatic** (ADK, LangChain, LlamaIndex) when you need custom logic, complex workflows, or fine-grained control.

| Platform     | Google Cloud                | AWS            | Open Source                    |
| ------------ | --------------------------- | -------------- | ------------------------------ |
| No-code      | Vertex AI Agent Builder     | Bedrock Agents | -                              |
| Programmatic | Agent Development Kit (ADK) | AgentCore      | LangChain, LlamaIndex, AutoGen |

### Playbooks and system instructions

**Playbook (Conversational Agents):** When you build a generative AI agent with **Conversational Agents**, you define a **playbook** for how the agent should behave. In the playbook you set the agent’s **goal** (e.g. customer support, answering questions, generating content), **detailed instructions** on how to act, and **rules** to follow. You can also **link to external tools** (e.g. data stores for RAG). Once the playbook is defined, you test and interact with the agent. In system design terms, the playbook is the **system-level configuration** that shapes every turn.

**System instructions (general):** The same idea appears elsewhere as **system instructions**: context, **persona**, and **constraints** provided _before_ any user input, so the model’s behavior and responses align with your intent. They help with:

| Concern         | Role of system instructions                                                                              |
| --------------- | -------------------------------------------------------------------------------------------------------- |
| **Consistency** | Keep tone and persona stable across turns                                                                |
| **Accuracy**    | Ground the model in specific knowledge; reduce hallucinations                                            |
| **Relevance**   | Keep responses in the intended domain (e.g. product support only)                                        |
| **Safety**      | Avoid inappropriate or unhelpful content; set boundaries (e.g. “don’t guess; admit when you don’t know”) |

**Metaprompting:** A useful technique is **metaprompting**—using the LLM to **generate, modify, or interpret other prompts**. For example: one prompt says “You are an expert at building virtual agent assistants; for the given company and role, produce a system prompt a developer can use.” You run that once, get a **system prompt** (goal + instructions + rules), then use that as the system instructions for your actual agent. Metaprompting makes prompt creation more **dynamic and adaptable** and is common when defining playbooks or system instructions from high-level specs (company, use case, scope, constraints).

**Production note:** Prototyping in **Google AI Studio** (or similar) with system instructions is a good way to explore behavior. For **enterprise** agents you typically need more: **Conversational Agents** (or equivalent) for adversarial defense, tool wiring, guardrails, and observability.

> [!TIP]
> 💡 **Aha:** The playbook (or system instructions) is the **contract** for your agent: goal + rules + optional tools. Define it first; metaprompting can help you generate it from a short brief (company, role, scope, constraints).

### Tool Types

**Tools** are how the agent interacts with the world: APIs, DBs, search, code. The agent chooses _which_ tool to call and _with what arguments_; the tool runs and returns a result, which the agent uses for the next step.

| Tool Type             | Execution   | Description                                           | Best For                       |
| --------------------- | ----------- | ----------------------------------------------------- | ------------------------------ |
| **Extensions (APIs)** | Agent-side  | Standardized bridges to external APIs                 | Multi-service access           |
| **Function Calling**  | Client-side | Model outputs function name + args; your app executes | Security, audit, human-in-loop |
| **Data Stores**       | Agent-side  | Connect to vector DBs, knowledge bases                | RAG, real-time info            |
| **Plugins**           | Agent-side  | Pre-built integrations (calendar, CRM)                | Rapid capability addition      |

> [!TIP]
> 💡 **Aha:** **Function calling** (client-side) gives you control: the model outputs a tool name + args, and _your app_ decides whether to run it. Use it when you need security, audit, or human-in-the-loop. **Agent-side** tools run automatically when the model requests them—faster but less control.

---

### Agent Protocols: MCP and A2A

**MCP (Model Context Protocol)** and **A2A (Agent-to-Agent / Agent2Agent)** are open standards that define how agents get **tools and context** (MCP) and how **agents talk to other agents** (A2A). Both matter when you build multi-tool or multi-agent systems.

**MCP (Model Context Protocol)**

**MCP** is an open protocol (Anthropic, 2024) that standardizes how applications provide **tools and context** to LLMs. It acts as a universal connector: an LLM or agent connects to **MCP servers**, which expose tools, prompts, and resources (files, DBs, APIs) in a consistent way. So instead of each vendor defining its own tool format, you run or connect to MCP servers and the model gets a uniform interface.

| Aspect        | Description                                                                           |
| ------------- | ------------------------------------------------------------------------------------- |
| **Purpose**   | Standardize how models get tools, prompts, and resources from external systems        |
| **Adoption**  | Anthropic (Claude), OpenAI (Agents SDK), Microsoft (Agent Framework)                  |
| **Use cases** | AI-powered IDEs, custom workflows, connecting agents to Slack, Figma, databases, etc. |

**When it matters:** Use MCP when you want **portable tooling**—the same MCP server can back multiple agents or products. It also helps when you integrate many external systems (CRMs, docs, search) without writing custom glue per vendor.

**A2A (Agent-to-Agent / Agent2Agent Protocol)**

**A2A** is an open standard (Google, 2025) for **communication and collaboration between AI agents** built by different vendors and frameworks. It addresses interoperability: agents from different stacks (e.g. Vertex AI, LangChain, Salesforce) can discover each other, negotiate UX, and exchange tasks and state **without** sharing internal memory, resources, or tools.

| Aspect                  | Description                                                                                                          |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Purpose**             | Enable agent-to-agent collaboration across vendors and frameworks                                                    |
| **Mechanisms**          | **Agent Cards** (JSON metadata: identity, capabilities), capability discovery, task/state management, UX negotiation |
| **Transport**           | JSON-RPC 2.0 over HTTP(S)                                                                                            |
| **Relationship to MCP** | A2A handles **agent ↔ agent**; MCP handles **model ↔ tools/context**. They complement each other.                    |

**When it matters:** Use A2A when you run **multi-agent** or **cross-vendor** workflows (e.g. your agent hands off to a partner’s agent, or you compose agents from different platforms). It gives you a shared protocol for discovery, tasks, and security instead of one-off integrations.

> [!TIP]
> 💡 **Aha:** **MCP** = “how does _this_ agent get its tools and context?” **A2A** = “how do _multiple_ agents from different systems work together?” For a single agent with your own tools, MCP is the standard to consider. For agent-to-agent orchestration across products or vendors, A2A is the standard to consider.

---

### Reasoning Frameworks

**Chain-of-Thought (CoT):** The model generates **intermediate reasoning steps** ("think step-by-step") before the final answer. No tool use—just internal logic. Use when you need interpretability or multi-step reasoning without external data.

**ReAct (Reason + Act):** Combines **reasoning** with **tool use** in a loop. Each turn is either a _Thought_ (what to do next), an _Action_ (tool name + args), or an _Observation_ (tool result). The model keeps going until it can give a final answer.

| Phase              | What Happens                                                                 |
| ------------------ | ---------------------------------------------------------------------------- |
| **1. Reasoning**   | Agent analyzes task, selects tools                                           |
| **2. Acting**      | Agent executes selected tool                                                 |
| **3. Observation** | Agent receives tool output                                                   |
| **4. Repeat**      | Agent reasons from the observation, then next Thought/Action or final answer |

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

> [!TIP]
> 💡 **Aha:** ReAct makes the reasoning **visible** (Thought) and **grounded** (Action → Observation). The model can’t wander off; each step is either "I think…" or "I do X" followed by real tool output. That reduces hallucination in tool use because the next thought is conditioned on actual observations.

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
- _Best for_: Simple use cases, single domain (e.g. support bot with KB + CRM + ticketing)

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
- _Best for_: Domains where agents **collaborate as peers** (e.g. research agent + writing agent + fact-check agent that hand off or run in parallel; no one agent "owns" the plan)

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
- _Best for_: Workflows with a **fixed or predictable sequence** (e.g. research → draft → review → publish) where one "conductor" should own the plan

---

**Multi-Agent vs Hierarchical: Clear distinction**

| Aspect                          | Multi-Agent                                                                               | Hierarchical                                                                      |
| ------------------------------- | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Who decides the plan?**       | Distributed: agents hand off, or a router chooses; no single owner                        | **One supervisor** owns the plan and assigns steps                                |
| **Who do specialists talk to?** | Each other (handoffs) or an aggregator; flow is peer-to-peer or fan-out                   | **Only the supervisor**; specialists do not talk to each other                    |
| **Control shape**               | **Flat** or **peer-to-peer**: many agents, shared or emergent coordination                | **Tree**: one node (supervisor) at the top, specialists as children               |
| **Flow**                        | Emergent (handoffs, parallel, negotiate)                                                  | **Top-down**: Supervisor → assign step → Specialist → result → Supervisor         |
| **When to use**                 | You want **peers** that hand off or run in parallel and someone (or the group) aggregates | You want **one conductor** that plans and delegates in sequence or in a clear DAG |

> [!TIP]
> 💡 **Aha:** **Multi-agent** = "several agents, no single boss; they hand off or run in parallel." **Hierarchical** = "one boss (supervisor) that assigns tasks to specialists and gets results back; specialists don’t talk to each other." Use multi-agent when control should be shared or emergent; use hierarchical when one agent should own the plan and delegate.

---

**4. Additional Patterns**

Beyond single-, multi-, and hierarchical agents, three common _orchestration shapes_ show up in production: stages in a fixed order, independent experts run in parallel, and adversarial roles that argue before a judge. Use these when the task has a natural flow (sequence), benefits from multiple viewpoints (fan-out), or must be stress-tested (debate).

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

| Pattern                 | Architecture                  | Use Case                                                        |
| ----------------------- | ----------------------------- | --------------------------------------------------------------- |
| **Sequential Pipeline** | A → B → C (fixed order)       | Content creation (outline → draft → edit), ETL-style flows      |
| **Parallel Fan-out**    | Query → [A, B, C] → Aggregate | Research, multi-perspective analysis, ensembles                 |
| **Debate/Adversarial**  | Pro vs Con → Judge            | High-stakes decisions, red teaming, counterargument stress-test |

> [!TIP]
> 💡 **Aha:** Single agent = one brain, many tools. Multi-agent = many brains, each with its own tools; you need handoffs. Hierarchical = one brain that delegates; specialists don't talk to each other directly.

### Context Engineering

**The Problem**: As agents run longer, context (chat history, tool outputs, documents) **explodes**. Simply using larger context windows is not a scaling strategy.

> [!TIP]
> 💡 **Aha:** More context isn’t always better. Models often **underuse** the middle of long prompts ("lost in the middle"). So putting the most important instructions or retrieval at the **start and end** of the context, and keeping working context small and focused, improves both quality and cost. Tiered context (working / session / memory / artifacts) is how you scale _usage_ of context without scaling _size_ of every call.

**The Three-Way Pressure on Context:**

| Pressure                   | Problem                                                         |
| -------------------------- | --------------------------------------------------------------- |
| **Cost & latency spirals** | Cost and time-to-first-token grow with context size             |
| **Signal degradation**     | Irrelevant logs distract the model ("lost in the middle")       |
| **Physical limits**        | RAG results and traces eventually overflow even largest windows |

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

| Layer               | Purpose                         | Lifecycle                     |
| ------------------- | ------------------------------- | ----------------------------- |
| **Working Context** | Immediate prompt for this call  | Ephemeral                     |
| **Session**         | Durable log of events           | Per-conversation              |
| **Memory**          | Long-lived searchable knowledge | Cross-session                 |
| **Artifacts**       | Large files                     | Addressed by name, not pasted |

**Multi-Agent Context Scoping:** When one agent delegates to another, control what the sub-agent sees. **Agents as Tools** = sub-agent gets only the instructions and inputs you pass. **Agent Transfer** = sub-agent gets a configurable view over Session (e.g. last N turns).

| Pattern             | Description                                          |
| ------------------- | ---------------------------------------------------- |
| **Agents as Tools** | Sub-agent sees only specific instructions and inputs |
| **Agent Transfer**  | Sub-agent inherits a configurable view over Session  |

---

### Google ADK (Agent Development Kit)

**ADK** is Google's open-source framework for building, deploying, and orchestrating AI agents. It's model-agnostic (optimized for Gemini but works with others), deployment-agnostic (local, Cloud Run, Vertex AI Agent Engine), and compatible with other frameworks.

**Installation:**
```bash
pip install google-adk        # Python
npm install @google/adk       # TypeScript
go get google.golang.org/adk  # Go
```

**Core concepts:**

| Concept | Description |
| ------- | ----------- |
| **LlmAgent** | Agent powered by an LLM; has instructions, tools, and sub-agents |
| **Workflow Agents** | Orchestrate sub-agents: `SequentialAgent`, `ParallelAgent`, `LoopAgent` |
| **Tools** | Functions the agent can call (custom, pre-built, or MCP) |
| **Session State** | Shared state across agents in the same invocation |
| **Agent Transfer** | LLM-driven delegation to sub-agents via `transfer_to_agent()` |
| **AgentTool** | Wrap an agent as a tool for another agent to call |

**Workflow agents:**

| Agent | Behavior |
| ----- | -------- |
| `SequentialAgent` | Run sub-agents in order; each sees shared state from previous |
| `ParallelAgent` | Run sub-agents concurrently; all share the same state |
| `LoopAgent` | Repeat sub-agents until `max_iterations` or `escalate=True` |

**Multi-agent patterns in ADK:**

| Pattern | How to build |
| ------- | ------------ |
| **Coordinator/Dispatcher** | Parent `LlmAgent` with sub-agents; LLM decides which to call via `transfer_to_agent` |
| **Sequential Pipeline** | `SequentialAgent` with sub-agents; use `output_key` to pass data via state |
| **Parallel Fan-Out/Gather** | `ParallelAgent` + `SequentialAgent` for aggregation |
| **Hierarchical Decomposition** | Nest agents; parent uses `AgentTool` to call child as tool |
| **Generator-Critic** | `SequentialAgent` with generator → reviewer; reviewer reads generator's `output_key` |
| **Iterative Refinement** | `LoopAgent` with refiner → checker; loop until checker escalates |

**Practical example: Customer support agent with ADK**

```python
from google.adk.agents import LlmAgent, SequentialAgent

# Define tools as functions
def get_order_status(order_id: str) -> dict:
    """Look up order status from database."""
    return {"order_id": order_id, "status": "shipped", "eta": "2026-02-01"}

def create_support_ticket(issue: str, priority: str) -> dict:
    """Create a support ticket in the ticketing system."""
    return {"ticket_id": "TKT-12345", "status": "created"}

def search_knowledge_base(query: str) -> dict:
    """Search the knowledge base for relevant articles."""
    return {"articles": [{"title": "Return Policy", "content": "..."}]}

# Create specialist agents
order_agent = LlmAgent(
    name="OrderAgent",
    model="gemini-2.0-flash",
    description="Handles order status inquiries and shipping questions.",
    instruction="You help customers with order status. Use get_order_status tool.",
    tools=[get_order_status]
)

knowledge_agent = LlmAgent(
    name="KnowledgeAgent",
    model="gemini-2.0-flash",
    description="Answers policy and FAQ questions.",
    instruction="Search the knowledge base to answer customer questions.",
    tools=[search_knowledge_base]
)

escalation_agent = LlmAgent(
    name="EscalationAgent",
    model="gemini-2.0-flash",
    description="Creates tickets for issues requiring human review.",
    instruction="Create support tickets for complex issues.",
    tools=[create_support_ticket]
)

# Create coordinator that routes to specialists
support_coordinator = LlmAgent(
    name="SupportCoordinator",
    model="gemini-2.0-flash",
    instruction="""You are a customer support coordinator. Route requests:
    - Order status/shipping → OrderAgent
    - Policy/FAQ questions → KnowledgeAgent  
    - Complex issues needing human help → EscalationAgent
    Always be helpful and professional.""",
    description="Routes customer requests to the appropriate specialist.",
    sub_agents=[order_agent, knowledge_agent, escalation_agent]
)

# Run the agent (conceptual)
# adk run support_coordinator
```

**Running ADK agents:**
```bash
# Create project
adk create my_agent

# Run with CLI
adk run my_agent

# Run with web UI (dev only)
adk web --port 8000
```

**ADK vs other frameworks:**

| Framework | Best for | Key difference |
| --------- | -------- | -------------- |
| **ADK** | Google ecosystem, Vertex AI, multi-agent | Workflow agents (Sequential, Parallel, Loop); Vertex AI Agent Engine deployment |
| **LangChain** | Prototyping, broad integrations | Chain-based; many connectors; LangGraph for agents |
| **LlamaIndex** | RAG-first applications | Data framework; strong indexing and retrieval |
| **CrewAI** | Role-based multi-agent | Crew metaphor with roles and tasks |

**Deployment options:**
- **Local**: `adk run` or `adk web` for development
- **Cloud Run**: Containerize and deploy as serverless
- **Vertex AI Agent Engine**: Managed, scalable agent hosting on GCP

> [!TIP]
> 💡 **Aha:** ADK's workflow agents (`SequentialAgent`, `ParallelAgent`, `LoopAgent`) let you build complex pipelines without custom orchestration code. Use `output_key` to pass data through shared state. Start with the coordinator pattern for multi-domain tasks.

---

## E.5 LLM Evaluation & Quality

**Why this comes next:** E.1–E.4 gave you the **request path** (serving, RAG, agents). The next question is **did we build the right thing?** This section answers that—quality, grounding, safety—so you can ship with confidence and iterate.

**What "knowledge quality" means here:** For LLM and RAG systems, quality is **groundedness** (is the answer supported by the context?), **relevance** (does it address the question?), and **retrieval quality** (did we fetch the right chunks?). You rarely have gold labels for every request, so evaluation mixes **reference-free** automated metrics (e.g. faithfulness, relevancy) with **sampled human review** to calibrate and catch edge cases. This section is tool-first: each concept is tied to frameworks you can run today.

---

### Evaluation Frameworks & Metrics

**RAGAS** (Python: `pip install ragas`) is the de facto open-source choice for **reference-free** RAG evaluation. You pass a dataset of `(user_input, retrieved_contexts, response)` plus optional `reference`; RAGAS runs LLM-as-judge and embedding-based metrics and returns scores. Used by LangChain, LlamaIndex, and LangSmith integrations.

| Metric                | What It Measures                      | How (in RAGAS)                                                                    | Tool                                                  |
| --------------------- | ------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------- |
| **Faithfulness**      | Is response grounded in context?      | LLM extracts claims → checks each against retrieved docs                          | `ragas.metrics.Faithfulness`                          |
| **Answer Relevancy**  | Does answer address the question?     | Inverse of LLM-generated “counterfactual” questions needed to recover answer      | `ragas.metrics.AnswerRelevancy`                       |
| **Context Precision** | Are relevant docs ranked above noise? | Ground-truth relevant items ranked high → higher score                            | `ragas.metrics.ContextPrecision` (needs ground truth) |
| **Context Recall**    | Did we retrieve what we need?         | Overlap between answer-supporting context and retrieved context; or vs. reference | `ragas.metrics.ContextRecall` / `LLMContextRecall`    |

**Practical RAGAS workflow:** Build a list of dicts with `user_input`, `retrieved_contexts`, `response`, and optionally `reference`. Load into `EvaluationDataset.from_list(dataset)`, then call `evaluate(dataset=..., metrics=[Faithfulness(), AnswerRelevancy(), ...], llm=evaluator_llm)`. Use a **different** LLM for evaluation than for generation to reduce self-consistency bias. See [RAGAS docs](https://docs.ragas.io/en/stable/getstarted/rag_eval/).

**Other tools:**

- **LangSmith** (LangChain): Predefined RAG evaluators (correctness, relevance, groundedness), dataset runs, human annotation queues, and online feedback. Use `client.run_evaluator` or the LangSmith UI to run evals on logged runs. Strong when your stack is already LangChain.
- **Giskard** (Python: `pip install giskard`): RAG Evaluation Toolkit (RAGET)—testset generation, knowledge-base–aware tests, and scalar metrics. Good for “test-suite” style regression and CI.
- **Arize Phoenix** (Python: `pip install arize-phoenix`): Open-source LLM tracing + evals. Phoenix Evals include **hallucination**, relevance, toxicity; they run over OpenTelemetry traces. Use for production monitoring and “eval on sampled traffic.”
- **Braintrust** (Python: `braintrust`): `Eval()` / `EvalAsync()` over datasets; you define **scorers** (functions that score outputs). Fits custom logic and proprietary benchmarks.
- **TruLens**: Focus on “RAG triad” (context relevance, grounding, relevance) with minimal config; integrates with LlamaIndex and other frameworks.

---

### Hallucination Detection: Approaches & Tools

| Approach                            | What It Does                                            | Accuracy | Latency         | Tools / How                                                                                                                                                               |
| ----------------------------------- | ------------------------------------------------------- | -------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Self-consistency**                | Sample N answers, check agreement                       | Moderate | High (N× calls) | Custom loop or Braintrust/Phoenix over multiple runs                                                                                                                      |
| **NLI / cross-encoder**             | Entailment model: premise = context, hypothesis = claim | High     | +50–100 ms      | Sentence-transformers NLI, or Phoenix “groundedness”–style evals                                                                                                          |
| **LLM-as-Judge**                    | “Is this claim supported by the context?”               | High     | +100–200 ms     | **RAGAS** `Faithfulness`, **LangSmith** groundedness, **Phoenix** hallucination template, **Braintrust** custom scorer                                                    |
| **Specialized faithfulness models** | Fine-tuned “faithfulness vs. hallucination” judge       | Highest  | ~+50 ms         | **Vectara FaithJudge** ([GitHub](https://github.com/vectara/FaithJudge)): benchmark + model for RAG QA/summarization; use when you need max agreement with human judgment |

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

> [!TIP]
> 💡 **Aha:** You don’t need gold labels for every request. **Reference-free** metrics (RAGAS faithfulness, answer relevancy, Phoenix hallucination) answer “is this grounded?” and “does this match the question?” without human annotations. Use them on a sample in production, then a **small human-labeled set** to set thresholds and sanity-check.

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

| Tool                     | What It Does                                                                   | When to Use                                           |
| ------------------------ | ------------------------------------------------------------------------------ | ----------------------------------------------------- |
| **RAGAS**                | Reference-free RAG metrics (faithfulness, relevancy, context precision/recall) | Batch RAG evals, CI, offline benchmarks; Python-first |
| **LangSmith**            | Evaluators, datasets, runs, human annotation                                   | LangChain-based apps; need UI + queues + feedback     |
| **Phoenix**              | Tracing + evals (hallucination, relevance, toxicity) over OTLP                 | Production monitoring, eval-on-sampled-traffic        |
| **Giskard**              | RAG test suite, testset generation, scalar metrics                             | Regression and “test suite” style RAG evaluation      |
| **Braintrust**           | Custom scorers, `Eval`/`EvalAsync`, experiments                                | Proprietary benchmarks, custom logic, experiments     |
| **FaithJudge** (Vectara) | Faithfulness/hallucination benchmark + model                                   | High-stakes RAG; max agreement with human judgment    |

---

### Evaluation data pipeline at scale

The metrics and tools above assume you have prediction data to evaluate. At scale, you need a **data pipeline**: predictions flow from the LLM → event stream → stream processor → evaluation/metrics layer and time-series store → dashboards and alerting. This is the _evaluation_ pipeline (log predictions, run quality/safety/cost metrics); the _training_ pipeline (user interactions → fine-tuning data) is E.6.

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

## E.6 GenAI Data Pipeline Architecture

**In the big picture** (see [GenAI System: Big Picture](#b1-genai-system-big-picture-frontend-to-backend)), this is the **training-data pipeline**: the path from "users interacted with the system" to "we have clean, formatted examples for fine-tuning." Evaluation (E.5) tells you _what_ to improve (quality, safety, drift); this pipeline gives you the _data_ to improve it (fine-tuning, RLHF, few-shot curation). It is _distinct_ from the evaluation pipeline (E.5), which moves _prediction_ data into metrics and alerts. Here we focus on **collecting user interactions** (prompts, responses, feedback), processing them at scale, and producing training-ready datasets.

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

| Component             | Google Cloud            | AWS                     |
| --------------------- | ----------------------- | ----------------------- |
| **Event Streaming**   | Pub/Sub                 | Kinesis Data Streams    |
| **Stream Processing** | Dataflow                | Kinesis Analytics       |
| **Data Lake**         | Cloud Storage           | S3                      |
| **Data Warehouse**    | BigQuery                | Redshift                |
| **Feature Store**     | Vertex AI Feature Store | SageMaker Feature Store |
| **Training**          | Vertex AI Training      | SageMaker Training      |
| **Orchestration**     | Vertex AI Pipelines     | SageMaker Pipelines     |

---

## E.7 Cost Optimization for GenAI Systems

**In the big picture** (see [GenAI System: Big Picture](#b1-genai-system-big-picture-frontend-to-backend)), this is **how we keep inference affordable**. E.1–E.6 gave you the request path (serving, RAG, agents), evaluation, and training data; **cost** and **scale** determine how you run it affordably and at customer load. Cost scales with tokens (input + output) and model tier, so optimization is about **reducing spend per request**—shorter prompts, caching, model routing, quantization, and when relevant fine-tuning ROI. _Throughput_ and _capacity_ are in E.8 Scalability; here we focus on _cost per request_.

**T-shaped summary:** Cost = f(tokens, model). Levers: prompt optimization, response/prompt caching, routing easy queries to smaller models, quantization, and continuous batching (better GPU use → same throughput with fewer machines). Deep dive below.

---

### Token-Based Cost Model

**Cost Components:**

- **Input tokens**: Tokens in prompt (including context)
- **Output tokens**: Generated tokens (typically 2-4x more expensive)
- **Model tier**: Different models have different costs

> [!TIP]
> 💡 **Aha:** GenAI cost scales with **length**, not just request count. A 10× longer prompt or answer can mean ~10× cost per call. So trimming context, caching prefixes, and routing easy queries to smaller models all directly lower spend.

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

| Technique          | Savings               | Trade-off                   |
| ------------------ | --------------------- | --------------------------- |
| Shorter prompts    | 20-40% input tokens   | May lose context            |
| Fewer examples     | 50-200 tokens/example | May reduce quality          |
| Prompt compression | Variable              | Compression cost vs savings |

**Few-shot sweet spot**: 2-3 examples usually sufficient. Research shows diminishing returns after 3 examples—the model has learned the pattern.

**2. Caching Strategy**

| Strategy         | Hit Rate          | Savings       | Best For       |
| ---------------- | ----------------- | ------------- | -------------- |
| Prompt caching   | High for prefixes | 2-5x speedup  | System prompts |
| Response caching | 10-30%            | 100% for hits | FAQ systems    |
| Semantic caching | 30-50%            | Varies        | Q&A systems    |

**3. Model Selection (Tiered Strategy)**

| Model                            | Cost                 | Quality | Use For               |
| -------------------------------- | -------------------- | ------- | --------------------- |
| **Large (GPT-4, Gemini Ultra)**  | $0.03-0.06/1K output | Best    | Complex reasoning     |
| **Medium (GPT-3.5, Gemini Pro)** | ≈$0.002/1K output    | Good    | Most production tasks |
| **Small (Gemini Flash)**         | ≈$0.001/1K output    | Basic   | Simple, high-volume   |

**Model Routing Strategies:**

| Strategy            | How It Works                                       | Savings           |
| ------------------- | -------------------------------------------------- | ----------------- |
| **Routing**         | Classify query → send to single optimal model      | 40-60%            |
| **Cascading**       | Start small → escalate to larger if low confidence | 50-80%            |
| **Cascade Routing** | Combines both: route + escalation                  | Best cost/quality |

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

> [!TIP]
> 💡 **Aha:** Routing and cascading both assume "hard" and "easy" queries. If you can **predict** hardness (e.g. by query length, intent, or a tiny classifier), you send easy ones to small/cheap models and reserve the big model for the rest. The leverage comes from that prediction being cheap and reasonably accurate.

**4. Fine-tuning ROI**

- **Upfront cost**: $100-1000s
- **Break-even**: If fine-tuning costs $1000 and saves $0.001 per request, break-even at 1M requests
- **Benefits**: Better quality for domain, can use smaller base model

**5. Quantization**

Reducing numerical precision shrinks model size and speeds inference. **FP32** (32-bit float), **FP16** (16-bit), **INT8** (8-bit integer), **INT4** (4-bit) are common levels.

| Precision   | Memory Reduction | Quality Loss |
| ----------- | ---------------- | ------------ |
| FP32 → FP16 | 2x               | Minimal      |
| FP16 → INT8 | 4x               | Some         |
| INT8 → INT4 | 8x               | Significant  |

**Why FP16 is safe**: Modern **GPUs** (graphics processing units) have Tensor Cores optimized for FP16. Quality loss is minimal (<1%) but memory/cost savings are significant.

> [!TIP]
> 💡 **Aha:** Weights don’t need 32-bit precision for good answers; most signal lives in a smaller range. Quantization **compresses** that range (FP32→FP16→INT8→INT4). You trade a little quality for large memory and speed gains. FP16 is the first step almost everyone takes because hardware is built for it and the drop is tiny.

**6. Continuous Batching**

- Static batching: 40–60% GPU utilization
- Continuous batching: 80–95% GPU utilization
- **Result**: 2–3× higher throughput → fewer machines for the same load (cost and scale). Throughput/parallelism patterns (model parallelism, pipeline parallelism) are in E.8.

---

## E.8 Scalability Patterns for GenAI

**In the big picture** (see [GenAI System: Big Picture](#b1-genai-system-big-picture-frontend-to-backend)), this is **how we serve more load**: the LLM layer is GPU-heavy and stateful (KV cache), so scaling is about **throughput and capacity**—horizontal replication, model/pipeline parallelism, and caching that increases effective req/s. _Cost per request_ is in E.7; here we focus on _requests per second_ and _utilization_.

**T-shaped summary:** Levers: stateless serving (more replicas), model parallelism (split layers across GPUs), pipeline parallelism (different layers on different GPUs), and caching (KV cache for prefixes, response cache for identical/similar queries). Deep dive below.

---

### Horizontal Scaling

**Challenge**: LLM inference is GPU-intensive and stateful (KV cache).

**Solutions:**

| Pattern                  | Description                            | Trade-off                                  |
| ------------------------ | -------------------------------------- | ------------------------------------------ |
| **Stateless Serving**    | Load balancer → Multiple LLM servers   | Higher memory (each server has full model) |
| **Model Parallelism**    | Split model across GPUs                | Communication overhead                     |
| **Pipeline Parallelism** | Different GPUs handle different layers | Better utilization                         |

**Model Parallelism Visual:**

```
Input → GPU 1 (Layers 1-10) → GPU 2 (Layers 11-20) → GPU 3 (Layers 21-30) → Output
```

### Caching Strategies for Scale

_Cost_ impact of caching is in E.7; here we focus on **throughput** impact: same hardware serves more requests when prefixes or responses are reused.

| Strategy                  | Throughput / latency impact                               | Best For                            |
| ------------------------- | --------------------------------------------------------- | ----------------------------------- |
| Prompt caching (KV cache) | 2–3× effective throughput for repeated prefixes           | System prompts, long context        |
| Response caching          | Near-instant for cache hits; frees GPU for other requests | Identical or near-identical queries |
| Semantic caching          | Higher hit rate → more requests served from cache         | Similar queries (e.g. Q&A)          |

### Training Efficiency Techniques

Training large GenAI models (billions of parameters) requires specialized techniques. These also matter for **fine-tuning** in production.

**1. Gradient Checkpointing**

Instead of storing all activations during forward pass (memory-hungry), store only a subset and **recompute** the rest during backward pass. Trade-off: **2–3× less memory** for **~20% more compute**.

**2. Mixed Precision Training (AMP)**

Use **FP16** (16-bit) for most operations, **FP32** (32-bit) only where needed (e.g., loss scaling). Benefits:
- **2× less memory** (weights + activations)
- **2–3× faster** on modern GPUs (Tensor Cores)
- Minimal quality loss with proper loss scaling

**3. Distributed Training**

| Technique | What it does | When to use |
| --------- | ------------ | ----------- |
| **Data Parallelism** | Same model on each GPU; split data across GPUs; sync gradients | Model fits in one GPU; large dataset |
| **Model (Tensor) Parallelism** | Split layers/tensors across GPUs (e.g., split matrix multiply) | Single layer too large for one GPU |
| **Pipeline Parallelism** | Different layers on different GPUs; micro-batch pipelining | Very deep models (many layers) |
| **Hybrid (3D) Parallelism** | Combine data + tensor + pipeline | Training 100B+ parameter models |

**4. ZeRO and FSDP**

- **ZeRO** (Zero Redundancy Optimizer, Microsoft): Shards optimizer states, gradients, and parameters across GPUs to reduce memory redundancy.
- **FSDP** (Fully Sharded Data Parallel, Meta/PyTorch): Similar to ZeRO; shards model parameters across GPUs and gathers them on-demand.

Both enable training **larger models** on the same hardware by eliminating redundant copies.

> [!TIP]
> 💡 **Aha:** In interviews, if asked "how would you train a 70B model on 8 GPUs?", the answer combines: **FSDP or ZeRO** (shard parameters), **gradient checkpointing** (reduce activation memory), **mixed precision** (FP16), and possibly **pipeline parallelism** if layers are very large.

---

## E.9 Monitoring & Observability for GenAI

**In the big picture** (see [GenAI System: Big Picture](#b1-genai-system-big-picture-frontend-to-backend)), this is **how we observe the system**: metrics, traces, and drift detection across the request path and the evaluation/training pipelines. Quality metrics and eval pipeline are in E.5; here we focus on **what to track** and **which platform services** support it.

**T-shaped summary:** Track quality (task accuracy, safety), performance (latency, throughput), cost (tokens, model tier), reliability (errors, timeouts), and safety (toxicity, jailbreak). Use Cloud Monitoring / CloudWatch, logging, tracing (Trace / X-Ray), and model monitoring for drift. Deep dive below.

---

### Key Metrics to Track

| Category        | Metrics                                             |
| --------------- | --------------------------------------------------- |
| **Quality**     | Task accuracy, ROUGE/BLEU, human evaluation         |
| **Performance** | P50/P95/P99 latency, throughput, tokens/second      |
| **Cost**        | Cost per request, token usage, model tier breakdown |
| **Reliability** | Error rate, timeout rate, availability              |
| **Safety**      | Toxicity score, jailbreak attempts, bias detection  |

### Platform Services

| Function            | Google Cloud                           | AWS                     |
| ------------------- | -------------------------------------- | ----------------------- |
| **Metrics**         | Cloud Monitoring, Vertex AI Monitoring | CloudWatch              |
| **Logging**         | Cloud Logging                          | CloudWatch Logs         |
| **Tracing**         | Cloud Trace                            | X-Ray                   |
| **Drift Detection** | Vertex AI Model Monitoring             | SageMaker Model Monitor |

---

## E.10 Security & Guardrails

**In the big picture** (see [GenAI System: Big Picture](#b1-genai-system-big-picture-frontend-to-backend)), this is **how we protect the system**: inputs (prompt injection, jailbreak, PII), outputs (harmful content, PII leakage), and access (IAM, API keys). Guardrails sit _around_ the request path—input checks before the LLM, output checks after—and work with HTTP-level protections (Cloud Armor, WAF) and data protection (DLP).

**T-shaped summary:** Threats: direct/indirect prompt injection, data leakage, jailbreaking, unauthorized access. Mitigations: input/output guardrails, spotlighting, least-privilege tools, Model Armor (or Bedrock Guardrails). Use defense-in-depth: gateway → guardrails → LLM → guardrails → response. Deep dive below.

---

### Key Security Concerns

> [!IMPORTANT]
> 💡 **Aha:** LLMs take natural language as input, so **any** user text can be an attempt to override instructions ("Ignore previous instructions…"). Guardrails and defense-in-depth exist because you can't whitelist "good" prompts—you have to detect and constrain _malicious_ or out-of-scope intent at the boundary.

| Threat                        | Risk                                                                                 | Mitigation                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| **Direct Prompt Injection**   | User injects malicious instructions                                                  | Input validation, guardrails                                        |
| **Indirect Prompt Injection** | Hidden instructions in external content                                              | Content isolation, spotlighting                                     |
| **Data Leakage**              | Training data memorization, **PII** (personally identifiable information) in outputs | Output filtering, **DLP** (data loss prevention)                    |
| **Jailbreaking**              | Bypassing safety controls                                                            | Multi-layer defense, red teaming                                    |
| **Access Control**            | Unauthorized model access                                                            | **IAM** (identity and access management), API keys, least privilege |

### Prompt Injection Defense-in-Depth

| Layer          | Technique        | Description                                 |
| -------------- | ---------------- | ------------------------------------------- |
| **Input**      | Spotlighting     | Clearly delimit user input vs system prompt |
| **Input**      | Input validation | Regex, blocklists, encoding detection       |
| **Input**      | Guardrails check | Detect injection attempts before LLM        |
| **Processing** | Least privilege  | Limit tools/data agent can access           |
| **Output**     | Guardrails check | Validate output aligns with user intent     |
| **Output**     | PII filtering    | Detect/redact sensitive data                |

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

| Threat                 | Cloud Armor | Model Armor      |
| ---------------------- | ----------- | ---------------- |
| SQL injection in HTTP  | ✅          | ❌ (not its job) |
| DDoS / rate limiting   | ✅          | ❌               |
| **Prompt injection**   | ❌          | ✅               |
| **Jailbreak attempts** | ❌          | ✅               |
| **PII in LLM output**  | ❌          | ✅               |

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

| Regulation                                                      | Key Requirements                                       |
| --------------------------------------------------------------- | ------------------------------------------------------ |
| **GDPR** (General Data Protection Regulation)                   | Right to explanation, data deletion, privacy by design |
| **HIPAA** (Health Insurance Portability and Accountability Act) | Healthcare data protection, audit logging              |
| **PCI-DSS** (Payment Card Industry Data Security Standard)      | Payment data security, no storage of card numbers      |

### Security Stack Summary

| Layer               | Google Cloud                                 | AWS                            |
| ------------------- | -------------------------------------------- | ------------------------------ |
| **LLM Security**    | Model Armor                                  | Bedrock Guardrails             |
| **HTTP Security**   | Cloud Armor                                  | WAF (web application firewall) |
| **Data Protection** | Cloud DLP (data loss prevention)             | Macie                          |
| **Secrets**         | Secret Manager                               | Secrets Manager                |
| **Network**         | VPC (virtual private cloud) Service Controls | VPC                            |
| **Access**          | IAM (identity and access management)         | IAM                            |
| **Audit**           | Cloud Audit Logs                             | CloudTrail                     |

### Post-Processing for Bias and Safety

Beyond security threats, LLM outputs require **post-processing** to ensure they are unbiased, respectful, and appropriate. This is especially important for user-facing features like autocomplete, chatbots, and content generation.

**Common Post-Processing Strategies:**

| Strategy | What it does | Example |
| -------- | ------------ | ------- |
| **Pronoun Replacement** | Replace gender-specific pronouns with neutral alternatives | "he/she" → "they" when gender unknown |
| **Gender-Neutral Words** | Replace gendered terms with neutral equivalents | "chairman" → "chairperson", "policeman" → "police officer" |
| **Sensitive Term Filtering** | Flag and replace terms implying age, race, disability bias | Predefined blocklist with neutral alternatives |
| **NSFW Filtering** | Detect and remove explicit language | Keyword lists + pattern matching + classifier |
| **Confidence Thresholding** | Only show suggestions above confidence threshold | Suppress low-confidence predictions |
| **Length Filtering** | Remove suggestions that are too long or too short | Max 10 words for autocomplete suggestions |

> [!TIP]
> 💡 **Aha:** Post-processing is **cheap and fast**—rule-based checks run in microseconds. They're your last line of defense before output reaches users. Combine with Model Armor/Guardrails for defense-in-depth.

---

## F.1 Real-World Examples: Applying the Stack

This section is where **theory meets shipping**: real stacks (LangChain, Vertex, Bedrock, vLLM), real numbers (tokens, cost, latency), and customer-ready scenarios. It comes **after** all core concepts (E.1–E.10) so every term is defined. Each example follows the same **45-minute Interview Framework** from the [Quick Reference](#interview-framework-45-min-structure)—Clarify Requirements → High-Level Architecture → Deep Dive → Bottlenecks & Trade-offs—so you can practice answering in a structured way. We spell out _why_ each requirement matters, add **back-of-the-envelope estimations** (tokens, cost, latency) so you can practice doing the math in an interview, and point to concrete stacks (**LangChain** / **LlamaIndex**, **Vertex AI** / **Bedrock**, vLLM, RAGAS, etc.). Use these as interview-style walkthroughs, not as bullet lists to memorize. For **end-to-end solutioning** (Scope → Design → Deploy → Communicate) with hypotheticals, stakeholder loop-in, and presenting to CxO vs Product vs live customer, see [Quick Reference: End-to-end solutioning](#end-to-end-solutioning-scope--design--deploy--communicate)—it uses F.1-style designs inside a full Scope/Design/Deploy/Communicate flow with worked examples.

---

### Example 1: Code Generation Assistant (like GitHub Copilot)

_In an interview you’d start by clarifying what “good” looks like: how fast, how accurate, and what we’re willing to pay. Then you’d sketch the path from IDE to model and back._

**1. Clarify Requirements (5–10 min)**

| Dimension        | What to pin down                                                                                                            | Why it matters                                                                                           |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Token budget** | Input: ~2K tokens (prefix + cursor context); output: 20–100 tokens per completion. Cap total context at e.g. 8K.            | Larger context = higher cost and slower TTFT; you need a hard cap for pricing and latency.               |
| **Latency**      | P95 < 200 ms time-to-first-token for inline completions. Batch jobs (e.g. index workspace) can be 1–2 s.                    | Users feel lag above ~200 ms; the rest of the budget goes to gateway, RAG, and model.                    |
| **Quality**      | Completions must compile and match project style. Low tolerance for hallucination.                                          | Wrong or irrelevant suggestions hurt trust; you’ll measure correctness and relevance (RAGAS, LangSmith). |
| **Cost**         | Per-token pricing; monthly budget. Prefer smaller/faster models and routing by complexity (E.7).                             | Cost scales with active devs × completions per day × tokens; routing keeps easy cases cheap.             |
| **Safety**       | No PII/secrets in prompts or logs; optional filters; Model Armor / Bedrock Guardrails. Data residency if code is sensitive. | Code can contain secrets; compliance may require “data never leaves region.”                             |

📊 **Rough estimation (code assistant)**

- **Volume:** 50 completions per dev per day × 2K input + 50 output ≈ 100K input + 2.5K output tokens per dev/day. For 500 devs: **~50M input + 1.25M output tokens/day**.
- **Cost (ballpark):** At ≈$0.25/1M input and ≈$0.50/1M output (small code model): 50 × 0.25 + 1.25 × 0.50 ≈ **$14/day** ≈ **$400/month** for LLM only. Caching and routing can cut this 30–50%.
- **Latency budget (200 ms target):** Gateway < 10 ms, RAG (embed + retrieve) < 50 ms, LLM TTFT < 140 ms. So you need a small/fast model and a lean RAG path.

**2. High-Level Architecture (10–15 min)**

- **Flow:** IDE → API gateway (auth, rate limit) → orchestration (RAG: embed + retrieve code context) → LLM (completion) → post-process (format, length cap) → response.
- **Components:** API gateway (e.g. Cloud Run); orchestration = **LangChain** or **LlamaIndex**; RAG = vector store (Chroma, Pinecone) + embeddings; LLM = **Vertex AI Codey** / **Bedrock** CodeWhisperer or **vLLM** (CodeLlama, StarCoder).
- **APIs:** POST /complete (prefix, cursor, options); optional indexing API for workspace sync.
- **Include:** RAG for context, caching (same prefix → reuse or KV cache), model routing (simple vs complex → small vs larger model).

**3. Deep Dive (15–20 min)**

- **RAG:** Chunk by file/function (e.g. **LlamaIndex** CodeIndex, **LangChain** by language); code-capable embeddings; top-k on cursor context; optional rerank. Keep chunks small to stay within token budget.
- **Model & routing:** Small model for most completions; route to larger model when context is big or a complexity heuristic fires (E.7).
- **Eval & observability:** **RAGAS** / **LangSmith** on (prompt, context, completion); **Phoenix** for production traces and latency.
- **Security:** Length limits; PII/secret filters; **Model Armor** / **Bedrock Guardrails**; no raw code in logs for sensitive repos.

**4. Bottlenecks & Trade-offs (5–10 min)**

- **KV cache:** 2–8K context keeps memory reasonable; limit concurrency per GPU or use continuous batching (vLLM).
- **Quality vs cost:** Smaller model = cheaper and faster, but may drop quality on complex code; routing balances both.
- **Latency vs throughput:** Inline = low latency, one request at a time; batch indexing can use batching for throughput.
- **Single vs multi-agent:** One “completion + context” path is enough here; multi-agent adds complexity without clear benefit.

🛠️ **Stack snapshot:** LangChain/LlamaIndex (RAG + routing) + Vertex Codey or Bedrock + vLLM (optional) + RAGAS/LangSmith/Phoenix (eval) + guardrails.

---

### Example 2: Customer Service Chatbot with RAG and Tools

_Here the user expects an answer that’s grounded in your docs and in real data (orders, tickets). You need to clarify how fast answers should be, how much you’re willing to spend per conversation, and what “correct” means (faithful to sources, no made-up policies)._

**1. Clarify Requirements (5–10 min)**

| Dimension        | What to pin down                                                                                             | Why it matters                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| **Token budget** | Conversation: 4–32K context per turn; RAG: 2–4K retrieved tokens. Cap response at e.g. 500 tokens.           | Long context = higher cost and slower; you need a cap for pricing and latency.                                       |
| **Latency**      | P95 < 3–5 s full response (RAG + tool calls + LLM); TTFT < 1 s so the user sees something quickly.           | Users wait for a full answer; TTFT < 1 s keeps the UI feeling responsive.                                            |
| **Quality**      | Faithful to docs and tool outputs; no hallucinated policies. Relevancy of answers.                           | Wrong or irrelevant answers hurt trust and compliance; RAGAS faithfulness + relevancy + human review on escalations. |
| **Cost**         | Per-token; monthly budget. Cache frequent questions; smaller model for simple intents (E.7).                  | Cost = conversations × turns × tokens; caching and routing cut cost.                                                 |
| **Safety**       | Compliance (PCI, PII); no leaking internal docs or customer data. Guardrails; PII filtering in tool outputs. | One leak can be catastrophic; guardrails and least-privilege tools are non-negotiable.                               |

📊 **Rough estimation (chatbot)**

- **Volume:** 10K conversations/day × 5 turns × (3K input + 200 output) ≈ **150M input + 10M output tokens/day** (order of magnitude; adjust by real usage).
- **Cost (ballpark):** At ≈$0.50/1M input and ≈$1.50/1M output (mid-tier chat model): 150 × 0.5 + 10 × 1.5 = 75 + 15 = **$90/day** ≈ **$2.7K/month** for LLM. Response cache (e.g. 20% hit rate) and routing simple queries to a smaller model can cut this 25–40%.
- **Latency budget (4 s target):** Gateway < 50 ms, agent + RAG retrieval < 500 ms, tool calls 1–2 × 200 ms = 200–400 ms, LLM (first token) < 1 s, LLM (full) < 2 s. So RAG and tools must be fast; LLM carries most of the latency.

**2. High-Level Architecture (10–15 min)**

- **Flow:** User → API gateway → orchestration (agent) → [RAG retriever + tools (order, ticket, escalate)] → LLM → post-process (format, guardrails) → response.
- **Components:** API gateway; orchestration = **LangChain** `create_react_agent` or **LlamaIndex** `ReActAgent`; RAG = **Vertex RAG Engine** or **Bedrock Knowledge Bases** (or LangChain + Chroma/Pinecone); LLM = **Vertex AI** (Gemini) or **Bedrock** (Claude, Llama); tools = MCP or custom APIs (orders, CRM, escalation).
- **Data flow:** Query → agent picks tool vs RAG vs direct answer → RAG returns top-k chunks; tools return structured data → LLM synthesizes; optional rerank before injection.
- **Include:** RAG (knowledge base), caching (response or semantic cache for frequent Qs), model routing (simple FAQ vs multi-tool).

**3. Deep Dive (15–20 min)**

- **RAG:** Chunk by semantic units (e.g. 512 tokens) or doc/section; Vertex/Bedrock or Cohere embeddings; hybrid retrieval if you need keyword + vector; rerank to top-5 before putting in context (E.2).
- **Model & routing:** One model for chat + tool use (Gemini, Claude); optional routing: small model for FAQ-only, larger for multi-step.
- **Eval & observability:** **RAGAS** (faithfulness, answer relevancy) on logged (query, context, response); **LangSmith** for datasets and human review; track escalation rate and tool success.
- **Security:** **Model Armor** / **Bedrock Guardrails** on input/output; IAM and least privilege on tools; filter PII in tool _outputs_ before they reach the LLM or user (E.10).

**4. Bottlenecks & Trade-offs (5–10 min)**

- **KV cache:** 32K context per turn increases memory; summarize or truncate history to control length.
- **Quality vs cost:** Larger model = better tool use; smaller + routing cuts cost for simple queries.
- **Latency vs throughput:** Tool calls add round-trips; parallelize where possible; async for non-blocking flows (e.g. ticket creation).
- **Single vs multi-agent:** One agent with tools (RAG + order + ticket + escalate) is the norm; multi-agent only if you need distinct roles and more capability.

🛠️ **Stack snapshot:** LangChain/LlamaIndex (agent + tools) + Vertex RAG Engine or Bedrock Knowledge Bases + Vertex/Bedrock LLM + RAGAS/LangSmith (eval) + Model Armor/Bedrock Guardrails.

**In production:** Full customer engagement often adds **Agent Assist** (suggested responses, knowledge-base hints, real-time transcribe/summarize when escalating to humans) and **Conversational Insights** (sentiment, topics, Generative FAQ for FAQ gaps and trending questions). A full contact center runs on **CCaaS** (omnichannel, multimodal, agent routing) with Conversational Agents + Agent Assist + Insights on top—see E.4 Customer engagement & contact center.

---

### Example 3: Content Generation Platform (research → draft → grounding)

_This is a multi-step pipeline: research from the web, then draft, then fact-check against sources, then SEO. Users typically accept 30–90 s end-to-end (async). You need to clarify token caps per step, cost per article, and how strict “faithful to sources” is._

**1. Clarify Requirements (5–10 min)**

| Dimension        | What to pin down                                                                                                   | Why it matters                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| **Token budget** | Research: 10–50K tokens (snippets); draft: 2–4K output; grounding: full draft + sources. Per-step caps.            | Unbounded research or draft blows cost; caps keep pricing predictable.        |
| **Latency**      | End-to-end 30–90 s (async). Per-step: research ~5–10 s, draft ~15–30 s, grounding ~10–20 s.                        | Users expect “background” generation; per-step times drive capacity planning. |
| **Quality**      | High faithfulness: every claim grounded in sources. RAGAS faithfulness; optional human spot-checks.                | Ungrounded claims damage trust; you’ll measure and monitor faithfulness.      |
| **Cost**         | Per-token; routing: Flash/small for research + SEO, Pro/large for draft (E.7); monthly budget and per-article caps. | Most tokens are in research + draft; routing keeps research/SEO cheap.        |
| **Safety**       | No harmful or copyrighted content; cite sources; optional guardrails on output.                                    | Citations and guardrails protect you and the reader.                          |

📊 **Rough estimation (content platform)**

- **Volume (per article):** Research input ~20K tokens (snippets), draft input ~20K + output ~3K, grounding input ~25K. Total ≈ **68K tokens per article** (input-heavy). At 100 articles/day: **~6.8M tokens/day** (mix of Flash and Pro).
- **Cost (ballpark):** Assume 70% on Flash (≈$0.15/1M input, ≈$0.60/1M output) and 30% on Pro (≈$0.50/1M input, ≈$1.50/1M output). Rough: 100 articles × (≈50K Flash + ≈18K Pro) → **≈$15–25/day** ≈ **$500–750/month** for LLM. Caching research for similar briefs can cut 10–20%.
- **Latency (per article, ~60 s target):** Research 5–10 s (search API + optional summarization), draft 15–30 s (depends on length), grounding 10–20 s (retrieval + check), SEO 2–5 s. Bottleneck is usually the draft step; you can parallelize multiple research queries.

**2. High-Level Architecture (10–15 min)**

- **Flow:** Brief → API gateway → orchestration (sequential pipeline) → [research (search) → draft (LLM) → grounding (RAG/Vertex grounding) → SEO (template or LLM)] → post-process (citations, format) → output.
- **Components:** API gateway; orchestration = **LangChain** `SequentialChain` or DAG; research = **Tavily** / **Google Search** / Vertex Search; draft = **Vertex AI** (Gemini) or **Bedrock** (Claude); grounding = **Vertex AI grounding** or **Bedrock** retrieval + cite-check; SEO = small LLM or templates.
- **Data flow:** Brief → research returns snippets → draft LLM with snippets as context → grounding checks claims vs sources → SEO → multi-format output.
- **Include:** RAG/grounding (sources as retrieval), caching (reuse research for similar briefs if TTL ok), model routing (Flash for research/SEO, Pro for draft).

**3. Deep Dive (15–20 min)**

- **RAG / grounding:** Research = search API (ranked snippets). Grounding = evidence per claim via **Vertex grounding with Google Search** / **Bedrock** retrieval, or NLI-style / RAGAS faithfulness on (claim, source). Chunking matters if you build your own source KB.
- **Model & routing:** **Vertex** / **Bedrock**; Flash for research summarization and SEO, Pro for full draft (E.7).
- **Eval & observability:** **RAGAS** faithfulness and relevancy on (brief, sources, draft); **LangSmith** / **Braintrust** for A/B prompts and models; optional **Giskard** for regression.
- **Security:** Input/output guardrails; source attribution and citation; no unsanctioned content in final output without citation.

**4. Bottlenecks & Trade-offs (5–10 min)**

- **KV cache:** 50K research context increases memory per request; truncate or summarize research before the draft step.
- **Quality vs cost:** Pro for draft improves quality; Flash for research and SEO keeps cost down.
- **Latency vs throughput:** Sequential steps; parallelize only where independent (e.g. multiple research queries). Batch similar briefs for throughput if async.
- **Single vs multi-agent:** One sequential chain (research → draft → grounding → SEO) is the default; splitting into multiple agents (researcher vs writer) adds flexibility and complexity—use only if you need distinct roles.

🛠️ **Stack snapshot:** LangChain (sequential pipeline + tools) + Vertex/Bedrock LLMs + Vertex grounding or RAG + RAGAS (eval) + optional Giskard for regression tests.

**Variant: internal knowledge workers (Gemini Enterprise).** For **internal** users (e.g. advisors, analysts), **Gemini Enterprise** offers agents + **unified search** across connected business systems (not just uploaded docs). Use **trusted/curated sources only** (e.g. government reports, internal research). **Plan-then-verify-then-execute:** agent proposes a research plan → human verifies → agent executes (searches, asks new questions, iterates) → output = report + source links + optional **audio summary**. **NotebookLM Enterprise** = deep dive into specific documents/sources (Q&A, summarize); Gemini can connect to it for personalized context (e.g. client notes). See E.4 Enterprise knowledge workers (Gemini Enterprise).

---

### Example 4: Smart Compose / Email Autocomplete (like Gmail)

_Real-time text completion as users type. Key constraints: imperceptible latency (<100 ms), high consistency (deterministic), and bias-free suggestions. This is a classic decoder-only Transformer use case._

**1. Clarify Requirements (5–10 min)**

| Dimension | What to pin down | Why it matters |
| --------- | ---------------- | -------------- |
| **Latency** | P99 < 100 ms (imperceptible). Suggestion must appear before user types next character. | Any visible lag breaks the UX; users type faster than slow models can respond. |
| **Token budget** | Input: partial email (100–500 tokens) + context (subject, recipient). Output: 2–10 tokens (short phrase). | Short outputs = fast; long suggestions are ignored anyway. |
| **Quality** | High acceptance rate; completions must be grammatically correct and contextually relevant. | Users reject bad suggestions; acceptance rate is the key online metric. |
| **Consistency** | Deterministic: same input → same suggestion. No surprising outputs. | Users expect predictable, repeatable behavior for autocomplete. |
| **Safety** | No biased assumptions (gender, race, age); no inappropriate content. | Suggestions are visible instantly; post-processing for bias is essential. |
| **Scale** | 1.8B users; up to 500 emails/user/day; not all trigger suggestions. Assume 10% of keystrokes trigger. | Massive scale; model must be small/fast; caching is critical. |

📊 **Rough estimation (email autocomplete)**

- **Volume:** Assume 100M active sessions/day, 20 suggestions/session = **2B suggestion requests/day** = ~23K QPS average.
- **Token budget:** ~200 input + ~5 output per request = 205 tokens. At 2B requests: **~400B input + ~10B output tokens/day**.
- **Cost (if using external API—not practical at this scale):** At ≈$0.10/1M tokens: $40K/day. **Must use internal/self-hosted model** (small, distilled).
- **Latency budget (100 ms):** Triggering < 5 ms, inference < 80 ms, post-processing < 15 ms. Requires small model + on-device or edge inference.

**2. High-Level Architecture (10–15 min)**

```
User Typing → Triggering Service → Phrase Generator (Beam Search) → Filtering (length, confidence) → Post-Processing (bias) → Display Suggestion
```

- **Triggering Service**: Monitors keystrokes. Only triggers model when:
  - User has typed enough context (e.g., 3+ words)
  - Pause in typing (e.g., 100ms since last keystroke)
  - Not in the middle of a word
- **Phrase Generator**: Decoder-only Transformer with beam search (beam width 3–5). Returns top-k completions with confidence scores.
- **Filtering**: Remove suggestions that are (a) too long (>10 words), (b) low confidence (<0.15), (c) duplicates.
- **Post-Processing**: Rule-based bias removal—pronoun neutralization, gender-neutral terms, NSFW filtering.
- **Response**: Top remaining suggestion (or nothing if all filtered out).

**Components:**
- **Model**: Small decoder-only Transformer (~100M–1B params), distilled from larger model. Trained in two stages: (1) pretrain on general web text, (2) finetune on email corpus.
- **Tokenization**: Subword (BPE or SentencePiece) for vocabulary efficiency.
- **Sampling**: Beam search (deterministic, consistent).
- **Serving**: On-device (mobile) or edge (low-latency regions). Not practical to hit cloud LLM per keystroke.

**3. Deep Dive (15–20 min)**

- **Model architecture**: Decoder-only Transformer; positional encoding (fixed sine-cosine for generalization); 6–12 layers; ~100M params for on-device.
- **Training**: (1) Pretrain on web corpus (Common Crawl); (2) Finetune on anonymized email corpus. ML objective = next-token prediction; loss = cross-entropy.
- **Input context**: Combine email body + subject + recipient in prompt template:
  ```
  [Subject: {subject}]
  [To: {recipient}]
  [Body: {partial_body}]
  ```
- **Beam search**: Track top 3 sequences; prune at each step; stop at `<EOS>` or max 10 tokens.
- **Post-processing rules**: Replace "he/she" → "they"; "chairman" → "chairperson"; blocklist for sensitive terms; NSFW keyword filter.
- **Evaluation**:
  - Offline: **Perplexity** (lower = better prediction), **ExactMatch@3** (% of 3-word predictions that match ground truth)
  - Online: **Acceptance rate** (% suggestions accepted), **Usage rate** (% emails using feature), **Avg completion time reduction**

**4. Bottlenecks & Trade-offs (5–10 min)**

- **Latency vs quality**: Smaller model = faster but less accurate. Distillation from larger model helps.
- **Consistency vs diversity**: Beam search gives consistency; if diversity needed (e.g., creative writing), switch to top-p sampling.
- **Personalization vs cold start**: Personalized models improve acceptance rate but require per-user data; start with global model, add personalization later.
- **On-device vs cloud**: On-device = fastest latency, no network cost; cloud = larger model, easier updates. Hybrid: on-device for common cases, cloud fallback for complex.
- **Triggering sensitivity**: Trigger too often = annoying; too rarely = missed opportunities. A/B test threshold.

🛠️ **Stack snapshot:** Small decoder-only Transformer (distilled) + on-device serving (TFLite, Core ML) or edge (Cloud Run, Lambda@Edge) + beam search + rule-based post-processing + Perplexity/ExactMatch@N eval + acceptance rate monitoring.

---

### Example 5: Language Translation Service (like Google Translate)

_Sequence-to-sequence transformation: source language → target language. Uses encoder-decoder architecture with cross-attention. Key decisions: bilingual vs multilingual models, language detection, and handling named entities._

**1. Clarify Requirements (5–10 min)**

| Dimension | What to pin down | Why it matters |
| --------- | ---------------- | -------------- |
| **Languages** | How many? Start with 4 (English, Spanish, French, Korean). Plan for 130+. | Bilingual = N×(N-1) models; multilingual = 1 model. Huge difference in complexity. |
| **Input length** | Up to 1,000 words; longer documents chunked. | Affects context window, memory, latency. |
| **Language detection** | Auto-detect source language (users may not know). | Need separate language detector component. |
| **Latency** | P95 < 500 ms for short text; longer for documents. | Real-time for chat; async acceptable for documents. |
| **Quality** | High accuracy; must handle idioms, grammar, named entities. | BLEU/METEOR benchmarks; user feedback loop. |
| **Offline support** | Cloud-first; on-device for mobile (optional). | Cloud = larger models; on-device = smaller, quantized. |

📊 **Rough estimation (translation service)**

- **Volume:** 1B users × avg 2 translations/day = **2B translations/day** = ~23K QPS.
- **Token budget:** Avg 50 words input → ~75 tokens; output similar. ~150 tokens/request.
- **Cost (if external API):** 2B × 150 tokens = 300B tokens/day. At $0.10/1M = $30K/day. **Self-hosted is essential at this scale.**
- **Latency budget (500 ms):** Language detection < 50 ms, encoding < 100 ms, decoding < 300 ms (beam search), post-processing < 50 ms.

**2. High-Level Architecture (10–15 min)**

```
User Input → Language Detector → Translation Service (Encoder-Decoder + Beam Search) → Post-Processing → Output
```

**Components:**

1. **Language Detector**: Encoder-only Transformer + classification head. Classifies input into N languages.
2. **Translation Service**: Routes to appropriate model based on (source, target) pair.
   - Option A: **Bilingual models** — One model per language pair (e.g., EN→FR, EN→ES). Higher quality, but N×(N-1) models.
   - Option B: **Multilingual model** — Single model for all languages. Simpler, but may sacrifice quality.
3. **Beam Search**: Deterministic decoding for consistent translations.
4. **Post-Processing**: Handle named entities (restore placeholders), formatting, punctuation.

**Architecture Choice: Encoder-Decoder**

| Component | Why |
| --------- | --- |
| **Encoder** | Bidirectional attention; fully understands source before generating |
| **Decoder** | Causal attention (masked); generates target one token at a time |
| **Cross-Attention** | Decoder attends to encoder outputs; aligns source with target |

**3. Deep Dive (15–20 min)**

- **Tokenization**: Subword (BPE or SentencePiece) — handles multiple languages efficiently, ~50K–100K vocab.
- **Named Entity Handling**: Replace entities (names, places, URLs) with placeholders before translation; restore after.
  ```
  Input:  "The California city, Burlingame, is named after Anson Burlingame."
  Masked: "The ENTITY_1 city, ENTITY_2, is named after ENTITY_3."
  Translate → then restore ENTITY_1 = California, etc.
  ```
- **Training (Two-Stage)**:
  1. **Pretraining**: MLM (masked language modeling) on multilingual web corpus (C4, Wikipedia in all languages). Creates base model (e.g., T5, mT5, mBART).
  2. **Finetuning**: Supervised on parallel sentence pairs (source, target). 300M+ pairs. ML objective = next-token prediction; loss = cross-entropy.
- **Bilingual vs Multilingual**:
  | Approach | Pros | Cons |
  | -------- | ---- | ---- |
  | **Bilingual** | Higher quality; easier to debug/improve per-pair | N×(N-1) models; expensive to maintain |
  | **Multilingual** | Single model; transfer learning between languages | May sacrifice quality on low-resource pairs |
- **Evaluation**:
  - Offline: **BLEU** (precision), **METEOR** (semantic matching), **ROUGE** (recall)
  - Online: **User feedback** (thumbs up/down), **Suggest edit rate**, **Engagement** (return usage)

**4. Bottlenecks & Trade-offs (5–10 min)**

- **Bilingual vs Multilingual**: For 4 languages, 4×3 = 12 bilingual models is manageable. For 130 languages, multilingual is required (with specialized models for high-traffic pairs).
- **Language detection accuracy**: Misdetection = wrong model = bad translation. Use high-confidence threshold; fallback to asking user.
- **Named entities**: Without placeholder approach, model may mistranslate proper nouns ("California" → "Californie"). Placeholder approach adds complexity but improves quality.
- **Long sequences**: 1,000 words may exceed context window. Chunk by sentence/paragraph, translate, reassemble.
- **Latency vs quality**: Beam search with beam width 5 is slower but better than greedy. For real-time chat, use beam width 3 or speculative decoding.

🛠️ **Stack snapshot:** Encoder-decoder Transformer (T5, mBART) + SentencePiece tokenization + beam search + language detector (encoder-only) + named entity placeholder system + BLEU/METEOR eval + user feedback loop.

**Base Models to Consider:**
- **Google T5/mT5**: Text-to-text framework; multilingual
- **Meta mBART/NLLB (No Language Left Behind)**: Specialized for translation; 200+ languages
- **Vertex AI Translation API**: Managed service (if not building from scratch)

---

### Example 6: Personal Assistant Chatbot (like ChatGPT)

_General-purpose conversational AI. Three-stage training (Pretraining → SFT → RLHF). Key challenges: safety, multi-turn context, and alignment to human preferences._

**1. Clarify Requirements (5–10 min)**

| Dimension | What to pin down | Why it matters |
| --------- | ---------------- | -------------- |
| **Context window** | 4K, 8K, 32K, or 128K tokens | Affects memory, cost, multi-turn capability |
| **Tasks** | General Q&A, coding, creative writing, reasoning | Determines evaluation benchmarks |
| **Modalities** | Text-only or multimodal (images, audio) | Architecture complexity |
| **Safety** | Must avoid harmful, biased, or false content | Requires RLHF + guardrails |
| **Latency** | P50 < 2s time-to-first-token; streaming for long responses | UX expectation |
| **Personalization** | Per-user memory or stateless | Privacy vs UX trade-off |
| **Languages** | English-first or multilingual | Data and eval requirements |

📊 **Rough estimation (chatbot service)**

- **Volume:** 100M users × 10 messages/day = **1B messages/day** = ~12K QPS.
- **Token budget:** Avg 500 input (context + prompt) + 200 output = 700 tokens/request. At 1B requests: **~700B tokens/day**.
- **Cost:** At ≈$0.50/1M input, ≈$1.50/1M output: 500B × 0.50 + 200B × 1.50 = **$550K/day**. Need aggressive caching, routing, and quantization.
- **Latency budget (2s TTFT):** Safety filter < 100 ms, prompt enhancement < 50 ms, LLM inference TTFT < 1.8s.

**2. High-Level Architecture (10–15 min)**

```
User Message → Safety Filter → Prompt Enhancer → Session Manager (add history)
                                                           ↓
                                              Response Generator (LLM + Top-p)
                                                           ↓
                                              Response Safety Evaluator → Output (stream)
                                                           ↓
                                              Rejection Response (if unsafe)
```

**Components:**

1. **Safety Filter**: Block harmful prompts before LLM (Model Armor, Bedrock Guardrails)
2. **Prompt Enhancer**: Fix typos, expand abbreviations, add system prompt
3. **Session Manager**: Maintain conversation history within context window
4. **Response Generator**: LLM + top-p sampling (temperature 0.7 for balance)
5. **Response Safety Evaluator**: Check output for toxicity, PII, harmful content
6. **Rejection Response Generator**: Polite refusal with explanation

**3. Deep Dive (15–20 min)**

- **Model architecture**: Decoder-only Transformer; RoPE positional encoding (for long context); Grouped Query Attention (GQA) for efficiency; 7B–70B params depending on quality/cost trade-off.
- **Three-stage training**:
  1. **Pretraining**: Trillions of tokens (Common Crawl, C4, books, code, Wikipedia). ML objective = next-token prediction.
  2. **SFT**: 10K–100K (prompt, response) pairs (Alpaca, FLAN, Dolly). Same objective, but on instruction format.
  3. **RLHF**: Train reward model on human preference rankings → optimize SFT model with PPO to maximize reward.
- **Sampling**: Top-p (nucleus) sampling with temperature 0.7. Repetition penalty to avoid loops.
- **Session management**: Concatenate previous turns into context. If exceeds window, summarize older turns or truncate.
- **Evaluation**:
  - Task-specific: **MMLU** (multitask), **HumanEval** (code), **GSM8K** (math), **TruthfulQA** (factuality)
  - Safety: **RealToxicityPrompts**, **CrowS-Pairs** (bias), **AdvBench** (adversarial)
  - Online: **User feedback** (thumbs up/down), **LMSYS Arena** ranking, **engagement metrics**

**4. Bottlenecks & Trade-offs (5–10 min)**

- **Model size vs cost**: 7B model is fast/cheap but less capable; 70B is smarter but 10× more expensive. Use routing: small model for simple queries, large model for complex.
- **Context length vs memory**: 128K context = huge KV cache. Consider chunking, summarization, or RAG for knowledge-intensive tasks.
- **RLHF quality vs diversity**: Too much RLHF → "sycophantic" model that always agrees. Balance with diversity in reward model training.
- **Streaming vs batching**: Users expect streaming (word-by-word). But batching improves throughput. Stream for interactive; batch for API/background.
- **Safety vs helpfulness**: Overly cautious model refuses legitimate requests. Tune guardrails to balance.
- **Personalization vs privacy**: Per-user memory improves UX but raises privacy concerns. Consider opt-in, on-device storage, or session-only memory.

🛠️ **Stack snapshot:** Decoder-only Transformer (LLaMA, Gemini, GPT) + RoPE + three-stage training (Pretrain/SFT/RLHF) + top-p sampling + session management + safety filters (Model Armor) + MMLU/HumanEval/TruthfulQA eval + LMSYS Arena for online eval.

**Models to Consider:**
- **OpenAI GPT-4/GPT-4o**: State-of-the-art; API-only
- **Google Gemini 1.5**: Long context (1M tokens); API or Vertex AI
- **Meta LLaMA 3**: Open-source; 8B–405B params
- **Anthropic Claude 3**: Strong safety; API-only
- **Mistral/Mixtral**: Open-source; MoE architecture

---

### Example 7: Image Captioning System

_Generate descriptive text for images. Multimodal: Image Encoder + Text Decoder with cross-attention. Applications: asset naming, alt-text, content moderation, recommendation cold-start._

**1. Clarify Requirements (5–10 min)**

| Dimension | What to pin down | Why it matters |
| --------- | ---------------- | -------------- |
| **Caption style** | Short (2–5 words) for file naming vs detailed (1–2 sentences) for alt-text | Affects training data and model output length |
| **Image types** | General everyday images vs domain-specific (medical, technical) | Domain-specific needs specialized training data |
| **Latency** | 1–2 seconds acceptable; not real-time | Can use larger encoder for quality |
| **Minimum resolution** | 256×256 pixels minimum | Low-res images → unclear captions; reject or warn |
| **Languages** | English-only or multilingual | Data and model requirements |
| **Safety** | No biased or offensive captions | Post-processing filter required |
| **Ambiguous images** | Skip suggestion if confidence low | Avoid bad suggestions; use confidence threshold |

📊 **Rough estimation (image captioning)**

- **Volume:** 10M image uploads/day; 50% trigger captioning = 5M captions/day = ~60 QPS.
- **Compute per image:** Encoder ~100ms, decoder ~200ms (beam search) = ~300ms/image.
- **Cost:** Self-hosted: ~$0.001–0.005/image. API (Gemini Vision): ~$0.01–0.05/image depending on size.
- **Latency budget (1.5s):** Image preprocessing < 100ms, encoding < 300ms, decoding < 800ms, post-processing < 300ms.

**2. High-Level Architecture (10–15 min)**

```
Input Image → Preprocessing (resize, normalize) → Image Encoder (ViT/CLIP) → Sequence of Embeddings
                                                                                      ↓
                                                          Text Decoder (GPT-style) + Cross-Attention
                                                                                      ↓
                                                          Beam Search → Confidence Check → Post-Processing → Caption
```

**Components:**

1. **Image Preprocessing**: Resize to 256×256 (or 224×224), center-crop to preserve aspect ratio, normalize pixel values
2. **Image Encoder**: ViT or CLIP encoder → sequence of patch embeddings (e.g., 16×16 patches → 256 embeddings)
3. **Text Decoder**: Decoder-only Transformer with cross-attention to image embeddings
4. **Beam Search**: Deterministic decoding (beam width 3–5) for consistent, high-quality captions
5. **Confidence Check**: If cumulative probability < threshold, skip suggestion
6. **Post-Processing**: Filter offensive words; replace biased terms with neutral alternatives

**3. Deep Dive (15–20 min)**

- **Image Encoder**: ViT-B/16 (16×16 patches); output = sequence of 196 embeddings (for 224×224 image). Pretrained on ImageNet or CLIP.
- **Text Decoder**: GPT-2 or LLaMA (frozen or finetuned). Cross-attention layers attend to image embeddings.
- **Training (Two-Stage)**:
  1. **Pretrain encoder** (CLIP contrastive learning or ViT on ImageNet)
  2. **Pretrain decoder** (GPT on web text)
  3. **Finetune together** on image-caption pairs (400M pairs from LAION). ML objective = next-token prediction; loss = cross-entropy.
- **Data Preparation**:
  - **Caption**: Remove non-English, deduplicate (CLIP similarity), filter low-relevance (CLIP score < 0.25), summarize long captions (LLaMA), normalize, tokenize (BPE)
  - **Image**: Remove low-res (<256×256), remove low-quality (LAION Aesthetics), resize + center-crop, normalize pixels
- **Sampling**: Beam search (not top-p) for consistency and coherence. Stop at `<EOS>` or max 20 tokens.
- **Evaluation**:
  - Offline: **CIDEr** (consensus across references), **BLEU-4**, **METEOR**
  - Online: **Engagement** (click-through on suggested names), **User edit rate** (how often users modify caption)

**4. Bottlenecks & Trade-offs (5–10 min)**

- **Encoder output format**: Single token = fast but generic; sequence = detailed but more memory. Use sequence for quality.
- **Caption length vs detail**: Short captions (2–5 words) for file naming; longer for alt-text. Train on appropriate data or add length control.
- **Confidence threshold**: Too high = skip too many; too low = bad suggestions. Tune on validation set.
- **Domain adaptation**: General model may fail on domain-specific images (medical, technical). Finetune on domain data if needed.
- **Offensive content**: Model may generate biased or offensive captions. Post-processing filter + blocklist essential.
- **Beam search vs creativity**: Beam search gives consistent, safe captions. For creative applications, consider top-p sampling.

🛠️ **Stack snapshot:** ViT/CLIP encoder + GPT-2/LLaMA decoder + cross-attention + beam search + CLIP filtering for data + CIDEr/BLEU eval + post-processing filter.

**Models to Consider:**
- **BLIP-2**: Frozen image encoder + LLM + Q-Former bridge
- **BLIP-3 (xGen-MM)**: Latest multimodal family; open-source
- **LLaVA**: ViT + LLaMA; open-source; good for VQA too
- **Gemini Vision API**: Managed service; easy integration
- **Vertex AI Vision**: Image captioning as managed service

---

### Example 8: Document Q&A System (like ChatPDF)

_Answer employee questions using internal company documents (Wiki, PDFs, forums). This is the canonical RAG example: retrieve relevant chunks from a large corpus, then generate a grounded answer with citations._

**1. Clarify Requirements (5–10 min)**

| Dimension | What to pin down | Why it matters |
| --------- | ---------------- | -------------- |
| **Document types** | PDFs (text, tables, diagrams), Wiki pages, forum posts | Determines parsing strategy (rule-based vs AI-based) |
| **Corpus size** | 5M pages; 20% annual growth | ANN algorithm choice; index size planning |
| **Formats** | Single-column, double-column, mixed | AI-based parsing needed for varied layouts |
| **Languages** | English-only or multilingual | Embedding model and LLM selection |
| **Latency** | 2–5 seconds acceptable | Can afford reranking and larger models |
| **Citations** | Must include document references | Need to track chunk provenance |
| **Follow-ups** | Support multi-turn conversations | Session management in architecture |

📊 **Rough estimation (Document Q&A)**

- **Indexing scale:** 5M pages × 1500 chars/page ÷ (500 chunk - 200 overlap) ≈ 25M text chunks + 15M image chunks = **40M total chunks**.
- **Volume:** 50K queries/day = ~0.6 QPS (low traffic); peak 10 QPS.
- **Retrieval:** ANN search over 40M vectors at 768 dimensions. HNSW latency: ~5–20ms.
- **Cost:** Embedding (one-time): 40M × 0.0001/1K tokens ≈ $4K. LLM per query: ~2K context × $0.001/1K tokens = $0.002/query → **$100/day** for LLM.

**2. High-Level Architecture (10–15 min)**

**Indexing Pipeline:**
```
PDF/Wiki → Document Parser (AI-based) → Chunks (text, tables, images)
                                              ↓
                                        Embedding Model (CLIP for text+images)
                                              ↓
                                        Vector Database (FAISS/Pinecone)
```

**Query Pipeline:**
```
User Query → Safety Filter → Query Expansion (optional)
                                    ↓
                              Text Encoder → ANN Search (HNSW) → Top-20 chunks
                                                                      ↓
                                                              Cross-Encoder Rerank → Top-5
                                                                      ↓
                                                              Prompt Engineering + LLM
                                                                      ↓
                                                              Response with Citations
```

**Components:**
1. **Document Parser**: Layout-Parser or Google Document AI for varied PDF layouts
2. **Chunking**: Recursive splitting (500 tokens, 200 overlap); preserve section boundaries
3. **Embedding**: CLIP (shared text-image space) or text-embedding-004
4. **Vector DB**: FAISS (self-hosted) or Vertex AI Vector Search (managed); use HNSW for scale
5. **Reranker**: Cross-encoder (e.g., ms-marco-MiniLM) on top-20 → top-5
6. **LLM**: Gemini or GPT-4 with retrieved context + citation instructions

**3. Deep Dive (15–20 min)**

- **Document Parsing**: AI-based (Layout-Parser) for mixed layouts. Extract text blocks with coordinates, tables as markdown, images as captions or CLIP embeddings.
- **Chunking**: RecursiveCharacterTextSplitter with 500 char chunks, 200 overlap. Add metadata: page number, section header, document ID for citations.
- **Indexing**: CLIP text encoder for text chunks; CLIP image encoder for figures. Store in shared embedding space for cross-modal retrieval.
- **ANN**: HNSW (FAISS or Pinecone) for 40M vectors. Build time ~hours; query time ~10ms. IVF as fallback if index size is a concern.
- **Query Expansion**: LLM rewrites query for clarity + generates HyDE hypothetical answer to improve retrieval.
- **Retrieval**: Hybrid (dense + BM25) → RRF merge → cross-encoder rerank. Return top-5 with source metadata.
- **Generation**: Prompt = system instructions + top-5 chunks (with doc IDs) + user query + "cite your sources". Top-p sampling, temperature 0.7.
- **Evaluation**: RAGAS (faithfulness, answer relevancy, context precision/recall). Track citation accuracy manually on sample.

**4. Bottlenecks & Trade-offs (5–10 min)**

- **Parsing accuracy vs speed**: AI-based parsing is slower but handles varied layouts. Batch parse offline; update index incrementally.
- **Chunk size trade-off**: Smaller = more precise retrieval, less context per chunk. Larger = more context, may dilute relevance. 500 tokens + overlap is a good start.
- **Index size vs recall**: HNSW has higher memory but best recall. IVF (clustering) uses less memory but requires tuning.
- **Cross-modal retrieval**: CLIP aligns text and images, but image captions may be more reliable for search. Test both approaches.
- **Citation accuracy**: LLM may cite wrong source or fabricate. Use structured output (source_id) and validate against retrieved chunks.
- **Freshness**: Documents change; need incremental re-indexing pipeline. Delta updates or full rebuild on schedule.

🛠️ **Stack snapshot:** Layout-Parser/Document AI + CLIP/text-embedding-004 + FAISS/Pinecone (HNSW) + LangChain RecursiveTextSplitter + cross-encoder rerank + Gemini/GPT-4 + RAGAS eval + citation validation.

**RAFT consideration:** If retrieval is noisy (many similar docs), consider RAFT finetuning—train LLM on (query, mixed golden+distractor context, answer) to ignore irrelevant chunks.

---

### Example 9: Realistic Face Generation System (like StyleGAN)

_Generate diverse, high-quality synthetic faces for entertainment, marketing, or training data. GAN-based approach with optional attribute control (age, expression, hairstyle)._

**1. Clarify Requirements (5–10 min)**

| Dimension | What to pin down | Why it matters |
| --------- | ---------------- | -------------- |
| **Output resolution** | 1024×1024 target | Higher resolution = more compute, more data needed |
| **Diversity** | Balanced ethnicity, age, gender | Avoid bias; need diverse training data |
| **Attribute control** | Optional: edit age, hair, expression | Requires StyleGAN-style architecture |
| **Latency** | < 1 second per image | Single forward pass through generator |
| **Training data** | 70K diverse face images (licensed) | Quality and diversity determine output quality |
| **Safety** | No deepfakes of real people | Watermarking; usage policies |

📊 **Rough estimation (face generation)**

- **Training data:** 70K images × 3 channels × 1024×1024 = ~200GB raw. Augmented 5×: 1TB.
- **Training compute:** StyleGAN2 on 70K images: ~1–2 weeks on 8×V100 GPUs.
- **Inference:** Single forward pass: ~20–50ms on GPU. Can generate ~20–50 faces/second.
- **Serving cost:** ~$0.001–0.01 per image depending on GPU utilization.

**2. High-Level Architecture (10–15 min)**

```
User Request → Face Generator Service → [Sample Noise] → Generator (StyleGAN) → Output Image
                     ↓ (optional)
              Attribute Control → Modify Latent Vector → Generator
```

**Training Pipeline:**
```
Training Data → Preprocess (resize, normalize, augment) → GAN Training Loop
                                                              ↓
                                        Generator ←→ Discriminator (adversarial)
                                                              ↓
                                        Evaluation (FID, IS) → Deploy if improved
```

**Components:**
1. **Generator**: Upsampling blocks (ConvTranspose2D → BatchNorm → ReLU) × N → Tanh output
2. **Discriminator**: Downsampling blocks (Conv2D → BatchNorm → LeakyReLU) × N → Sigmoid
3. **Latent Space**: 512-dim noise vector sampled from N(0,1)
4. **StyleGAN extensions**: Style mapping network for attribute control

**3. Deep Dive (15–20 min)**

- **Architecture**: StyleGAN2 (or StyleGAN3) generator. Mapping network transforms noise to style vectors. Style vectors injected at each resolution level (4×4, 8×8, ..., 1024×1024).
- **Training**:
  - Adversarial training: alternate discriminator (k steps) and generator (1 step)
  - Loss: Non-saturating GAN loss (modified minimax) or Wasserstein loss with gradient penalty (WGAN-GP)
  - Regularization: R1 regularization, path length regularization (StyleGAN2)
- **Normalization**: BatchNorm in generator; spectral normalization in discriminator for stability.
- **Sampling**: Random sampling from N(0,1) for diversity; truncated sampling (ψ=0.7) for higher quality, less diversity.
- **Attribute control** (if required): Find attribute directions in latent space (e.g., "age vector"). Add/subtract to modify attributes while preserving identity.
- **Evaluation**:
  - Offline: **FID** (lower = closer to real distribution), **Inception Score** (higher = quality + diversity)
  - Online: Human evaluation (pairwise comparison), user feedback, latency monitoring

**4. Bottlenecks & Trade-offs (5–10 min)**

- **Quality vs diversity**: Truncated sampling increases quality but reduces diversity. Adjust truncation parameter ψ based on use case.
- **Training stability**: GANs are notoriously unstable. Use WGAN-GP or progressive growing; monitor discriminator/generator loss balance.
- **Mode collapse**: Generator may produce limited variety. Mitigations: minibatch discrimination, unrolled GAN, Wasserstein loss.
- **Resolution vs speed**: 1024×1024 is slower than 256×256. For real-time, consider lower resolution or distilled models.
- **Diversity vs bias**: Training data must be balanced. Use attribute classifiers to measure distribution; resample if biased.
- **Deepfake concerns**: Generated faces may be misused. Add watermarks; track usage; implement content policies.

🛠️ **Stack snapshot:** StyleGAN2/StyleGAN3 (NVIDIA) + PyTorch/TensorFlow + 8×V100/A100 GPUs + FID/IS evaluation + human eval pairwise comparison + watermarking.

**Models/Resources:**
- **StyleGAN2-ADA**: Adaptive augmentation for limited data
- **StyleGAN3**: Alias-free, better video generation
- **NVIDIA pretrained models**: thispersondoesnotexist.com uses StyleGAN

---

### Example 10: Text-to-Image Generation System (like DALL-E, Stable Diffusion)

_Generate images from text prompts. Diffusion-based approach with text conditioning, safety filtering, and super-resolution for high-quality output._

**1. Clarify Requirements (5–10 min)**

| Dimension | What to pin down | Why it matters |
| --------- | ---------------- | -------------- |
| **Output resolution** | 1024×1024 target | Train at lower res + super-resolution cascade |
| **Prompt length** | Max 128 words | Text encoder context limit |
| **Image types** | Landscapes, portraits, abstract, realistic | Diverse training data needed |
| **Latency** | < 10 seconds per image | Diffusion steps + super-resolution |
| **Training data** | 500M image-caption pairs | Quality and diversity; filtering critical |
| **Languages** | English initially; extensible | Text encoder choice |
| **Safety** | No violence, NSFW, harmful content | Prompt filter + output filter |
| **Bias** | Fair across age, race, gender | Balanced training data; evaluation |

📊 **Rough estimation (text-to-image)**

- **Training data:** 500M image-caption pairs after filtering. LAION-style dataset.
- **Training compute:** Large diffusion model (3B+ params): ~months on 256+ GPUs (A100).
- **Inference:** 20–50 DDIM steps × ~50ms/step = 1–2.5s base. Super-resolution adds 0.5–1s. Total: ~2–4s on A100.
- **Serving cost:** ~$0.01–0.05 per image depending on model size and hardware.

**2. High-Level Architecture (10–15 min)**

**Training Pipeline:**
```
Raw Data (images + captions) → Filtering (NSFW, quality, dedup) → Caption Enhancement (CLIP, BLIP-3)
                                                                           ↓
                                           Text Encoder (T5/CLIP) → Pre-compute + Cache Embeddings
                                                                           ↓
                                           Diffusion Training (U-Net or DiT) + Super-Resolution Training
```

**Inference Pipeline:**
```
User Prompt → Prompt Safety → Prompt Enhancement (LLM) → Text Encoder (T5)
                                                              ↓
                                   [Noise] + Text Embeddings → Diffusion Model (DDIM + CFG)
                                                              ↓
                                              64×64 → Super-Res #1 → 256×256 → Super-Res #2 → 1024×1024
                                                              ↓
                                                        Harm Detection → Final Image
```

**Components:**
1. **Text Encoder**: T5 or CLIP; converts prompt to embeddings
2. **Diffusion Model**: U-Net or DiT; predicts noise conditioned on text
3. **Super-Resolution**: Cascade of 2–3 models to upscale
4. **Safety Filters**: Prompt classifier + output image classifier

**3. Deep Dive (15–20 min)**

- **Data preparation**:
  - Images: Remove small (<64×64), deduplicate, filter NSFW/low-aesthetic (LAION Aesthetics), resize + normalize
  - Captions: Handle missing (generate with BLIP-3), filter low CLIP similarity, enhance short captions
- **Architecture**: U-Net with cross-attention to text embeddings. Downsampling: Conv2D → BatchNorm → ReLU → MaxPool → Cross-Attention. Upsampling: ConvTranspose2D → BatchNorm → ReLU → Cross-Attention.
- **Training**: Forward process adds noise; model predicts noise. Loss = MSE(true noise, predicted noise). Timestep embedding tells model noise level.
- **Sampling**: DDIM (20–50 steps instead of 1000). CFG with guidance scale w=7–15 for text adherence.
- **Super-resolution**: Train separate models. Base → 256×256 → 1024×1024. Each is a smaller diffusion model conditioned on low-res input.
- **Evaluation**:
  - Quality: **FID** (lower = better)
  - Alignment: **CLIPScore** (higher = better)
  - Diversity: **IS** (higher = better)
  - Benchmark: **DrawBench** (curated prompts for comprehensive testing)
  - Human eval: Pairwise comparison for photorealism and text alignment

**4. Bottlenecks & Trade-offs (5–10 min)**

- **Quality vs speed**: More DDIM steps = higher quality, slower. 20–50 is typical trade-off.
- **CFG guidance scale**: Higher w = better text adherence, less diversity. w=7–15 typical.
- **Resolution vs cost**: Training at 1024² is expensive. Latent diffusion (Stable Diffusion) trains at 64×64 latent, decodes to 512×512—much cheaper.
- **Training data quality**: Garbage in, garbage out. Heavy filtering (CLIP similarity, aesthetics) is critical.
- **Safety vs usefulness**: Aggressive prompt filtering may block legitimate requests. Tune thresholds.
- **Prompt enhancement**: Expands "a dog" → detailed description. Improves quality but adds latency.
- **Latent diffusion trade-off**: Much faster training/inference, but VAE decoder may lose fine details. Pixel-space diffusion (Imagen) is higher quality but slower.

🛠️ **Stack snapshot:** T5/CLIP text encoder + U-Net/DiT diffusion + DDIM sampler + CFG + super-resolution cascade + CLIP filtering (data) + FID/CLIPScore/DrawBench (eval) + prompt safety classifier + output harm detector.

**Models to Consider:**
- **Stable Diffusion**: Open-source; latent diffusion; 512×512 → 1024×1024
- **DALL-E 3**: OpenAI; prompt understanding; API-only
- **Imagen 3**: Google; pixel-space diffusion; high quality
- **Midjourney**: Closed-source; artistic focus
- **Adobe Firefly**: Commercial; trained on licensed data

---

### Example 11: Text-to-Video Generation System (like Sora, Movie Gen)

_Generate 5-second 720p videos from text prompts. Latent diffusion with DiT, temporal layers for consistency, and super-resolution for quality._

**1. Clarify Requirements (5–10 min)**

| Dimension | What to pin down | Why it matters |
| --------- | ---------------- | -------------- |
| **Video length** | 5 seconds target | Longer = exponentially more compute |
| **Resolution** | 720p (1280×720) | Train at lower res + super-resolution |
| **Frame rate** | 24 FPS → 120 frames | Temporal super-resolution can help |
| **Latency** | Minutes acceptable initially | Optimization for speed comes later |
| **Training data** | 100M video-caption pairs | Quality filtering critical |
| **Pretrained model** | Have text-to-image model | Can leverage for video training |
| **Audio** | Silent videos initially | Audio is separate problem |
| **Safety** | No harmful content | Prompt + output filtering |

📊 **Rough estimation (text-to-video)**

- **Training data:** 100M videos; after filtering + latent precomputation, ~200TB storage.
- **Training compute:** DiT on 100M videos: ~months on 6000+ H100 GPUs (Sora-scale).
- **Compression:** 120 frames × 1280×720 = 110M pixels → with 8×8 compression: 216K (512× smaller).
- **Inference:** 50 DDIM steps × ~500ms/step = ~25s for latent. + super-resolution: ~2–5 minutes total.
- **Serving cost:** ~$0.10–1.00 per video depending on duration/resolution.

**2. High-Level Architecture (10–15 min)**

**Training Pipeline:**
```
Videos → Filter (quality, NSFW, dedup) → Standardize (5s clips, 24fps, 720p)
                                              ↓
                        VAE Encoder → Precompute Latents → Storage (200TB)
                                              ↓
                        Captions → Re-caption (LLaVA) → Text Encoder → Cache Embeddings
                                              ↓
                        DiT Training (latent space) + Image-Video Joint Training
```

**Inference Pipeline:**
```
Prompt → Safety → Enhancement → Text Encoder (T5)
                                       ↓
             Noise → DiT (LDM) + CFG → Denoised Latent
                                       ↓
             VAE Decoder → Low-res Video (160×90 @ 8fps)
                                       ↓
             Spatial SR (720p) → Temporal SR (24fps) → Harm Detection → Final Video
```

**Components:**
1. **VAE (Compression Network)**: Compress 512× for efficient training
2. **DiT with Temporal Layers**: Temporal attention + temporal convolution
3. **Text Encoder**: T5 for text embeddings
4. **Spatial Super-Resolution**: Upscale 160×90 → 1280×720
5. **Temporal Super-Resolution**: Interpolate 8fps → 24fps

**3. Deep Dive (15–20 min)**

- **Compression network (VAE)**: 8× temporal (120→15 frames) + 8×8 spatial (1280×720 → 160×90). Train separately; freeze during diffusion training.
- **DiT architecture**:
  - 3D patchify (spatial + temporal patches)
  - RoPE for 3D positional encoding (x, y, t)
  - Temporal attention: each patch attends across frames
  - Temporal convolution: 3D conv for local motion patterns
- **Training**:
  - Joint training on images (1-frame videos) + videos to leverage large image datasets
  - MSE loss on predicted vs true noise
  - Precompute and cache all latents + embeddings before training
- **Sampling**: DDIM (50 steps) + CFG (w=7–15)
- **Super-resolution**:
  - Spatial: Separate diffusion model conditioned on low-res input
  - Temporal: Frame interpolation model (generate intermediate frames)
- **Evaluation**:
  - Quality: **FID** (per-frame average)
  - Temporal consistency: **FVD** (Fréchet Video Distance using I3D features)
  - Alignment: **CLIP similarity** (per-frame average)
  - Benchmarks: VBench, Movie Gen Bench

**4. Bottlenecks & Trade-offs (5–10 min)**

- **Latent vs pixel diffusion**: Latent is 512× cheaper but VAE decoder may lose fine details. Pixel-space (Imagen Video) is higher quality but much slower.
- **Image vs video training data**: Images are abundant; videos are scarce. Joint training or pretrain-finetune helps.
- **Temporal consistency vs quality**: More temporal attention = better consistency, more compute.
- **Resolution vs speed**: Generate at 360p + SR is faster than native 720p. Trade-off quality.
- **Video length**: 5s is manageable; 60s requires hierarchical generation (plan → clips → stitch).
- **CFG guidance scale**: Higher = better prompt adherence, less diversity. Tune per use case.
- **Super-resolution cascade**: Each stage adds latency but enables higher final quality with cheaper base model.

🛠️ **Stack snapshot:** VAE (compression) + DiT (temporal attention/conv, 3D patches, RoPE) + T5 encoder + DDIM + CFG + spatial/temporal SR + FVD/FID/CLIP eval + distributed training (6000+ GPUs).

**Models to Consider:**
- **Sora** (OpenAI): DiT; variable duration/resolution; "world simulator"
- **Movie Gen** (Meta): DiT + LDM; 16s at 768p; joint image-video training
- **Stable Video Diffusion**: U-Net based; image-to-video
- **Runway Gen-3**: Commercial; fast; video-to-video
- **Veo** (Google): High quality; integrated with Vertex AI

---

### Cross-example takeaways

| Concern                                    | Tools to reach for                                                                                                  |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| **Orchestration (RAG, agents, pipelines)** | LangChain, LlamaIndex                                                                                               |
| **Managed RAG / embeddings**               | Vertex AI RAG Engine, **Vertex AI Search** (website/commerce/internal KB), Bedrock Knowledge Bases                  |
| **Internal knowledge workers**             | **Gemini Enterprise** (agents + unified search), **NotebookLM Enterprise** (document-focused Q&A, summarize, audio) |
| **LLM hosting**                            | Vertex AI (Codey, Gemini), Bedrock (Claude, CodeWhisperer, etc.), or vLLM for self-hosted                           |
| **Evaluation (reference-free)**            | RAGAS (batch), LangSmith (datasets + humans), Phoenix (traces + evals)                                              |
| **Guardrails**                             | Model Armor (Google), Bedrock Guardrails (AWS), Guardrails AI / NeMo (open source)                                  |

The full **45-min Interview Framework** (Clarify → High-Level Architecture → Deep Dive → Bottlenecks & Trade-offs) is in [Quick Reference: Interview Framework](#interview-framework-45-min-structure). _Note:_ Cost numbers in the examples use illustrative per-token rates; real pricing varies by provider and model—use them to practice estimation, not as exact quotes.

## G.1 Strategy and Planning (for integration and impact)

_Gen AI evolves quickly; no one stays an "expert" without adapting. This section summarizes how to plan for integration, measure impact, and stay ahead—useful for leadership discussions and certification._

**Plan for generative AI integration:** (1) **Establish a clear vision** — align with business goals. (2) **Prioritize high-impact use cases** — start where value is measurable. (3) **Invest in capabilities** — tools, data, skills. (4) **Drive organizational change** — adoption, workflows. (5) **Measure and demonstrate value** — see below. (6) **Champion responsible AI** — safety, fairness, compliance (E.10).

**Define key metrics:** Choose metrics that align with business objectives. Common targets: **ROI** (financial benefits of gen AI initiatives vs. costs), **revenue** (direct impact on sales/profits), **cost reduction**, **efficiency** (throughput, time-to-resolution), **customer experience**, **security**. If ROI matters, compare benefits to costs; if revenue matters, measure direct impact on sales and profits.

**Plan for change:** Even when solutions work, be prepared to adapt. Technology and models change rapidly; customers and employees expect you to keep up. **Tips:** Regularly review and refine strategy based on latest advancements and org needs; stay informed (industry news, research, expert opinions); engage with the gen AI community (conferences, workshops, forums); invest in training and upskilling; attract and retain talent with a culture of learning and innovation.

> [!TIP]
> 💡 **Aha:** Successfully leading with gen AI means **continuous learning and adaptation**. Set a clear strategic vision, stay flexible, refine strategy with data-driven insights, and foster a culture of learning. This guide gives you the technical foundation; strategy and planning help you apply it at scale.

---

## G.2 Quick Reference

Use this section to **prove technical ability** and to **design GenAI systems that ship to customers at scale**—in interviews (system design + hypotheticals) and in practice (Scope → Design → Deploy → Communicate).

### What FAANG Interviewers Evaluate

| Dimension                   | What They Test                                             |
| --------------------------- | ---------------------------------------------------------- |
| **LLM Awareness**           | Token limits, context windows, model types, pricing models |
| **Architectural Reasoning** | How retrieval, prompt logic, post-processing connect       |
| **Cost-Latency Tradeoffs**  | Balancing inference cost, response latency, quality        |
| **Safety & Governance**     | Reliable outputs, guardrails, compliance                   |
| **Observability**           | Handling non-deterministic outputs, failure modes          |

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

| Decision               | Option A                   | Option B                        |
| ---------------------- | -------------------------- | ------------------------------- |
| RAG vs Fine-tuning     | Fresh data, per-query cost | Behavioral change, upfront cost |
| Large vs Small Model   | Higher quality             | Lower cost, faster              |
| Dense vs Hybrid Search | Semantic matching          | + Keyword precision             |
| Single vs Multi-Agent  | Simpler, faster            | More capable, modular           |
| Sync vs Async Eval     | Immediate                  | Cost-effective                  |

### Interview Talking Points by Stage

Use these as prompts during each stage of a GenAI system design interview.

**Clarifying Requirements:**
- What is the business objective? (e.g., customer support, content creation, code assistance)
- What are the system features that affect ML design? (e.g., multi-language, feedback loops)
- What data is available? How large? Labeled or unlabeled?
- What are the constraints? (cloud vs on-device, compute budget, compliance)
- What is the expected scale? (users, requests/day, growth)
- What are the latency and quality requirements? (real-time vs async, quality vs speed trade-off)

**Framing as ML Task:**
- What are the system's inputs and outputs? (text → text, text → image, etc.)
- Which modalities? (text, image, audio, video)
- Single model or multiple specialized models?
- Which algorithm? (diffusion, autoregressive, VAE, GAN) — and why?

**Data Preparation:**
- Data sources and diversity?
- Data sensitivity and anonymization needs?
- Bias detection and mitigation?
- Data quality filtering (noise, duplicates, inappropriate content)?
- Preprocessing for model consumption (tokenization, embeddings)?

**Model Development:**
- Architecture options and trade-offs? (e.g., U-Net vs DiT for diffusion)
- Training methodology? (pretraining → finetuning → alignment)
- Training data for each stage?
- Loss function(s) and ML objective?
- Training challenges and mitigations? (stability, memory, compute)
- Efficiency techniques? (gradient checkpointing, mixed precision, distributed training)
- Sampling methods? (greedy, beam search, top-k, top-p) — pros/cons?

**Evaluation:**
- Offline metrics? (perplexity, BLEU, FID, RAGAS faithfulness, etc.)
- Online metrics? (CTR, conversion, latency, engagement, retention)
- Bias and fairness evaluation?
- Robustness and adversarial testing?
- Human evaluation methods? (A/B tests, expert reviews)

**Overall System Design:**
- What are all the system components? (model, preprocessing, filtering, postprocessing, upscaling)
- Safety mechanisms? (NSFW filters, guardrails, Model Armor)
- User feedback and continuous learning loops?
- Scalability? (load balancing, distributed inference, model parallelism)
- Security and privacy? (PII handling, adversarial defense, data leakage prevention)

### Role Related Knowledge (RRK) interview — structure and prep

_Some roles use a **Role Related Knowledge** round that mixes GenAI system design with hypotheticals and consultative skills. Use this as a checklist; confirm exact format with your recruiter._

**Time split (example):**

| Segment                      | Duration | Focus                                                                                                            |
| ---------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------- |
| **System design (GenAI)**    | ~30 min  | Design a system (RAG, agent, etc.) — use the [Interview Framework](#interview-framework-45-min-structure) above. |
| **Hypothetical questions**   | ~15 min  | 4–6 scenario-based questions to assess technical ability + **consultative skills**.                              |
| **Questions to interviewer** | ~15 min  | Your questions about Google, the team, ways of working, other perspectives.                                      |

**Goal of hypotheticals:** Assess **technical ability** and **application development with LLM + consultative skills** — e.g. advising clients, defining scope, leading from conflict, setting strategy, problem troubleshooting, developing partnerships through engagement, looping in stakeholders.

**Answer structure (STAR + future):** Use a structure similar to **STAR** — **S**cenario, **T**ask, **A**ction, **R**esult — and add **future thinking** (what you’d do next, how you’d iterate, risks to watch).

**Adapt to audience:**

- **Non-technical (CxOs):** High-level value, business impact, risk, timeline, cost. Avoid jargon; focus on outcomes and trade-offs in business terms.
- **Product / technical:** Details are appropriate — e.g. **open-source LLMs** (pros: control, cost, customization; cons: ops, security, updates) vs managed (Vertex, Bedrock). RAG flow, serverless vs microservice, metrics.

**Example hypothetical themes to prepare:**

| Theme                                         | What to be ready for                                                                                                                                                             |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Showcase GenAI to a customer**              | “How would you present a GenAI solution to a live customer in a professional manner?” Consider **timeline** and **budget** — you can ask the interviewer to clarify constraints. |
| **Loop in stakeholders**                      | How you involve the right people (eng, product, security, legal) and keep them aligned.                                                                                          |
| **Scope → Design → Deploy**                   | Define clear **business requirements**, **relevant metrics**, and **communicate to stakeholders**.                                                                               |
| **POC to production**                         | Your approach: validate with POC (use case, success criteria), then production (reliability, scale, guardrails, observability).                                                  |
| **Design the flow**                           | When to choose **RAG** vs other patterns; **serverless vs microservice** on a specific cloud (e.g. GCP).                                                                         |
| **Lead a public partner on LLM on GCP/Cloud** | How you guide a partner or customer to adopt an LLM product using GCP or cloud offerings — governance, enablement, rollout.                                                      |

### End-to-end solutioning (Scope → Design → Deploy → Communicate)

Use this flow to answer hypotheticals in a structured way. It matches the recruiter themes: define business requirements and metrics, loop in stakeholders, design the flow (RAG, serverless vs microservice), POC→prod, and communicate to different audiences.

| Phase              | What you do                                                                                                                                                                                                                                                                | Recruiter themes                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 🔷 **Scope**       | Define **business requirements**, **success criteria**, **key metrics**. Identify **stakeholders** (eng, product, security, legal) and what they care about. Ask **clarifying questions**: already on GCP vs first-time migration vs hybrid? Timeline, budget, compliance? | Loop in stakeholders; define clear business requirements and relevant metrics |
| 🔷 **Design**      | Choose **architecture** (RAG vs other, agent vs single call); **hosting** (serverless vs microservice on GCP/Cloud); data flow, APIs, guardrails. Tie to E.1–E.10 (this guide).                                                                                              | Design the flow (RAG); build on serverless vs microservice                    |
| 🔷 **Deploy**      | **POC** first: validate use case, success criteria, one clear metric. Then **production**: reliability, scale, guardrails, observability. Call out timeline and budget trade-offs.                                                                                         | POC to prod                                                                   |
| 🔷 **Communicate** | **CxO**: high-level value, risk, cost, timeline; no jargon. **Product/technical**: open-source vs managed LLMs, RAG flow, serverless vs microservice, metrics. **Present to live customer**: professional, confirm timeline/budget and constraints.                        | Explain for non-technical (CxO) vs Product; present solution to customer      |

---

**Example A: Customer wants GenAI for support (end-to-end)**

📋 **Scenario:** A retail customer wants to add an AI chatbot for customer support. They’re considering GCP but haven’t committed.

📌 **Task:** Show how you’d take them from idea to production and present the solution.

**Action (🔷 Scope → 🔷 Design → 🔷 Deploy → 🔷 Communicate):**

- 🔷 **Scope:** I’d ask: Are you already on GCP or first-time migration or hybrid? What’s the timeline and budget? Who owns success — support team, product, eng? I’d define **business requirements**: deflect X% of tier-1 tickets, answer from knowledge base + order lookup, escalate to human when needed. **Metrics**: deflection rate, CSAT, resolution time, cost per conversation. I’d **loop in stakeholders**: eng (architecture), product (scope), security (PII, compliance), legal (terms), support (escalation flow).
- 🔷 **Design:** I’d propose **RAG + agent** (knowledge base + order/ticket tools + escalate) on **GCP**: **Vertex AI** (Gemini) + **Vertex RAG Engine** or **Vertex AI Search** for the knowledge base; **Cloud Run** or **GKE** for the API — **serverless** (Cloud Run) if traffic is spiky and we want low ops, **microservices** (GKE) if we need more control and multiple services. Guardrails: **Model Armor**, input/output filters, PII handling. (Details: F.1 Example 2.)
- 🔷 **Deploy:** **POC** (4–6 weeks): one channel (e.g. web), one knowledge domain, success = deflection rate and CSAT on a pilot. Then **production**: add channels, scale, observability (traces, evals), and runbooks. I’d call out **timeline** (e.g. POC 6 weeks, prod 3 months) and **budget** (LLM cost, infra, labor) so the customer can plan.
- 🔷 **Communicate:** For **CxO**: “We’ll reduce tier-1 load by X%, improve CSAT, with clear cost and timeline; we’ll start with a POC to de-risk.” For **Product**: “RAG over your docs + tools for orders/tickets; we can go serverless on Cloud Run or microservices on GKE depending on scale.” For the **live customer**: present the flow (Scope → Design → Deploy), show a simple diagram, confirm timeline and budget, and ask what they’d want to see in a follow-up.

🎯 **Result:** Clear requirements, metrics, and stakeholder alignment; a concrete design (RAG + agent, GCP, serverless vs microservice); a POC→prod path with timeline and budget; and messaging that fits CxO vs Product vs customer.

🔮 **Future thinking:** I’d plan for **Agent Assist** and **Conversational Insights** when they add live agents; revisit model choice and routing as traffic grows (E.7).

---

**Example B: Public partner adopting LLM on GCP (end-to-end)**

📋 **Scenario:** A public-sector or large partner wants to adopt an LLM-based product using GCP. You’re leading the engagement.

📌 **Task:** Describe your approach from first contact to production and how you’d present it.

**Action (🔷 Scope → 🔷 Design → 🔷 Deploy → 🔷 Communicate):**

- 🔷 **Scope:** I’d ask **clarifying questions**: Are they already on GCP or first-time migration or hybrid? What’s the primary use case (internal knowledge search, citizen-facing Q&A, document processing)? Timeline, budget, and **compliance** (data residency, audit)? I’d define **business requirements** and **metrics** (e.g. time to answer, accuracy, cost per query). I’d **loop in stakeholders**: their IT (infra, security), business owners (use case), procurement (budget); our side: solutions, eng, legal. I’d align on **governance** and **responsible AI** (fairness, safety, explainability) early.
- 🔷 **Design:** I’d recommend **GCP** (Vertex AI, RAG Engine or Vertex AI Search, optional **Gemini Enterprise** for internal knowledge workers). **Serverless** (Cloud Run + Vertex) for fast time-to-value and lower ops; **microservices** if they need strict isolation, custom pipelines, or multi-region. I’d include **guardrails** (Model Armor), **access control** (IAM, VPC), and **audit** (Cloud Audit Logs). For “design the flow”: RAG for domain data, agent only if they need tools (APIs, DBs).
- 🔷 **Deploy:** **POC** (6–8 weeks): one use case, one data source, success = accuracy and user satisfaction. Then **production**: scale, SLAs, monitoring, and handover. I’d be explicit about **timeline** and **budget** (licenses, infra, services) and any dependency on their teams (data, access).
- 🔷 **Communicate:** For **CxO**: “We’ll deliver a pilot in X weeks with clear success criteria; then we scale with your governance and compliance in mind.” For **technical**: “Vertex AI + RAG; serverless vs microservice trade-offs; we’ll document the architecture and runbooks.” For the **live customer**: present the end-to-end plan (Scope → Design → Deploy), one-page diagram, timeline and budget, and next steps; ask about their decision process and any blockers.

🎯 **Result:** Partner has a clear path (scope, design, deploy) with stakeholder alignment, compliance in mind, and messaging for leadership vs technical; you’ve demonstrated consultative skills and structure.

🔮 **Future thinking:** I’d plan for **feedback loops** (evals, user feedback) and **iteration** (model upgrades, new data sources); consider **Gemini Enterprise** or **NotebookLM Enterprise** if they need internal knowledge discovery later.

---

> [!TIP]
> 💡 **Aha:** End-to-end solutioning = **Scope** (requirements, metrics, stakeholders, clarifying questions) → **Design** (RAG/agent, serverless vs microservice, GCP/Cloud) → **Deploy** (POC then prod, timeline, budget) → **Communicate** (CxO vs Product vs live customer). Use Examples A and B as templates; swap in your own scenarios and tie to E.1–F.1.

### How this addresses each question (tangible mapping)

Below, each **recruiter question or theme** is mapped to **where** you answer it in this section and **what you can say** in a concrete way. Use this as a cheat sheet when practicing.

| Question / theme                                                                                  | Where it's addressed                                                                                             | What you can say (tangible)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **"How would you showcase GenAI to a customer?"**                                                 | Example A, **Communicate** (live customer); Example B, **Communicate** (live customer)                           | "I'd present the flow Scope → Design → Deploy, show a one-page diagram, confirm **timeline** and **budget** explicitly, and ask what they'd want to see in a follow-up. I'd ask the interviewer to clarify timeline and budget constraints so I can tailor the proposal."                                                                                                                                                                                                                                                                                                                                                       |
| **"How would you loop in stakeholders?"**                                                         | Example A, **Scope**; Example B, **Scope**                                                                       | "I'd identify **eng** (architecture), **product** (scope), **security** (PII, compliance), **legal** (terms), **support** (escalation). For a partner I'd add their IT, business owners, procurement and our solutions, eng, legal. I'd align on what each cares about **before** design so we don't surprise anyone."                                                                                                                                                                                                                                                                                                          |
| **"Explain for non-technical (CxO) vs Product"**                                                  | Example A, **Communicate**; Example B, **Communicate**; **Adapt to audience** (above)                            | "**CxO:** 'We'll reduce tier-1 load by X%, improve CSAT, with clear cost and timeline; we'll start with a POC to de-risk.' **Product:** 'RAG over your docs plus tools for orders/tickets; we can go serverless on Cloud Run or microservices on GKE depending on scale and control.' I avoid jargon with CxO; I go into RAG, serverless vs microservice, and metrics with Product."                                                                                                                                                                                                                                            |
| **"Open-source LLMs: pros/cons for Product vs high-level for CxO"**                               | **Adapt to audience** (above); Design in both examples                                                           | "**CxO:** high-level value, risk, cost, timeline—no mention of RAG or serverless unless they ask. **Product:** open-source LLMs—**pros:** control, cost, customization; **cons:** ops, security, model updates. I'd recommend **managed** (Vertex, Bedrock) for faster time-to-value and enterprise support unless they have a strong reason to self-host."                                                                                                                                                                                                                                                                     |
| **"Scope, Design, Deploy; define business requirements, metrics, communicate to stakeholders"**   | End-to-end **table**; Example A and B, **all four phases**                                                       | "I follow **Scope → Design → Deploy → Communicate**. In **Scope** I define business requirements (e.g. deflect X% of tier-1 tickets, answer from knowledge base + order lookup, escalate when needed), **metrics** (deflection rate, CSAT, resolution time, cost per conversation), and **loop in stakeholders** and what they care about. In **Communicate** I tailor the message to CxO vs Product vs live customer."                                                                                                                                                                                                         |
| **"How would you lead a public partner's use of LLM on GCP?"**                                    | **Example B**, full flow                                                                                         | "I'd ask **clarifying questions** first: already on GCP or first-time migration or hybrid? Primary use case? Timeline, budget, **compliance** (data residency, audit)? I'd **loop in** their IT, business owners, procurement and our solutions, eng, legal. I'd recommend **GCP** (Vertex AI, RAG Engine or Vertex AI Search, optional Gemini Enterprise), **serverless** for fast time-to-value or **microservices** for isolation. **POC** 6–8 weeks, one use case, then production. I'd **present** the plan with a one-page diagram, timeline, budget, and next steps, and ask about their decision process and blockers." |
| **"What is your approach from POC to production?"**                                               | Example A, **Deploy**; Example B, **Deploy**                                                                     | "**POC first:** one use case, one channel or one data source, **4–8 weeks**, success = **one clear metric** (e.g. deflection rate, CSAT, or accuracy). Then **production:** add channels, scale, guardrails (Model Armor), observability (traces, evals), runbooks. I'd call out **timeline** (e.g. POC 6 weeks, prod 3 months) and **budget** (LLM cost, infra, labor) so the customer can plan."                                                                                                                                                                                                                              |
| **"Design the flow (RAG) or build the application on serverless vs microservice"**                | Example A, **Design**; Example B, **Design**                                                                     | "**Flow:** I'd choose **RAG** when the model lacks domain knowledge or data changes often; **agent** when we need tools (orders, tickets, APIs). On GCP: Vertex AI + RAG Engine or Vertex AI Search. **Hosting:** **Serverless** (Cloud Run) for spiky traffic and low ops; **microservices** (GKE) when we need more control, multiple services, or strict isolation. I'd state the trade-off so the customer can decide."                                                                                                                                                                                                     |
| **"Present your solution to a live customer in a professional manner; consider timeline/budget"** | Example A, **Communicate** (live customer); Example B, **Communicate** (live customer); **Clarifying questions** | "I'd present **Scope → Design → Deploy** in order, show a **simple diagram** (e.g. user → gateway → agent/RAG → LLM → response), **confirm timeline and budget** explicitly—and I'd **ask** the interviewer or customer to clarify constraints so the proposal is realistic. I'd ask what they'd want to see in a follow-up and what their decision process looks like."                                                                                                                                                                                                                                                        |
| **"Structure like STAR + future thinking"**                                                       | **Example A and B** (Scenario, Task, Action, Result, Future thinking)                                            | "I use **Scenario** (who, what they want), **Task** (what I'm being asked to do), **Action** (Scope → Design → Deploy → Communicate with concrete bullets), **Result** (what we get: clear requirements, design, path, messaging), and **Future thinking** (e.g. Agent Assist when they add live agents, or feedback loops and Gemini Enterprise for internal knowledge). I keep each phase to a few sentences so I don't run over time."                                                                                                                                                                                       |
| **"Ask clarifying questions: e.g. already on GCP or first-time migration or hybrid?"**            | **Scope** in both examples; **Clarifying questions** list (below)                                                | "I'd ask: **Already on GCP or first-time migration or hybrid?** What's the **timeline** and **budget**? Who are the **key stakeholders** and what do they care about? For partners I'd add: **primary use case**, **compliance** (data residency, audit). I'd ask this **before** proposing a design so the solution fits their context."                                                                                                                                                                                                                                                                                       |

**Domain-specific skills (leading from conflict, strategy, troubleshooting, consultative, advise clients, develop partnerships):** These show up in **how** you do Scope and Communicate—e.g. "I'd align stakeholders early to avoid conflict later"; "I'd set a clear strategy: POC first, then prod with defined success criteria"; "If the customer is stuck, I'd troubleshoot by clarifying requirements and constraints first"; "I'd advise the client to start with one use case and one metric"; "I'd develop the partnership by involving their IT and business owners in scope and design so they own the outcome." Use the same Scope → Design → Deploy → Communicate flow and plug in these behaviors.

---

**Clarifying questions you can ask (hypotheticals):**

- Is the customer **already on GCP** or **first-time migration** or **hybrid**?
- What are the main constraints — **timeline**, **budget**, **compliance**?
- Who are the key stakeholders and what do they care about?

**Questions to ask the interviewer (your 15 min):**

- About **Google**: team mission, how GenAI is used in the org, culture.
- About **teamwork**: how teams collaborate, how decisions are made, how conflict is handled.
- **Other perspectives**: “What do successful candidates do well in this round?” or “What would you want me to know about this role?”

> [!TIP]
> 💡 **Aha:** RRK combines **system design** (this guide) with **hypotheticals** (STAR + future, audience-aware) and **consultative skills** (scope, stakeholders, POC→prod, present to customer). Prepare a few concrete stories where you defined requirements, designed or deployed something with LLMs, and communicated to different audiences.

---

**What this guide gives you:** **Technical depth** (theory: serving, RAG, agents, evaluation, data pipeline, cost, scale, monitoring, security) so you can reason about trade-offs. **Practical implementation** (tools, stacks, rough estimations, F.1 examples) so you can point to real options (Vertex, Bedrock, LangChain, vLLM, RAGAS, etc.). **Shipping to customers at scale** (Scope → Design → Deploy → Communicate, POC→prod, stakeholder communication, end-to-end examples) so you can prove you can take a GenAI application from idea to production and present it clearly to technical and non-technical audiences. Always connect theory to implementation; that is how you demonstrate technical ability.

_For foundational system design concepts, see [System Design Essentials](./system-design-essentials.md)._

---

## G.5 Resources

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
- [Google ADK Documentation](https://google.github.io/adk-docs/) - Agent Development Kit for building multi-agent systems

### Google Cloud Documentation

- [Vertex AI Generative AI](https://cloud.google.com/vertex-ai/generative-ai/docs/overview)
- [Vertex AI Agent Builder](https://cloud.google.com/vertex-ai/docs/agent-builder/overview)
- [Customer Engagement Suite](https://cloud.google.com/dialogflow/contact-center/docs) - Conversational Agents, Agent Assist, Conversational Insights, CCaaS
- [Vertex AI RAG Engine](https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/overview)
- [Vertex AI Search](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/vertex-ai-search) - Search and recommendations; grounding with your data and Google Search; summaries and Q&A
- [Gemini Enterprise](https://support.google.com/googleapi/answer/gemini-enterprise) - Enterprise AI assistant: agents + unified search across connected business systems; plan-verify-execute; report + sources + audio
- [NotebookLM Enterprise](https://notebooklm.google.com/) - Document-focused: upload docs and web sources; Q&A, summarize, create content, audio summaries; can connect to Gemini Enterprise
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
- **Google Cloud Generative AI Leader certification:** [cloud.google.com/learn/certification/generative-ai-leader](https://cloud.google.com/learn/certification/generative-ai-leader) — proctored exam, ~90 min; use study guides and course lessons to prepare

---

_Last updated: January 2026_
