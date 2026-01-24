## PART VII: AI TOOLS & FRAMEWORKS (Agents + LLM Apps)

**Important note:** This part is **not** official exam content/scope. It’s included as an author-curated reference to capture extra details and practical ecosystem knowledge that often helps in real projects (and sometimes informs exam intuition).

This part is a **tooling map** (commercial + open-source) for building LLM apps and AI agents. The goal is to help you:

- Pick the right layer to “buy vs build” (managed agent platform vs OSS framework vs custom code).
- Recognize common tool categories that show up in exam-style scenarios (RAG, orchestration, guardrails, eval/observability, serving).
- Translate requirements (latency, governance, data access, compliance, cost) into a concrete stack.

### 7.1 A simple taxonomy (how the ecosystem fits together)

Think of agentic/LLM systems as a set of layers:

- **Models**: foundation models (proprietary or open weights).
- **App frameworks**: prompts, tool calling, RAG chains, memory, routing.
- **Agent orchestration**: multi-step execution, planners/routers, multi-agent supervision, retries/timeouts.
- **Knowledge / retrieval**: indexing, vector stores, reranking, grounding.
- **Safety & governance**: policies, guardrails, content safety, secrets/IAM boundaries, audit trails.
- **Evaluation & observability**: offline evals + online monitoring/tracing/cost.
- **Serving**: endpoints, batching, caching, streaming, concurrency control.

**EXAM TIP:** When a question says “fully managed” + “enterprise governance” + “rapid delivery” → prefer a **cloud-managed** offering. When it says “custom orchestration” + “bring your own model” + “fine control” → prefer **OSS frameworks + custom infra**.

### 7.1.1 Core ML/DL frameworks (foundation of “non-LLM” ML engineering)

Even if you work mostly on GenAI, the ML Engineer exam (and many real systems) still rely on classic ML/DL frameworks:

- **Deep learning**: PyTorch, TensorFlow/Keras, JAX (Flax/Haiku).
- **Classical ML**: scikit-learn, XGBoost, LightGBM, CatBoost.
- **Interchange / optimized inference**: ONNX + ONNX Runtime.
- **Legacy (still seen)**: MXNet, PaddlePaddle, CNTK.

#### Probabilistic / Bayesian programming (for uncertainty-aware modeling)

These show up when you need explicit uncertainty, probabilistic inference, and Bayesian modeling:

- **Stan** (probabilistic programming + HMC/NUTS): `https://mc-stan.org/`
- **PyMC** (Python Bayesian modeling): `https://www.pymc.io/`
- **Turing.jl** (Julia Bayesian modeling): `https://turinglang.org/`
- **Edward** (legacy research ecosystem; still referenced historically)

Example table: core model frameworks

| Category      | Tools (examples)                             |
| ------------- | -------------------------------------------- |
| Deep learning | PyTorch, TensorFlow, Keras, JAX (Flax/Haiku) |
| Classical ML  | scikit-learn, XGBoost, LightGBM, CatBoost    |
| Interchange   | ONNX, ONNX Runtime                           |
| Probabilistic | Stan, PyMC, Turing.jl                        |
| Legacy DL     | MXNet, PaddlePaddle, CNTK                    |

### 7.2 Google Cloud (Vertex AI + Google agent stack)

Core docs entry point:

- Generative AI on Vertex AI: `https://cloud.google.com/vertex-ai/generative-ai/docs`

High-yield building blocks you should recognize:

- **Grounding**: Vertex docs include grounding options like Google Search, Maps, and **Vertex AI Search**, plus “grounding responses using RAG”.
- **RAG Engine**: Vertex docs describe **RAG Engine** (RAG overview/quickstart/billing) and guidance for vector DB choices (including **Vertex AI Vector Search** and third-party options).

Practical “when to choose” map:

- **Fastest managed path for enterprise RAG**: Vertex AI Search / RAG Engine (vs rolling your own ingest + vector DB + retrieval + eval).
- **Need custom orchestration/agent design**: use code-first agent frameworks (your own or vendor-supported) + deploy on managed compute (Cloud Run / GKE / Vertex).
- **Need strong governance/observability**: integrate evaluation + logging/tracing + IAM least privilege + audit logs (AgentOps mindset).

### 7.3 AWS (Amazon Bedrock + managed agents)

Core docs entry points:

- Agents: `https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html`
- Amazon Bedrock docs home: `https://docs.aws.amazon.com/bedrock/`

What to look for in AWS-style questions:

- **Managed agents**: when the prompt emphasizes “use Bedrock”, “tool use”, “automate tasks”, “managed”, the answer usually centers on **Bedrock agents**.
- **Safety controls**: look for Bedrock’s “guardrails/safety” concepts when asked about content filtering and policy enforcement.
- **Managed retrieval**: look for “knowledge base”/managed retrieval primitives for RAG-like solutions.

### 7.4 Microsoft Azure (Azure AI Foundry + Azure OpenAI + agent service)

Core docs entry point:

- Microsoft Foundry documentation: `https://learn.microsoft.com/en-us/azure/ai-foundry/`

From the Foundry documentation navigation (high-yield):

- **Foundry Agent Service**: orchestrate/host agents; create custom agents; agent app templates.
- **Agentic retrieval**: create/connect knowledge bases for agentic retrieval.
- **Evaluate agentic workflows** + **AI Red Teaming Agent (preview)**.
- **Content Safety**: Azure content safety / safety docs are surfaced in the same toolchain.
- **Azure OpenAI in Foundry**: Azure OpenAI integration inside the Foundry model.

**EXAM TIP:** When you see “agentic retrieval + eval + red teaming + content safety” described together, it often implies a **platform suite** (not just “write a prompt”).

### 7.4.1 AWS vs GCP vs Azure (ML + GenAI + agents): engineering-focused comparison (2025–2026)

High-level positioning

| Cloud | ML & classical AI | GenAI platform (models, RAG)         | Agentic platform focus                                     |
| ----- | ----------------- | ------------------------------------ | ---------------------------------------------------------- |
| AWS   | SageMaker         | Bedrock                              | Bedrock Agents + AgentCore                                 |
| GCP   | Vertex AI         | Vertex AI (Gemini, RAG)              | Vertex AI Agent Builder + ADK, MCP, A2A                    |
| Azure | Azure ML / Fabric | Azure AI Foundry (OpenAI, Phi, etc.) | Azure AI Foundry Agent Service + Microsoft Agent Framework |

#### Core ML platforms

- **AWS**: **Amazon SageMaker** for training/tuning/processing/experiments/deployment/registry; tight integration with S3/Redshift/EMR/Kinesis/EKS.
- **GCP**: **Vertex AI** as a unified platform (training, AutoML, pipelines, registry, online/offline prediction); integrates natively with BigQuery/Dataflow/Dataproc/GKE.
- **Azure**: **Azure ML** for training/MLOps/registry/endpoints, increasingly integrated into **Microsoft Fabric**; connects with Fabric/Synapse, ADLS, and Power Platform.

#### GenAI / foundation model services

- **AWS – Amazon Bedrock**: managed access to multiple foundation models behind a unified API; includes RAG primitives (knowledge bases) and orchestration via agents.
- **GCP – Vertex AI (Gemini)**: Gemini models (text/vision/code/multimodal) via Vertex GenAI APIs; retrieval + grounding integrated with Vertex search and vector features.
- **Azure – Azure AI Foundry**: access to Azure OpenAI models (and other models via Foundry); grounding over web/search and enterprise data; integrated evaluation/governance surfaces.

#### Agent-specific services (what exam prompts often describe)

- **AWS: Bedrock Agents + AgentCore**
  - **Bedrock Agents**: managed agents that call tools (“action groups”), connect to knowledge bases, and execute multi-step workflows (RAG + tool use).
  - **AgentCore**: positioned as a more flexible runtime (serverless deployment patterns, policy/IAM integration, and observability via CloudWatch/OpenTelemetry-style tooling).
- **GCP: Vertex AI Agent Builder + ADK/MCP/A2A**
  - **Agent Builder**: platform surface to build/scale/govern enterprise agents (Gemini-first).
  - **ADK**: programmable agent behavior.
  - **MCP**: standardizes connecting agents to external tools/resources.
  - **A2A**: coordination between agents (including cross-system/cross-vendor coordination patterns).
- **Azure: Agent Service in Azure AI Foundry**
  - **Foundry Agent Service**: design/deploy/scale agents with governance + observability; integrates with Foundry evaluation and enterprise knowledge.
  - **Microsoft Agent Framework**: referenced as a broader agent framework layer; commonly paired with MCP concepts and Microsoft ecosystem integrations.

#### RAG, tools, and ecosystem integration

- **AWS**: knowledge bases for retrieval; tools via action groups/Lambda/Step Functions; integrates with S3/Redshift/OpenSearch and the AWS data stack.
- **GCP**: RAG features tied to BigQuery and search services; MCP/A2A emphasize tool reuse and interoperability.
- **Azure**: grounding over enterprise data (Fabric/Microsoft Graph) and strong “tool surface” via Teams/Outlook/SharePoint/Dynamics/Power Automate.

#### Practical selection guidance

- **Choose AWS** if you’re SageMaker-centric and want Bedrock-based GenAI/agents tightly coupled to IAM/VPC + Lambda/Step Functions.
- **Choose GCP** if BigQuery is your analytical core and you want Gemini-first development plus interoperability patterns (MCP/A2A) and Vertex agent surfaces.
- **Choose Azure** if you’re Microsoft 365/Dynamics/Fabric-centric and need enterprise agents deeply integrated into the Microsoft stack (Copilot-style workflows).

### 7.5 Open-source / multi-cloud frameworks (the “app & orchestration layer”)

These are common choices when you want portability or deeper control:

- **LangChain + LangGraph** (Python/JS): LLM app building + agent orchestration graphs. Docs: `https://docs.langchain.com/`
- **LlamaIndex**: RAG-first framework (indexing/retrieval patterns + connectors). Docs: `https://docs.llamaindex.ai/`
- **Haystack (deepset)**: RAG pipelines, retrievers/rerankers, production patterns. Docs: `https://docs.haystack.deepset.ai/`
- **Semantic Kernel (Microsoft)**: agent/plugins patterns + orchestration. Docs: `https://learn.microsoft.com/en-us/semantic-kernel/`
- **AutoGen (Microsoft Research)**: multi-agent conversation patterns. Repo: `https://github.com/microsoft/autogen`
- **CrewAI**: role-based multi-agent orchestration (teams of agents with roles/goals/tasks; often paired with tools + RAG). Docs: `https://docs.crewai.com/`
- **PydanticAI**: typed/validated agent outputs and tool calling patterns. Docs: `https://ai.pydantic.dev/`
- **DSPy**: programmatic prompt optimization / “compiled prompting”. Repo: `https://github.com/stanfordnlp/dspy`
- **Rasa**: assistants/chatbots with NLU + dialogue management. Docs: `https://rasa.com/docs/`
- **promptflow / Prompt Flow (Microsoft)**: workflow authoring + evaluation loops for LLM apps (Azure-centric, but OSS tooling exists). Repo: `https://github.com/microsoft/promptflow`

Other tools you may see in the wild (often more “product” than “framework”, or newer):

- **OpenAI Swarm** (lightweight multi-agent patterns; vendor ecosystem specific)
- **Phidata**, **Agno** (Python agent frameworks with pluggable LLMs/vector stores)
- **FastAgency** (production/agent acceleration patterns)
- **Lindy** (business workflow automation agents)
- **MetaGPT**, **OpenAgents**, **Letta**, **AutoGPT-style stacks** (agentic automation ecosystems; quality varies widely—evaluate carefully)

### 7.5.1 Commercial “agent builders” and provider SDKs (OpenAI / Anthropic / others)

These sit between “pure model API” and “full cloud platform”: they provide agent runtimes, tool calling patterns, and ecosystem integrations.

- **OpenAI platform (APIs + tooling)**:
  - **ChatGPT**: end-user product; often used for prototyping and ad-hoc analysis.
  - **OpenAI API**: model access for apps (tool/function calling patterns, structured outputs, etc.). Docs: `https://platform.openai.com/docs`
  - **OpenAI agent guidance/frameworks**: lightweight orchestration patterns (e.g., Swarm) and official “how to build agents” materials.
- **Anthropic (Claude ecosystem)**:
  - **Claude**: end-user assistant.
  - **Claude Code**: developer-focused “coding agent” workflow integrated with your repo/terminal (useful for refactors, debugging, code navigation).
  - **Model Context Protocol (MCP)**: open protocol used broadly for tool/context integration in agentic apps. Spec/docs: `https://modelcontextprotocol.io/`
- **Other commercial app platforms** (often appear in industry, less in exam questions):
  - **Cohere**, **Mistral**, etc. (provider APIs + embeddings/rerankers + deployment options; vendor-specific feature sets).
  - **Vellum**: visual LLM app builder + SDK for workflows/experiments/evals. `https://www.vellum.ai/`

**EXAM TIP:** Provider SDKs (OpenAI/Anthropic/etc.) help you ship faster, but enterprise answers still hinge on **governance, eval, monitoring, and least privilege tool execution**.

### 7.5.2 MCP vs A2A (interoperability protocols for the “agent internet”)

Two interoperability ideas are showing up more and more in agent stacks:

#### MCP: Model Context Protocol (agent/host ↔ tools/data)

**MCP** is an open standard (originating from Anthropic) for connecting LLMs/agents to tools, APIs, files, and databases via a consistent protocol.

- **Architecture**: client–server.
  - A **host** (chat app, IDE, agent runtime) runs an **MCP client**.
  - The client connects to one or more **MCP servers** that expose:
    - **Tools** (callable actions)
    - **Resources** (files, KB snippets, data sources)
    - **Prompts** (reusable prompt templates)
- **Why it matters**: replaces bespoke plugin systems and one-off integrations with a universal “adapter layer” for context + tools (capability discovery, structured calls, streaming, standardized errors).
- **Newer feature direction**: “sampling”-style features can allow an MCP server to ask the client’s model to generate text, enabling richer workflows without the server owning model keys.

**When to use MCP**

- You want a single tool spec (e.g., Jira, GitHub, Snowflake) reusable across multiple hosts/runtimes.
- You need secure, auditable access to enterprise systems via a well-defined interface (instead of ad-hoc plugins).

#### A2A: Agent2Agent protocol (agent ↔ agent)

**Agent2Agent (A2A)** is an open protocol (championed by Google’s agent ecosystem) that focuses on **horizontal interoperability**: agents discover each other, exchange messages, negotiate capabilities, and coordinate tasks across platforms.

- **What it enables**: a “team of agents” where different agents can live in different systems (teams, vendors, org boundaries) but still collaborate as peers.
- **Typical concepts**: agent identities, capability descriptions, secure communication, multi-step workflows.

**When to use A2A**

- You want cross-team/cross-org collaboration (finance agent + support agent + scheduler agent) without a single monolithic orchestrator.
- You expect partner/vendor agents to cooperate through a neutral “agent mesh” protocol.

#### MCP vs A2A (and how they combine)

- **MCP is vertical**: connect _one agent/host_ to _many tools/data sources_ (tool/data connectivity).
- **A2A is horizontal**: connect _many agents_ to _each other_ (coordination and delegation).

**Common best practice**

- Use **MCP** for tool/data connectivity and consistent enterprise integrations.
- Use **A2A** (or similar patterns) for multi-agent coordination across boundaries.

Simple illustration:

- An ops assistant uses **MCP** to access GitHub/Jira/Kubernetes, and uses **A2A** to delegate cost analysis to a FinOps agent owned by another team.

#### Other emerging agent protocol patterns

You may also see references to lighter-weight “agent workflow” specs (sometimes described as REST-style patterns) when MCP/A2A are considered too heavy for a given integration.

**How to choose quickly**

- **You need deterministic control + testability**: graph/state-machine orchestration (LangGraph-style).
- **You need RAG/connectors first**: LlamaIndex / Haystack-style.
- **You want strong typing/validation**: PydanticAI-style.
- **You want “multi-agent roles” quickly**: CrewAI/AutoGen-style.

Example table: agent and LLM application frameworks

| Category                     | Tools (examples)                                      |
| ---------------------------- | ----------------------------------------------------- |
| LLM apps + orchestration     | LangChain, LangGraph, Semantic Kernel, promptflow     |
| RAG + indexing-first         | LlamaIndex, Haystack                                  |
| Multi-agent orchestration    | CrewAI, AutoGen                                       |
| Typed/validated “agent code” | PydanticAI                                            |
| Classic assistants/chatbots  | Rasa                                                  |
| Vendor-native                | OpenAI (Agents/SDK patterns), Semantic Kernel, Vellum |

### 7.6 Vector stores & retrieval infrastructure (RAG substrate)

Common options (managed or self-hosted):

- **Postgres + pgvector** (self-hosted or managed Postgres): simplest “good enough” vector store for many teams.
- **Dedicated vector DBs**: Milvus, Weaviate, Qdrant, Pinecone, etc.
- **Cloud-native**: Vertex AI Vector Search / OpenSearch vector capabilities / Azure AI Search vector capabilities.
- **Hybrid search engines (dense + lexical)**: Elasticsearch, OpenSearch, Vespa, Solr (with vector plugins/capabilities).

Selection guide:

- **Small/medium scale + strong relational needs** → Postgres+pgvector
- **Large-scale ANN / low latency + ops budget** → managed vector search or dedicated vector DB
- **Enterprise search + governance** → managed search/RAG products (Vertex AI Search / Azure AI Search, etc.)

Example table: RAG components

| Layer         | Options (examples)                                            |
| ------------- | ------------------------------------------------------------- |
| Orchestration | LangChain, LangGraph, AutoGen, CrewAI                         |
| Indexing      | LlamaIndex, Haystack, Elasticsearch/OpenSearch, Vespa         |
| Vector DB     | Pinecone, Chroma, Qdrant, Weaviate, pgvector/Postgres, Milvus |

**Embedding / similarity providers** (often accessed via frameworks above): OpenAI, Cohere, Voyage AI, Jina, etc.

### 7.6.1 Model hubs and the Hugging Face ecosystem (the “GitHub for models/data” layer)

This layer is easy to miss if you only think in terms of “cloud platforms”, but it’s central in real-world GenAI work:

#### What Hugging Face actually is

- **Community hub** for models, datasets, and apps (discovery + collaboration + versioning).
- **Core libraries** (commonly used in production):
  - Transformers, Datasets, Tokenizers, Diffusers
  - Accelerate (training acceleration/distribution helpers)
  - PEFT (parameter-efficient fine-tuning), TRL (RL fine-tuning patterns)
- **Platform features**:
  - Model Hub, Dataset Hub, Spaces (demos/apps)
  - Inference Endpoints / hosted inference options (plus enterprise/on-prem patterns)

#### Where HF fits vs cloud platforms

- **HF is cloud-agnostic content + tooling**: you can pull HF models into **SageMaker**, **Vertex AI**, **Azure ML**, or your own serving stack.
- **Clouds compete at the platform layer** (infra + governance + managed endpoints + agent stacks) but also **integrate or mirror** “model catalog” concepts.

**EXAM TIP:** If the prompt says “choose a managed platform to deploy/operate” → cloud endpoints usually win. If it says “we need model/dataset discovery + community artifacts + fast iteration” → a **model hub** mindset (HF-like) is a better fit.

#### “HF-equivalent” ecosystems (model hubs and catalogs)

Major open/community hubs:

- **Hugging Face Hub** (dominant open model/dataset hub)
- **ModelScope** (Alibaba ecosystem; strong regional footprint)
- **Replicate** (catalog + easy hosting/inference API)
- **DagsHub** (collaborative ML platform; “GitHub + data/versioning + ML tooling” style)
- **Specialized SD marketplaces** (Stable Diffusion models/LoRAs; quality/governance varies widely)

Cloud-native “catalogs” (more curated/managed, less community-driven):

- **Vertex AI** model catalog / AI Studio directories
- **Bedrock** model catalog / AWS Marketplace models
- **Azure** AI model catalog / Azure AI Studio

#### Inference & serving alternatives (if you mainly think of HF as “host my model and give me an API”)

- **Replicate**: multi-model inference API with simple deployment story
- **Self-hosted serving stacks**: BentoML / KServe / Triton / Ray Serve (often serving HF models)
- **Cloud-native endpoints**: SageMaker endpoints, Vertex AI endpoints, Azure ML / Foundry endpoints

### 7.7 LLM serving / inference engines (when you self-host)

If you host open-weight models (or need extreme throughput), inference engines become a “first-class” tool choice:

- **vLLM**: `https://docs.vllm.ai/`
- **SGLang**: `https://github.com/sgl-project/sglang`
- **TensorRT-LLM**: `https://github.com/NVIDIA/TensorRT-LLM`
- **Text Generation Inference (TGI)**: `https://github.com/huggingface/text-generation-inference`

High-yield concepts to recognize across these projects:

- **Continuous batching**, **KV cache management**, **prefill vs decode scheduling**, **prefix/context caching**, **streaming**.

### 7.7.0 Open-source foundation models (beyond LLMs)

In practice, “open-source foundation models” usually means **open weights** you can download and run (license permitting), across multiple modalities:

- **Text (LLMs)**: general-purpose chat/instruct and code models (examples below).
- **Vision-language (VLMs)**: models that take images + text (for OCR-like reasoning, chart/doc Q&A, visual grounding).
- **Text-to-image / diffusion**: image generation and editing pipelines (often paired with LoRAs/control adapters).
- **Speech/audio**: speech-to-text and speech generation models used in voice agents.

**Engineering note:** Always check the **license**, distribution restrictions, and acceptable use policy. “Open weights” ≠ “free for any use”.

#### Image models (generation + editing) — what you should know

There are two common classes of “image models” used in production:

- **Vision-language models (VLMs)**: understand images and answer questions (see “Vision-language models” below).
- **Generative image models**: create/edit images (diffusion-style pipelines are the most common open-weight option).

**Common image generation/editing tasks**

- **Text-to-image**: generate images from prompts.
- **Image-to-image**: stylize/transform a source image (often for “variations”).
- **Inpainting/outpainting**: fill missing regions / extend borders.
- **Control/conditioning**: constrain generation with structure (edge maps, depth, pose) via control adapters.
- **LoRA adapters**: small finetunes for style/character/product customization.

**Open-weight model families you’ll see a lot**

- **Stable Diffusion / SDXL** (and derivatives): widely used open diffusion ecosystem.
- **FLUX.1** (and similar newer open-weight image models): often used for higher-quality generations (check license terms carefully).
- **PixArt-α** and other research-grade open models: sometimes used for specific quality/speed tradeoffs.

**Tooling that matters**

- **Diffusers** (Hugging Face) is the standard Python library for diffusion pipelines.
- **ComfyUI / Automatic1111-style UIs** are common for workflows (great for rapid iteration; production usually uses code).

**Engineering gotchas**

- **Latency**: diffusion requires multiple denoising steps; reduce steps, use faster samplers, quantize, or distill for speed.
- **VRAM**: image generation is memory-hungry; batch size and resolution dominate cost.
- **Safety**: image generation needs filtering (prompt + output classification) and provenance controls (watermarking/content policy).

#### Vision-language models (image understanding, visual Q&A, document/diagram reasoning)

VLMs are used for:

- **Image Q&A / captioning**
- **Document understanding** (tables, charts, screenshots)
- **Visual grounding** (refer to regions/objects)
- **Multimodal RAG** (retrieve relevant images/pages + answer)

Common open-weight components you’ll encounter:

- **CLIP-like encoders** (for image/text embeddings and retrieval)
- VLM families such as **LLaVA-like** and **Qwen-VL-like** models (names vary by release)

Engineering pattern (very common):

- Use **CLIP/SigLIP-style embeddings** for retrieval (image search / “find relevant page/frame”), then use a **VLM** to answer with the retrieved visuals as context.

#### Video models (understanding vs generation)

Video brings unique constraints: long sequences, high compute, and multi-modal signals (frames + audio + metadata).

**A) Video understanding (classification, tagging, moderation, highlights)**

Typical tasks:

- **Action recognition** (what is happening)
- **Event detection** (important moments)
- **Object/scene tagging** (what appears)
- **Moderation/safety** (unsafe content)
- **Temporal localization** (when an event happens)

Common model families/approaches you’ll see:

- **Frame sampling + image backbone** (cheap baseline): sample frames, run image model per frame, aggregate.
- **Video Transformers** (ViT-style extended over time): better temporal modeling, higher cost.
- **Two-stage retrieval + reasoning**:
  - Stage 1: embed frames/clips for retrieval
  - Stage 2: VLM/LLM summarizes or answers questions from selected clips

**Engineering defaults that work**

- **Sampling strategy matters**: uniform sampling for coverage; shot-boundary / scene-change sampling for efficiency.
- **Store embeddings at the clip level** for search and reuse across tasks.
- **Evaluate with temporal metrics** when localization matters (not just per-video accuracy).

**B) Video generation (text-to-video / image-to-video)**

Open-weight video generation evolves quickly; the recurring practical constraints are stable:

- **Compute cost** is high (many frames, high resolution, temporal coherence).
- **Serving is hard** (long runtimes, huge VRAM, often async/batch).
- **Quality evaluation** is tricky (you usually need human eval plus basic automated metrics).

**Engineering tip:** treat video generation as an **async job** (queue + batch GPUs) unless the product truly demands interactive latency.

### 7.7.1 Open-source / open-weight LLM examples (what people actually run)

When someone says “open-source LLM”, they often mean **open-weight** models you can run yourself (license permitting). Common examples you’ll see:

- **Qwen (Alibaba)**: widely used open-weight family across sizes; common choice for multilingual and strong general capability.
- **Llama (Meta)**: very common default for self-hosted chat/instruct deployments.
- **Mistral / Mixtral (Mistral AI)**: popular for performance/efficiency tradeoffs (and MoE variants).
- **Gemma (Google)** and other open releases: often used for smaller/faster deployments.

Where to find and evaluate them:

- **Model hubs** (Hugging Face, etc.) for discovery + downloads + benchmarks.
- Use an **eval harness** (Ragas/DeepEval/promptfoo-style) + your own golden set before swapping models.

**EXAM TIP:** “We need to run on-prem / avoid vendor lock-in / control weights” → open-weight models + self-hosted serving (vLLM/TGI/Triton) is the common direction.

### 7.7.2 API routers / aggregators (OpenRouter-style)

These products sit above multiple model providers and expose a single API surface:

- **Why teams use them**:
  - Normalize APIs across providers
  - Route by price/latency/quality/fallback
  - Centralize usage/cost controls
- **Tradeoffs**:
  - Adds another vendor in the trust/data path (security + compliance review)
  - Harder to use provider-specific features unless they’re exposed

If you’re using an API router, still treat **evaluation + monitoring** as mandatory, because routing changes can change quality.

### 7.7.3 Non-hyperscaler GPU & inference clouds (cost/perf alternatives)

Beyond AWS/GCP/Azure, you’ll see specialized GPU clouds that focus on price/performance for training and inference. Example:

- **Hyperbolic**: “open-access AI cloud” offering on-demand GPU clusters and serverless inference, marketed as API-compatible with common ecosystems and positioned on cost. See: `https://www.hyperbolic.ai/` ([Hyperbolic](https://www.hyperbolic.ai/)).

**Engineering checklist before adopting**: data residency/compliance, private networking, IAM/audit logs, model availability, SLAs, and observability/export (OpenTelemetry/metrics/logs).

### 7.8 Evaluation + observability (LLMOps / AgentOps tooling)

Typical categories:

- **Offline evaluation**: regression tests for prompts/RAG/agents; golden sets; rubric scoring.
- **Tracing/observability**: tool calls, intermediate steps, latency, tokens/cost, errors.
- **Safety evaluation**: prompt injection tests, jailbreak suites, policy compliance checks.

Representative tools (vendor-agnostic):

- **DeepEval**: `https://github.com/confident-ai/deepeval`
- **Ragas** (RAG evaluation): `https://github.com/explodinggradients/ragas`
- **TruLens**: `https://github.com/truera/trulens`
- **Arize Phoenix**: `https://github.com/Arize-ai/phoenix`
- **promptfoo**: `https://github.com/promptfoo/promptfoo`
- **LangSmith** (LangChain ecosystem): `https://docs.smith.langchain.com/`
- **Weights & Biases Weave**: `https://weave.wandb.ai/`

**EXAM TIP:** The “correct” operational answer is almost never “manually test a few prompts.” It’s usually **eval dataset + automated evals + monitoring/tracing + safety tests**.

### 7.8.1 MLOps, pipelines, and observability (classic ML + GenAI)

This is the “everything around the model” tool layer (data, pipelines, deployment, monitoring):

- **Experiment tracking / registry**: MLflow (`https://mlflow.org/`), Weights & Biases (`https://wandb.ai/`), Neptune (`https://neptune.ai/`).
- Also common: Polyaxon (`https://polyaxon.com/`), Comet (`https://www.comet.com/`).
- **Data/pipeline versioning**: DVC (`https://dvc.org/`), lakeFS (`https://lakefs.io/`), Delta Lake (`https://delta.io/`).
- Also common: Pachyderm (`https://www.pachyderm.com/`).
- **Orchestration**: Airflow (`https://airflow.apache.org/`), Prefect (`https://www.prefect.io/`), Dagster (`https://dagster.io/`), Kubeflow Pipelines (`https://www.kubeflow.org/docs/components/pipelines/`).
- Also common: Metaflow (`https://metaflow.org/`).
- **Feature stores**: Feast (`https://feast.dev/`), Tecton (`https://www.tecton.ai/`), Hopsworks (`https://www.hopsworks.ai/`).
- **Deployment / serving**: BentoML (`https://docs.bentoml.org/`), Seldon Core (`https://docs.seldon.io/`), KServe (`https://kserve.github.io/website/`), TorchServe (`https://github.com/pytorch/serve`), Triton (`https://github.com/triton-inference-server/server`).
- Also common: Ray Serve (`https://docs.ray.io/en/latest/serve/index.html`).
- **Monitoring / observability**: Evidently (`https://www.evidentlyai.com/`), WhyLogs (`https://github.com/whylabs/whylogs`), plus commercial platforms (Arize/WhyLabs/etc.).
- Also common: Fiddler (`https://www.fiddler.ai/`).

**EXAM TIP:** End-to-end managed platforms (Vertex AI / SageMaker / Azure ML / Databricks) bundle many of the above; exam prompts often reward “use the managed platform” when governance + scale + time-to-market are key.

Full platforms (end-to-end suites you’ll see in enterprises):

- **Databricks**: `https://www.databricks.com/`
- **Vertex AI**: `https://cloud.google.com/vertex-ai`
- **SageMaker**: `https://aws.amazon.com/sagemaker/`
- **Azure ML**: `https://learn.microsoft.com/en-us/azure/machine-learning/`
- **Snowflake Cortex**: `https://www.snowflake.com/en/data-cloud/cortex/`

### 7.9 Security & guardrails (tooling checklist)

Regardless of vendor/framework, the same controls repeat:

- **Least privilege**: tools run under constrained identities (service accounts/roles).
- **Secrets hygiene**: never put credentials in prompts; use secret managers; short-lived tokens.
- **Prompt injection defense**: treat retrieved docs/tool outputs as untrusted; separate instructions vs data; validate tool args.
- **Allow-lists**: tools/functions with explicit schemas; restrict high-risk actions; add human approval for destructive ops.
- **Auditability**: logs/traces retained; changes to prompts/tools/versioned.

### 7.10 “Default stacks” by scenario (quick mapping)

- **Fast managed enterprise agent**:
  - Vendor agent platform (Vertex/Azure/AWS) + managed RAG/search + built-in eval + observability
- **Portable app with custom orchestration**:
  - LangGraph/LlamaIndex + your chosen vector DB + your evaluation stack + deploy on Cloud Run/GKE
- **Self-hosted inference at scale**:
  - vLLM/SGLang/TGI + vector DB + strict gateway + eval/monitoring + GPU autoscaling

### 7.11 “AI tools” for developers and business users (products you’ll see everywhere)

These are not frameworks, but they matter in real engineering orgs (policy, security, workflows):

- **Coding assistants**: GitHub Copilot, Codeium, Replit Ghostwriter, “AI code review” tools (varies by org).
- **AI-native IDEs / coding agents**: Cursor (`https://www.cursor.com/`), Claude Code (Anthropic), etc.
- **General assistants**: ChatGPT, Claude, Gemini, Perplexity, Microsoft Copilot.
- **Automation / app builders**: Zapier, Make, n8n (AI nodes), voice-agent builders, no/low-code agent builders.
- **Business suites**: Intercom/Zendesk/HubSpot AI features, etc.

### 7.12 Integration platforms and no-/low-code AI

This is a fast-growing “glue layer” for agents in real companies:

- **iPaaS with AI**: Zapier, Make, n8n (AI/LLM nodes).
- **Agent / integration hubs**: Composio (`https://composio.dev/`), and similar connector hubs for Slack/GitHub/Salesforce/Jira, etc.
- **Visual LLM builders**: LangFlow (`https://github.com/langflow-ai/langflow`), Flowise (`https://github.com/FlowiseAI/Flowise`), plus commercial builders (varies widely).
- **Voice/real-time agent builders**: commonly used in call center/voicebot stacks (vendor landscape changes quickly).

**EXAM TIP:** No/low-code tools are great for speed, but production answers still require **security, auditability, and evaluation**.

### 7.13 Domain- and language-specific ML libraries (useful in “what tool would you use?” questions)

- **NLP**: spaCy (`https://spacy.io/`), NLTK (`https://www.nltk.org/`), Hugging Face Transformers/tokenizers (`https://huggingface.co/docs/transformers/`).
- **Computer vision**: OpenCV (`https://opencv.org/`), Detectron2 (`https://github.com/facebookresearch/detectron2`), MMDetection/MMCV (`https://github.com/open-mmlab/mmdetection`), Kornia (`https://github.com/kornia/kornia`).
- **Recommenders**: implicit (`https://github.com/benfred/implicit`), plus ecosystem-specific libs.
- **Time series**: Prophet (`https://github.com/facebook/prophet`), darts (`https://github.com/unit8co/darts`), GluonTS (`https://github.com/awslabs/gluonts`).
- **Reinforcement learning**: Stable-Baselines3 (`https://github.com/DLR-RM/stable-baselines3`), RLlib (`https://docs.ray.io/en/latest/rllib/`).
- **Non-Python ecosystems** (common in “awesome-ml” lists): Go/Java/C++/Julia libraries vary; pick based on deployment/runtime constraints and team skills.

### 7.14 Governance, risk, and compliance frameworks

- **NIST AI Risk Management Framework (AI RMF)**: `https://www.nist.gov/itl/ai-risk-management-framework`
- **Cloud Responsible AI toolkits**: major cloud platforms provide responsible AI guidance and tooling (evaluation, safety, governance, auditability) integrated into their ML/GenAI suites.

Practical “engineering constraints” to remember:

- **Data handling**: what can be sent to external models; retention; logging; PII policies.
- **Identity + access**: SSO/IAM integration; audit trails.
- **Evaluation**: how you measure “helpfulness” and prevent regressions.
