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

```mermaid
flowchart LR
  subgraph MCP["MCP (agent/host ↔ tools/data)"]
    Host["Host: IDE / chat app / agent runtime"] --> Client[MCP client]
    Client --> S1[MCP server: Jira]
    Client --> S2[MCP server: GitHub]
    Client --> S3[MCP server: Snowflake]
  end

  subgraph A2A["A2A (agent ↔ agent)"]
    A[Ops agent] <--> B[FinOps agent]
    A <--> C[Support agent]
    B <--> D[Scheduler agent]
  end
```

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

---

## 7.5.3 Deep Dive: Agent Frameworks with Code Examples

This section provides implementation-ready code snippets for the most important agent frameworks.

### Vertex AI Agent Engine (Google Cloud's Managed Agent Platform)

**Vertex AI Agent Engine** is Google Cloud's managed runtime for deploying, scaling, and operating AI agents in production. It handles infrastructure, scaling, traffic management, and integrates with Vertex AI's MLOps ecosystem.

#### Key capabilities

| Capability             | Description                                                                  |
| ---------------------- | ---------------------------------------------------------------------------- |
| **Managed deployment** | Deploy agents built with ADK or custom code; auto-scaling, traffic splitting |
| **Tool orchestration** | Built-in support for function calling, grounding, and tool execution         |
| **Session management** | Conversation state, memory persistence across turns                          |
| **Evaluation**         | Integrated agent evaluation and testing workflows                            |
| **Observability**      | Logging, tracing, metrics for agent behavior (latency, tool calls, errors)   |
| **Safety**             | Content filters, guardrails, and AI red teaming integration                  |

#### Agent Engine vs Agent Builder

| Concept                         | What It Is                                                                            |
| ------------------------------- | ------------------------------------------------------------------------------------- |
| **Vertex AI Agent Builder**     | Visual/low-code UI for building agents (playbooks, data stores, tools)                |
| **Vertex AI Agent Engine**      | Runtime/infrastructure for deploying agents (from ADK, custom code, or Agent Builder) |
| **ADK (Agent Development Kit)** | Code-first Python SDK for building agents programmatically                            |

**EXAM TIP:** When a question says "production deployment" + "managed" + "traffic splitting" + "agents" → think **Agent Engine**. When it says "code-first" + "custom logic" → think **ADK**.

---

### ADK (Agent Development Kit) — Google's Code-First Agent Framework

**ADK** is Google's open-source Python framework for building agents that can be deployed to Vertex AI Agent Engine or run locally.

Official repo: `https://github.com/google/adk-python`  
Official docs: `https://google.github.io/adk-docs/`

#### ADK core concepts

| Concept     | Description                                                                            |
| ----------- | -------------------------------------------------------------------------------------- |
| **Agent**   | The main orchestrator — decides what to do, calls tools, maintains state               |
| **Tool**    | A callable function the agent can invoke (API calls, database queries, code execution) |
| **Session** | Conversation context and memory                                                        |
| **Runner**  | Executes the agent locally or remotely                                                 |

#### ADK: Simple agent with tools (pseudocode — verify imports in ADK docs)

```python
# NOTE: ADK is evolving quickly; treat this as *structure*, not exact imports.
# Verify the current import paths / class names in the official docs:
# - https://google.github.io/adk-docs/
# - https://github.com/google/adk-python

# 1) Define tools as regular functions
def get_weather(city: str) -> str:
    # Call a real API in production
    return f"Weather in {city}: 72°F, sunny"

def search_kb(query: str) -> str:
    # Query your search / vector store in production
    return f"Found results for '{query}': [doc1, doc2, doc3]"

# 2) Register tools with the agent runtime
tools = [get_weather, search_kb]  # (wrapped/registered per ADK docs)

# 3) Create an agent with instructions + tools
agent = {
    "name": "assistant",
    "model": "gemini-2.0-flash",
    "instruction": (
        "Use get_weather for weather. Use search_kb for internal knowledge. "
        "Cite sources when using retrieved content."
    ),
    "tools": tools,
}

# 4) Run locally (runner/session setup per ADK docs)
result = run(agent, "What's the weather in Seattle and find docs about ML pipelines")
print(result)
```

#### ADK: Multi-agent with delegation (pseudocode)

```python
# 3-agent pipeline: research → write → review
researcher = {"name": "researcher", "instruction": "Research and cite sources.", "tools": [search_kb]}
writer = {"name": "writer", "instruction": "Write a clear summary from research."}
reviewer = {"name": "reviewer", "instruction": "Check accuracy, clarity, and add missing caveats."}

pipeline = sequential([researcher, writer, reviewer])  # per ADK composition primitives
final = pipeline.run("Create a summary of recent developments in agentic AI")
print(final)
```

#### ADK: Deploy to Vertex AI Agent Engine (pseudocode — verify current deployment API)

```python
# Pseudocode: deployment API names change; use official docs for the exact call.
# Typical inputs: agent artifact, project, region, display name, scaling config.
deployment = deploy_agent(
    agent=agent,
    project_id="your-project",
    location="us-central1",
    display_name="my-assistant-agent",
    min_replicas=1,
    max_replicas=10,
)
print(deployment.endpoint)
```

---

### Agentic RAG (Retrieval as an Agent Decision)

Traditional RAG: query → retrieve → generate (single pass).  
**Agentic RAG**: the agent **decides** when/whether to retrieve, can do **multi-step retrieval**, and **validates** retrieved context.

#### Why agentic RAG?

| Traditional RAG                    | Agentic RAG                                              |
| ---------------------------------- | -------------------------------------------------------- |
| Always retrieves on every query    | Agent decides if retrieval is needed                     |
| Single retrieval step              | Multi-hop: retrieve → reason → retrieve more             |
| No validation of retrieved docs    | Agent can verify relevance, ask clarifying questions     |
| Fixed retrieval query = user query | Agent can rewrite/decompose queries for better retrieval |

#### Agentic RAG with ADK (pseudocode — verify ADK tool wrappers in docs)

```python
def retrieve_documents(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve relevant documents from vector store."""
    # In production: embed query, search vector DB, return chunks
    return [
        {"content": "ML pipelines automate...", "source": "doc1.pdf", "score": 0.92},
        {"content": "Feature stores provide...", "source": "doc2.pdf", "score": 0.87},
    ]

def verify_source(claim: str, source_id: str) -> str:
    """Verify a specific claim against the original source document."""
    # In production: fetch full doc, check if claim is supported
    return f"Claim '{claim}' is supported by {source_id}"

agent = {
    "name": "agentic_rag",
    "model": "gemini-2.0-flash",
    "instruction": (
        "Decide if retrieval is needed. If needed, decompose the query, retrieve, and iterate. "
        "Verify key claims and cite sources. If you cannot find support in retrieved sources, say so."
    ),
    # Tools are wrapped/registered per ADK docs:
    "tools": [retrieve_documents, verify_source],
}

answer = run(agent, "Compare Feature Store vs raw feature tables for churn prediction. Cite sources.")
print(answer)
```

#### Multi-hop retrieval pattern

```python
# Agent instruction for multi-hop reasoning
multi_hop_instruction = """
When answering complex questions:

1. DECOMPOSE: Break the question into sub-questions
   Example: "How does X compare to Y for use case Z?"
   → Sub-Q1: "What is X and its capabilities?"
   → Sub-Q2: "What is Y and its capabilities?"
   → Sub-Q3: "What are requirements for use case Z?"

2. RETRIEVE: Search for each sub-question separately

3. SYNTHESIZE: Combine retrieved information to answer the original question

4. VALIDATE: Cross-check key facts across multiple sources

5. ITERATE: If gaps remain, formulate follow-up queries
"""
```

---

### Multi-Agent Architectures

Multi-agent systems use multiple specialized agents that collaborate on complex tasks.

#### Common patterns

| Pattern                 | Description                                       | When to Use                                |
| ----------------------- | ------------------------------------------------- | ------------------------------------------ |
| **Sequential pipeline** | Agent A → Agent B → Agent C (linear handoff)      | Content creation, review workflows         |
| **Parallel fan-out**    | Query sent to multiple agents; results aggregated | Research, multi-perspective analysis       |
| **Router/dispatcher**   | Classifier routes to specialist agents            | Customer support, domain-specific handling |
| **Supervisor/manager**  | Manager delegates to workers, tracks progress     | Complex projects, iterative refinement     |
| **Debate/adversarial**  | Agents argue opposing views; judge decides        | High-stakes decisions, red teaming         |

#### Supervisor pattern diagram

```
                    ┌─────────────────┐
                    │   Supervisor    │
                    │ (orchestrates)  │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  Researcher │   │   Writer    │   │  Reviewer   │
    │   Agent     │   │   Agent     │   │   Agent     │
    └─────────────┘   └─────────────┘   └─────────────┘
```

---

### CrewAI — Role-Based Multi-Agent Orchestration

**CrewAI** makes it easy to define "crews" of agents with roles, goals, and backstories that work together on tasks.

Official docs: `https://docs.crewai.com/`

#### CrewAI core concepts

| Concept     | Description                                       |
| ----------- | ------------------------------------------------- |
| **Agent**   | A role with goals, backstory, and tools           |
| **Task**    | A specific job assigned to an agent               |
| **Crew**    | A team of agents working together                 |
| **Process** | How tasks are executed (sequential, hierarchical) |

#### CrewAI: Research team example

```python
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool

# Tools
search_tool = SerperDevTool()  # Web search
web_rag_tool = WebsiteSearchTool()  # RAG over websites

# Define agents with roles
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in AI agents",
    backstory="""You are a senior researcher at a leading tech think tank.
    You have a knack for finding obscure but important information.
    You're known for thorough, well-sourced analysis.""",
    tools=[search_tool, web_rag_tool],
    verbose=True,
    llm="gemini/gemini-2.0-flash",  # or "openai/gpt-4o"
)

writer = Agent(
    role="Tech Content Strategist",
    goal="Craft compelling content about AI technology",
    backstory="""You are a renowned content strategist who specializes
    in making complex technical topics accessible to business leaders.
    You create clear, engaging narratives backed by solid research.""",
    verbose=True,
    llm="gemini/gemini-2.0-flash",
)

editor = Agent(
    role="Senior Editor",
    goal="Ensure content is accurate, clear, and publication-ready",
    backstory="""You are a meticulous editor with decades of experience
    in technical publishing. You catch errors others miss and improve
    clarity without losing technical accuracy.""",
    verbose=True,
    llm="gemini/gemini-2.0-flash",
)

# Define tasks
research_task = Task(
    description="""Research the latest developments in AI agent frameworks.
    Focus on: ADK, LangGraph, CrewAI, AutoGen.
    Compare their architectures, use cases, and production readiness.
    Include code examples where relevant.""",
    expected_output="Detailed research report with comparisons and examples",
    agent=researcher,
)

writing_task = Task(
    description="""Using the research report, write a comprehensive article
    about AI agent frameworks for technical decision makers.
    Make it engaging but technically accurate.
    Include a comparison table and recommendations.""",
    expected_output="Publication-ready article (1500-2000 words)",
    agent=writer,
    context=[research_task],  # Depends on research
)

editing_task = Task(
    description="""Review and edit the article for:
    - Technical accuracy
    - Clarity and flow
    - Grammar and style
    Provide the final polished version.""",
    expected_output="Final edited article ready for publication",
    agent=editor,
    context=[writing_task],
)

# Create crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,  # or Process.hierarchical
    verbose=True,
)

# Run the crew
result = crew.kickoff()
print(result)
```

#### CrewAI: Hierarchical process (manager delegates)

```python
from crewai import Crew, Process

# Create a crew with hierarchical management
managed_crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.hierarchical,
    manager_llm="gemini/gemini-2.0-flash",  # Manager uses this LLM
    verbose=True,
)

# The manager agent is created automatically
# It decides task order, delegation, and iteration
result = managed_crew.kickoff()
```

#### CrewAI: Custom tools

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class DatabaseQueryInput(BaseModel):
    query: str = Field(..., description="SQL query to execute")
    database: str = Field(default="analytics", description="Target database")

class DatabaseQueryTool(BaseTool):
    name: str = "database_query"
    description: str = "Execute SQL queries against internal databases"
    args_schema: type[BaseModel] = DatabaseQueryInput

    def _run(self, query: str, database: str = "analytics") -> str:
        # In production: execute query securely
        return f"Query results from {database}: [row1, row2, row3]"

# Use custom tool with agent
data_analyst = Agent(
    role="Data Analyst",
    goal="Extract insights from company data",
    backstory="Expert SQL analyst with deep knowledge of the data warehouse.",
    tools=[DatabaseQueryTool()],
    llm="gemini/gemini-2.0-flash",
)
```

---

### LangGraph — State Machine Orchestration for Agents

**LangGraph** provides explicit control flow for agent workflows using graph-based state machines. It's ideal when you need deterministic, testable, and debuggable agent behavior.

Official docs: `https://docs.langchain.com/oss/python/langgraph/overview`

#### LangGraph core concepts

| Concept          | Description                                                               |
| ---------------- | ------------------------------------------------------------------------- |
| **StateGraph**   | Defines the workflow as a graph of nodes and edges                        |
| **State**        | TypedDict that flows through the graph; nodes read/write to it            |
| **Node**         | A function that takes state, does work, returns updated state             |
| **Edge**         | Connection between nodes; can be conditional                              |
| **Checkpointer** | Persistence layer for state (enables pause/resume, time-travel debugging) |

#### LangGraph: Minimal workflow (aligned with upstream README)

```python
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

class State(TypedDict):
    text: str

def node_a(state: State) -> dict:
    return {"text": state["text"] + "a"}

def node_b(state: State) -> dict:
    return {"text": state["text"] + "b"}

graph = StateGraph(State)
graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)
graph.add_edge(START, "node_a")
graph.add_edge("node_a", "node_b")

app = graph.compile()
print(app.invoke({"text": ""}))  # {'text': 'ab'}
```

#### LangGraph: Multi-agent supervisor pattern (conceptual; see docs for the latest routing APIs)

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator

class MultiAgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str
    research_output: str
    writing_output: str
    final_output: str

# Specialist LLMs (could be different models)
supervisor_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
researcher_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
writer_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def supervisor_node(state: MultiAgentState) -> MultiAgentState:
    """Supervisor decides which agent to call next."""
    system = """You are a supervisor managing a research and writing team.
    Based on the current state, decide the next step:
    - 'researcher': if we need more research
    - 'writer': if research is complete and we need writing
    - 'FINISH': if the task is complete

    Respond with just the agent name or FINISH."""

    messages = [SystemMessage(content=system)] + state["messages"]
    response = supervisor_llm.invoke(messages)
    next_agent = response.content.strip().lower()

    return {"next_agent": next_agent, "messages": [response]}

def researcher_node(state: MultiAgentState) -> MultiAgentState:
    """Research agent gathers information."""
    system = """You are a research specialist. Gather comprehensive information
    on the topic and provide a detailed research summary."""

    messages = [SystemMessage(content=system)] + state["messages"]
    response = researcher_llm.invoke(messages)

    return {
        "research_output": response.content,
        "messages": [response]
    }

def writer_node(state: MultiAgentState) -> MultiAgentState:
    """Writer agent creates content based on research."""
    system = f"""You are a skilled writer. Using this research:

    {state.get('research_output', 'No research yet')}

    Create a well-structured, engaging piece of content."""

    messages = [SystemMessage(content=system)] + state["messages"]
    response = writer_llm.invoke(messages)

    return {
        "writing_output": response.content,
        "final_output": response.content,
        "messages": [response]
    }

def route_supervisor(state: MultiAgentState) -> Literal["researcher", "writer", "end"]:
    """Route based on supervisor decision."""
    next_agent = state.get("next_agent", "").lower()
    if "research" in next_agent:
        return "researcher"
    elif "writ" in next_agent:
        return "writer"
    else:
        return "end"

# Build multi-agent graph
multi_graph = StateGraph(MultiAgentState)

# Add nodes
multi_graph.add_node("supervisor", supervisor_node)
multi_graph.add_node("researcher", researcher_node)
multi_graph.add_node("writer", writer_node)

# Set entry
multi_graph.set_entry_point("supervisor")

# Routing from supervisor
multi_graph.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {"researcher": "researcher", "writer": "writer", "end": END}
)

# After specialist work, go back to supervisor
multi_graph.add_edge("researcher", "supervisor")
multi_graph.add_edge("writer", "supervisor")

# Compile with memory (enables pause/resume)
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
multi_app = multi_graph.compile(checkpointer=memory)

# Run with thread_id for persistence
config = {"configurable": {"thread_id": "project-123"}}
result = multi_app.invoke(
    {
        "messages": [HumanMessage(content="Write an article about agentic RAG patterns")],
        "next_agent": "",
        "research_output": "",
        "writing_output": "",
        "final_output": ""
    },
    config
)
```

#### LangGraph: Human-in-the-loop (approval gates) (conceptual; see docs for interrupt/checkpoint APIs)

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class ApprovalState(TypedDict):
    messages: Annotated[list, operator.add]
    draft: str
    approved: bool
    feedback: str

def generate_draft(state: ApprovalState) -> ApprovalState:
    """Generate initial draft."""
    # ... generate content ...
    return {"draft": "Generated draft content...", "approved": False}

def human_review(state: ApprovalState) -> ApprovalState:
    """This node will pause for human input."""
    # LangGraph automatically pauses here when using interrupt_before
    # Human provides feedback via external mechanism
    return state

def revise_draft(state: ApprovalState) -> ApprovalState:
    """Revise based on feedback."""
    # ... revise using state["feedback"] ...
    return {"draft": f"Revised based on: {state['feedback']}"}

def route_after_review(state: ApprovalState) -> Literal["revise", "end"]:
    if state.get("approved"):
        return "end"
    return "revise"

# Build graph with human checkpoint
approval_graph = StateGraph(ApprovalState)
approval_graph.add_node("generate", generate_draft)
approval_graph.add_node("review", human_review)
approval_graph.add_node("revise", revise_draft)

approval_graph.set_entry_point("generate")
approval_graph.add_edge("generate", "review")
approval_graph.add_conditional_edges("review", route_after_review, {"revise": "revise", "end": END})
approval_graph.add_edge("revise", "review")

# Compile with interrupt for human review
approval_app = approval_graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["review"]  # Pause before human review
)
```

---

### Framework Comparison: When to Use What

| Framework      | Best For                                      | Strengths                                    | Considerations                        |
| -------------- | --------------------------------------------- | -------------------------------------------- | ------------------------------------- |
| **ADK**        | Google Cloud production, Vertex AI deployment | Native GCP integration, managed scaling      | Google ecosystem focus                |
| **CrewAI**     | Role-based teams, creative workflows          | Easy multi-agent setup, role/goal paradigm   | Less fine-grained control than graphs |
| **LangGraph**  | Complex workflows, deterministic control      | Explicit state machine, debuggable, testable | More boilerplate than CrewAI          |
| **AutoGen**    | Research, conversational multi-agent          | Strong conversation patterns                 | Can be complex for simple use cases   |
| **LlamaIndex** | RAG-first applications                        | Best-in-class data connectors, indexing      | Less focus on agent orchestration     |

**EXAM TIP:** The exam tests concepts, not specific framework syntax. Understand the **patterns** (ReAct, routing, supervisor, agentic RAG) rather than memorizing API calls.

---

## 7.5.4 Context Management, Memory & Session State for Agents

Agents need **memory** to maintain coherent conversations, learn from interactions, and handle multi-turn tasks. This section covers the key concepts and implementation patterns.

### Why Context Management Matters

| Without Memory               | With Memory                          |
| ---------------------------- | ------------------------------------ |
| Agent forgets previous turns | Agent remembers conversation history |
| User must repeat context     | Agent builds on prior context        |
| No personalization           | Agent learns user preferences        |
| Each request is isolated     | Multi-step tasks work correctly      |
| No state across sessions     | User can resume conversations        |

### Types of Agent Memory

| Memory Type                | What It Stores                | Lifespan        | Use Case                           |
| -------------------------- | ----------------------------- | --------------- | ---------------------------------- |
| **Short-term (Working)**   | Current conversation turns    | Single session  | Multi-turn chat, task context      |
| **Long-term (Persistent)** | Facts, preferences, summaries | Across sessions | User profiles, learned preferences |
| **Episodic**               | Specific past interactions    | Across sessions | "Remember when we discussed X?"    |
| **Semantic**               | General knowledge, facts      | Permanent       | Domain knowledge, company info     |
| **Procedural**             | Learned workflows, patterns   | Permanent       | Task-specific learned behaviors    |

### Context Window Management

LLMs have finite context windows. As conversations grow, you must decide what to keep and what to drop.

#### Strategies for managing context

| Strategy               | How It Works                       | Pros                | Cons                  |
| ---------------------- | ---------------------------------- | ------------------- | --------------------- |
| **Sliding window**     | Keep last N messages               | Simple, predictable | Loses early context   |
| **Summarization**      | Periodically summarize older turns | Compresses history  | Loses details         |
| **Token budget**       | Keep messages until token limit    | Maximizes context   | Sudden drops          |
| **Importance scoring** | Keep "important" messages longer   | Preserves key info  | Complexity            |
| **Hybrid**             | Recent + summary + key facts       | Best of both        | Implementation effort |

#### Token-aware context management (Python example)

```python
from transformers import AutoTokenizer

class ContextManager:
    def __init__(self, model_name: str, max_tokens: int = 8000, reserve_for_response: int = 1000):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_context_tokens = max_tokens - reserve_for_response
        self.messages = []
        self.summary = ""

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim_if_needed()

    def _trim_if_needed(self):
        """Trim oldest messages if over token budget."""
        while self._total_tokens() > self.max_context_tokens and len(self.messages) > 2:
            # Keep system message (index 0) and most recent
            removed = self.messages.pop(1)
            # Optionally: add to summary instead of discarding
            self._update_summary(removed)

    def _total_tokens(self) -> int:
        total = self.count_tokens(self.summary) if self.summary else 0
        for msg in self.messages:
            total += self.count_tokens(msg["content"])
        return total

    def _update_summary(self, removed_message: dict):
        """Update running summary with removed content."""
        # In production: use LLM to summarize
        if self.summary:
            self.summary += f"\n- {removed_message['role']}: {removed_message['content'][:100]}..."
        else:
            self.summary = f"Earlier context:\n- {removed_message['role']}: {removed_message['content'][:100]}..."

    def get_messages_for_llm(self) -> list[dict]:
        """Return messages formatted for LLM, including summary if present."""
        result = []
        if self.summary:
            result.append({"role": "system", "content": f"Summary of earlier conversation:\n{self.summary}"})
        result.extend(self.messages)
        return result
```

### Session Management Patterns

**Session** = a logical conversation boundary (may span multiple requests, may persist across time).

#### Session lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                        SESSION LIFECYCLE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CREATE          UPDATE           PERSIST         RESUME        │
│  ───────         ──────           ───────         ──────        │
│  New session     Add messages     Save to DB      Load from DB  │
│  Generate ID     Update state     Serialize       Deserialize   │
│  Init context    Track metadata   Checkpoint      Restore state │
│                                                                 │
│                        EXPIRE / DELETE                          │
│                        ──────────────                           │
│                        TTL-based cleanup                        │
│                        User-requested deletion                  │
│                        GDPR/compliance                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Session state schema (typical structure)

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class SessionState:
    # Identity
    session_id: str
    user_id: str | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None

    # Conversation history
    messages: list[dict] = field(default_factory=list)
    summary: str = ""

    # Agent state
    current_task: str | None = None
    pending_actions: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)

    # Memory / learned context
    user_preferences: dict = field(default_factory=dict)
    facts: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Implementation: Session Management with ADK (conceptual)

```python
# NOTE: Treat this as a reference architecture, not exact imports.
# Verify current ADK session APIs in the official docs:
# - https://google.github.io/adk-docs/

session_store = make_session_store(backend="firestore", ttl_hours=24)

session_id = "user-123-session-456"
user_id = "user-123"
session = session_store.get_or_create(session_id=session_id, user_id=user_id)

agent = {"name": "assistant", "model": "gemini-2.0-flash", "instruction": "Use conversation history."}

run(agent, "My name is Alice and I work on ML pipelines", session=session)
reply = run(agent, "What did I say my job was?", session=session)
print(reply)

session_store.save(session)
```

### Implementation: Session Management with LangGraph (conceptual)

LangGraph uses **checkpointers** for state persistence.

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

# State includes conversation history
class ConversationState(TypedDict):
    messages: Annotated[list, operator.add]
    user_id: str
    session_metadata: dict

# LangGraph supports checkpointing for persistence; specific backends and import paths evolve.
# Follow the official docs for current options and examples:
# https://docs.langchain.com/oss/python/langgraph/overview
checkpointer = make_checkpointer(backend="sqlite", path="sessions.db")

# Build graph (simplified)
def chat_node(state: ConversationState) -> ConversationState:
    # Access full history via state["messages"]
    history = state["messages"]
    # ... process with LLM ...
    return {"messages": [response]}

graph = StateGraph(ConversationState)
graph.add_node("chat", chat_node)
graph.set_entry_point("chat")
graph.add_edge("chat", END)

# Compile with checkpointer
app = graph.compile(checkpointer=checkpointer)

# Run with thread_id (= session_id)
config = {"configurable": {"thread_id": "session-abc-123"}}

# Turn 1
result1 = app.invoke(
    {"messages": [HumanMessage("I'm working on a RAG system")], "user_id": "alice", "session_metadata": {}},
    config
)

# Turn 2 (automatically has history from checkpointer)
result2 = app.invoke(
    {"messages": [HumanMessage("What architecture should I use?")]},
    config
)

# Get full state at any point
state_snapshot = app.get_state(config)
print(state_snapshot.values["messages"])  # Full history

# Time-travel: get state at a specific checkpoint
history = list(app.get_state_history(config))
past_state = history[2]  # Third checkpoint
```

### Implementation: Memory with LangChain

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    VectorStoreRetrieverMemory
)
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Option 1: Buffer memory (keep all messages)
buffer_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Option 2: Window memory (keep last K turns)
window_memory = ConversationBufferWindowMemory(
    k=10,  # Keep last 10 exchanges
    memory_key="chat_history",
    return_messages=True
)

# Option 3: Summary memory (compress older turns)
summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

# Option 4: Vector store memory (semantic retrieval of past turns)
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./memory_db")

vector_memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    memory_key="relevant_history"
)

# Use with chain
from langchain.chains import ConversationChain

chain = ConversationChain(
    llm=llm,
    memory=summary_memory,  # Or any memory type
    verbose=True
)

# Conversation with automatic memory
response1 = chain.predict(input="My team is building a customer support bot")
response2 = chain.predict(input="What frameworks would you recommend?")  # Has context
```

### Long-Term Memory Patterns

For agents that need to remember across sessions (user preferences, facts, past interactions):

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LongTermMemory:
    """Persistent memory that survives across sessions."""

    user_id: str

    # User profile (learned over time)
    preferences: dict  # {"response_style": "concise", "expertise_level": "expert"}

    # Factual memory (things the user told us)
    facts: list[dict]  # [{"fact": "Works at Acme Corp", "confidence": 0.95, "source": "user_statement"}]

    # Episodic memory (past significant interactions)
    episodes: list[dict]  # [{"summary": "Helped debug RAG pipeline", "date": "2024-01-15", "outcome": "resolved"}]

    # Procedural memory (learned workflows for this user)
    procedures: list[dict]  # [{"task": "code_review", "steps": [...], "preferences": {...}}]


class LongTermMemoryStore:
    """Store and retrieve long-term memories."""

    def __init__(self, vectorstore, db_client):
        self.vectorstore = vectorstore  # For semantic search
        self.db = db_client  # For structured storage

    def add_fact(self, user_id: str, fact: str, source: str = "conversation"):
        """Store a learned fact about the user."""
        # Embed and store for semantic retrieval
        self.vectorstore.add_texts(
            texts=[fact],
            metadatas=[{"user_id": user_id, "type": "fact", "source": source}]
        )
        # Also store structured
        self.db.facts.insert({"user_id": user_id, "fact": fact, "source": source, "timestamp": datetime.utcnow()})

    def recall_relevant(self, user_id: str, query: str, k: int = 5) -> list[str]:
        """Retrieve memories relevant to current query."""
        results = self.vectorstore.similarity_search(
            query,
            k=k,
            filter={"user_id": user_id}
        )
        return [doc.page_content for doc in results]

    def get_user_profile(self, user_id: str) -> dict:
        """Get structured user preferences."""
        return self.db.profiles.find_one({"user_id": user_id}) or {}
```

### Production Session Management Checklist

| Concern              | Solution                                                              |
| -------------------- | --------------------------------------------------------------------- |
| **Persistence**      | Use durable storage (Firestore, PostgreSQL, Redis) not just in-memory |
| **Scalability**      | Stateless app servers + external session store                        |
| **TTL / Expiration** | Auto-expire old sessions (24h–7d typical)                             |
| **Privacy / GDPR**   | Allow users to delete their data; don't log PII unnecessarily         |
| **Security**         | Encrypt session data at rest; validate session ownership              |
| **Concurrency**      | Handle multiple requests to same session (locking or last-write-wins) |
| **Recovery**         | Checkpointing enables resume after crashes                            |
| **Observability**    | Log session lifecycle events; track memory usage                      |

### Vertex AI Agent Engine Session Management (conceptual)

Agent Engine provides a managed runtime where sessions are typically identified by a **session ID** and the platform persists conversation state across turns. The exact client API is evolving, so treat the following as conceptual:

```python
# Pseudocode: shape varies by SDK version / preview surface.
session_id = "user-alice-project-xyz"

resp1 = agent_engine.query("Hello, I'm starting a new project", session_id=session_id)
resp2 = agent_engine.query("What did we discuss earlier?", session_id=session_id)
print(resp2.text)
```

Use the official Vertex AI documentation for the current SDK surface and best practices around retention/TTL, privacy, and observability.

**EXAM TIP:** When questions mention "conversation history", "multi-turn", "remember previous context", or "resume conversation" → think **session management + memory patterns**.

---

## 7.5.5 GenAI & Agent Production Deployment

Moving agents from prototype to production requires treating them as **distributed systems** with rigorous engineering practices. This section covers the architectural patterns, security controls, and operational discipline needed for production-grade agent deployments.

Source: [Architecting efficient context-aware multi-agent framework for production](https://developers.googleblog.com/architecting-efficient-context-aware-multi-agent-framework-for-production/)

---

### Context Engineering: The Scaling Bottleneck

As agents run longer, the information they track—chat history, tool outputs, documents, reasoning—**explodes**. Simply using larger context windows is not a scaling strategy.

#### The three-way pressure on context

| Pressure                                      | Problem                                                                                                            |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Cost & latency spirals**                    | Model cost and time-to-first-token grow with context size; "shoveling" raw history makes agents slow and expensive |
| **Signal degradation ("lost in the middle")** | Irrelevant logs, stale tool outputs distract the model from the immediate instruction                              |
| **Physical limits**                           | RAG results, artifacts, and conversation traces eventually overflow even the largest windows                       |

#### The ADK thesis: context as a compiled view

Instead of treating context as a mutable string buffer, ADK treats **context as a compiled view over a richer stateful system**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTEXT COMPILATION PIPELINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   SOURCES                    COMPILER                    OUTPUT     │
│   ───────                    ────────                    ──────     │
│   Session (events)    →      Flows &          →      Working Context │
│   Memory (long-term)  →      Processors       →      (per-call view) │
│   Artifacts (files)   →      (ordered list)   →                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Three design principles

| Principle                              | Description                                                                             |
| -------------------------------------- | --------------------------------------------------------------------------------------- |
| **Separate storage from presentation** | Durable state (Sessions) vs per-call views (Working Context) evolve independently       |
| **Explicit transformations**           | Context is built through named, ordered processors—observable and testable              |
| **Scope by default**                   | Every model call sees the **minimum context required**; agents reach for more via tools |

#### The tiered context model

| Layer               | Purpose                                                       | Lifecycle                             |
| ------------------- | ------------------------------------------------------------- | ------------------------------------- |
| **Working Context** | Immediate prompt for this model call                          | Ephemeral (thrown away after call)    |
| **Session**         | Durable log of events (messages, tool calls, results)         | Per-conversation                      |
| **Memory**          | Long-lived searchable knowledge (preferences, past decisions) | Cross-session                         |
| **Artifacts**       | Large binary/text data (files, logs, images)                  | Addressed by name/version, not pasted |

#### Flows and processors: the compilation pipeline

```python
# Simplified ADK SingleFlow processor pipeline (conceptual)
self.request_processors = [
    basic.request_processor,
    auth_preprocessor.request_processor,
    request_confirmation.request_processor,    # HITL confirmation
    instructions.request_processor,            # System prompt
    identity.request_processor,                # Agent identity
    contents.request_processor,                # Conversation history
    context_cache_processor.request_processor, # Prefix caching
    planning.request_processor,                # Task decomposition
    code_execution.request_processor,          # Code sandbox
    output_schema_processor.request_processor, # Structured outputs
]

self.response_processors = [
    planning.response_processor,
    code_execution.response_processor,
]
```

**Key insight**: You're no longer rewriting "prompt templates"—you're adding/reordering processors in a pipeline.

#### Multi-agent context scoping

When a root agent invokes sub-agents, you must prevent **context explosion**:

| Pattern             | Description                                                   | Context Scope                                                |
| ------------------- | ------------------------------------------------------------- | ------------------------------------------------------------ |
| **Agents as Tools** | Sub-agent is a function: call with focused prompt, get result | Callee sees only specific instructions + necessary artifacts |
| **Agent Transfer**  | Control handed off to sub-agent to continue conversation      | Sub-agent inherits a configurable view over the Session      |

**Handoff modes**:

- **Full mode**: Pass full contents of caller's working context (useful when sub-agent needs entire history)
- **None mode**: Sub-agent sees no prior history; only receives new prompt you construct

**Translation during handoff**: Foundation models see `system`/`user`/`assistant` roles, not "Agent A" vs "Agent B". ADK **recasts** prior assistant messages as narrative context and **attributes** tool calls to avoid confusion.

---

### Distributed Systems for Agents

Production agents are distributed systems with unique challenges.

#### Agent deployment topology

```
                         ┌─────────────────┐
                         │   Load Balancer │
                         │  (Cloud LB / Kong / Envoy)
                         └────────┬────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
    ┌───────────┐           ┌───────────┐           ┌───────────┐
    │  Agent    │           │  Agent    │           │  Agent    │
    │  Instance │           │  Instance │           │  Instance │
    │  (Pod 1)  │           │  (Pod 2)  │           │  (Pod N)  │
    └─────┬─────┘           └─────┬─────┘           └─────┬─────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
    ┌───────────┐           ┌───────────┐           ┌───────────┐
    │  Session  │           │  Vector   │           │   Model   │
    │   Store   │           │   Store   │           │  Gateway  │
    │ (Redis/   │           │ (pgvector │           │ (Vertex/  │
    │  Firestore)│          │  /Milvus) │           │  OpenAI)  │
    └───────────┘           └───────────┘           └───────────┘
```

#### Key distributed systems concerns

| Concern                | Challenge                                                            | Solution                                                          |
| ---------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **Statefulness**       | Agents have multi-turn state; requests must route to correct session | External session store + session ID in request; stateless compute |
| **Idempotency**        | Tool calls may be retried; side effects must be safe                 | Idempotency keys; deduplication at tool level                     |
| **Consistency**        | Multiple agent instances may modify same session                     | Optimistic locking or append-only event logs                      |
| **Timeouts**           | LLM calls can take 10-60+ seconds                                    | Async processing; long-poll or SSE/WebSocket for clients          |
| **Cascading failures** | Agent calls LLM → LLM calls tool → tool calls external API           | Circuit breakers; bulkheads; graceful degradation                 |

#### Resilience patterns

```python
# Pseudocode: resilience in agent tool calls
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from circuitbreaker import circuit

class ToolExecutor:
    @circuit(failure_threshold=5, recovery_timeout=30)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    async def execute_tool(self, tool_name: str, args: dict, idempotency_key: str):
        # Check if already executed (idempotency)
        cached = await self.cache.get(f"tool:{idempotency_key}")
        if cached:
            return cached

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                self.tools[tool_name].run(**args),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise ToolTimeoutError(f"Tool {tool_name} timed out")

        # Cache result for idempotency
        await self.cache.set(f"tool:{idempotency_key}", result, ttl=3600)
        return result
```

---

### Security & Guardrails

Agent systems expand the attack surface beyond traditional APIs.

#### Threat model for agents

| Threat                   | Description                                | Mitigation                                                  |
| ------------------------ | ------------------------------------------ | ----------------------------------------------------------- |
| **Prompt injection**     | Malicious input manipulates agent behavior | Input sanitization; instruction hierarchy; output filtering |
| **Tool misuse**          | Agent calls tools with harmful parameters  | Tool-level validation; allowlists; rate limits              |
| **Data exfiltration**    | Agent leaks sensitive data via outputs     | Output filtering; PII detection; audit logging              |
| **Privilege escalation** | Agent gains access beyond intended scope   | Least-privilege tool permissions; scoped credentials        |
| **Model theft**          | Extraction of prompts/fine-tuning data     | Rate limiting; output watermarking; monitoring              |
| **Denial of service**    | Expensive operations exhaust resources     | Cost caps; token budgets; circuit breakers                  |

#### Defense in depth architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SECURITY LAYERS                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐                                                    │
│  │   INPUT     │  • PII detection (Cloud DLP)                       │
│  │   FILTER    │  • Prompt injection detection                       │
│  │             │  • Rate limiting per user/session                   │
│  └──────┬──────┘                                                    │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────┐                                                    │
│  │   AGENT     │  • Instruction hierarchy (system > user)            │
│  │   CORE      │  • Tool permission boundaries                       │
│  │             │  • Token/cost budgets per request                   │
│  └──────┬──────┘                                                    │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────┐                                                    │
│  │   TOOL      │  • Input validation per tool                        │
│  │   LAYER     │  • Scoped credentials (short-lived tokens)          │
│  │             │  • Audit logging of all calls                       │
│  └──────┬──────┘                                                    │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────┐                                                    │
│  │   OUTPUT    │  • Content safety filters                           │
│  │   FILTER    │  • PII redaction                                    │
│  │             │  • Grounding verification                           │
│  └─────────────┘                                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Guardrails implementation

```python
# Pseudocode: guardrails pipeline
from typing import Literal
from pydantic import BaseModel

class GuardrailResult(BaseModel):
    allowed: bool
    reason: str | None = None
    modified_content: str | None = None

class GuardrailsPipeline:
    def __init__(self):
        self.input_guardrails = [
            PIIDetector(),
            PromptInjectionDetector(),
            ContentSafetyFilter(),
            RateLimiter(),
        ]
        self.output_guardrails = [
            PIIRedactor(),
            GroundingVerifier(),
            ContentSafetyFilter(),
            CostTracker(),
        ]

    async def check_input(self, user_input: str, session_id: str) -> GuardrailResult:
        for guardrail in self.input_guardrails:
            result = await guardrail.check(user_input, session_id)
            if not result.allowed:
                return result
        return GuardrailResult(allowed=True)

    async def check_output(self, agent_output: str, context: dict) -> GuardrailResult:
        for guardrail in self.output_guardrails:
            result = await guardrail.check(agent_output, context)
            if not result.allowed:
                return result
            if result.modified_content:
                agent_output = result.modified_content
        return GuardrailResult(allowed=True, modified_content=agent_output)

# Tool-level security
class SecureTool:
    def __init__(self, name: str, allowed_actions: list[str], credential_scope: str):
        self.name = name
        self.allowed_actions = allowed_actions
        self.credential_scope = credential_scope

    async def execute(self, action: str, params: dict, auth_context: dict):
        # Validate action is allowed
        if action not in self.allowed_actions:
            raise PermissionError(f"Action {action} not allowed for tool {self.name}")

        # Get scoped credential (short-lived)
        credential = await self.get_scoped_credential(self.credential_scope, auth_context)

        # Audit log
        await self.audit_log(action, params, auth_context)

        # Execute with scoped credential
        return await self._execute_internal(action, params, credential)
```

#### Google Cloud Armor for agent APIs

Cloud Armor provides WAF and DDoS protection for agent endpoints:

```yaml
# Cloud Armor security policy (conceptual)
securityPolicy:
  name: agent-api-policy
  rules:
    - action: deny(403)
      priority: 1000
      match:
        expr:
          expression: "evaluatePreconfiguredExpr('xss-stable')"
      description: 'Block XSS attacks'

    - action: deny(403)
      priority: 1001
      match:
        expr:
          expression: "evaluatePreconfiguredExpr('sqli-stable')"
      description: 'Block SQL injection'

    - action: rate_based_ban
      priority: 2000
      match:
        config:
          intervalSec: 60
          conformAction: allow
          exceedAction: deny(429)
          rateLimitThreshold:
            count: 100 # 100 requests per minute per IP
      description: 'Rate limit per IP'

    - action: throttle
      priority: 3000
      match:
        versionedExpr: SRC_IPS_V1
      rateLimitOptions:
        conformAction: allow
        exceedAction: deny(429)
        enforceOnKey: HTTP_HEADER
        enforceOnKeyName: 'X-Session-ID'
        rateLimitThreshold:
          count: 20 # 20 requests per session per minute
      description: 'Rate limit per session'
```

#### Secret Manager for Agent Credentials

Agents often need credentials to access tools, APIs, and data sources. **Never put secrets in prompts or code** — use Secret Manager.

Source: [Confidential Applications on Google Cloud](https://developers.googleblog.com/dont-trust-verify-building-end-to-end-confidential-applications-on-google-cloud/)

```python
# Pseudocode: Secure credential access for agent tools
from google.cloud import secretmanager

class SecretProvider:
    """Fetch secrets from Secret Manager with caching and rotation awareness."""
    
    def __init__(self, project_id: str):
        self.client = secretmanager.SecretManagerServiceClient()
        self.project_id = project_id
        self._cache = {}
    
    def get_secret(self, secret_id: str, version: str = "latest") -> str:
        """Get secret value; use 'latest' for auto-rotation."""
        cache_key = f"{secret_id}:{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        name = f"projects/{self.project_id}/secrets/{secret_id}/versions/{version}"
        response = self.client.access_secret_version(request={"name": name})
        secret_value = response.payload.data.decode("UTF-8")
        
        # Cache for short duration (secrets can rotate)
        self._cache[cache_key] = secret_value
        return secret_value
    
    def get_tool_credential(self, tool_name: str) -> dict:
        """Get credentials for a specific tool."""
        return {
            "api_key": self.get_secret(f"{tool_name}-api-key"),
            "endpoint": self.get_secret(f"{tool_name}-endpoint"),
        }

# Usage in agent tool
class DatabaseTool:
    def __init__(self, secret_provider: SecretProvider):
        self.secrets = secret_provider
    
    async def execute(self, query: str):
        # Get fresh credentials (supports rotation)
        creds = self.secrets.get_tool_credential("database")
        
        # Connect with credentials from Secret Manager
        conn = await connect(
            host=creds["endpoint"],
            password=creds["api_key"]
        )
        return await conn.execute(query)
```

**Best practices for Secret Manager with agents**:

| Practice | Description |
|----------|-------------|
| **Use `latest` version** | Enables automatic secret rotation without code changes |
| **Short TTL caching** | Cache secrets for minutes, not hours (rotation awareness) |
| **Separate secrets per environment** | Different secrets for dev/staging/prod |
| **Audit access** | Enable Cloud Audit Logs for secret access |
| **Principle of least privilege** | Grant `secretAccessor` role only to service accounts that need it |

#### IAM Policies for Agent Systems

Agents require careful IAM design to prevent privilege escalation.

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    IAM ARCHITECTURE FOR AGENTS                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────────┐          ┌─────────────────────┐                  │
│  │   User Identity     │          │   Agent Service     │                  │
│  │   (end user)        │          │   Account           │                  │
│  └──────────┬──────────┘          └──────────┬──────────┘                  │
│             │                                │                             │
│             │ authenticates                  │ runs as                     │
│             ▼                                ▼                             │
│  ┌─────────────────────────────────────────────────────────┐               │
│  │                    AGENT API                             │               │
│  │  • Validates user identity                               │               │
│  │  • Maps user → allowed tools/scopes                      │               │
│  │  • Enforces user-level quotas                            │               │
│  └──────────┬──────────────────────────────────┬───────────┘               │
│             │                                  │                            │
│             │ user context                     │ agent identity             │
│             ▼                                  ▼                            │
│  ┌─────────────────────┐          ┌─────────────────────┐                  │
│  │   Tool A            │          │   Tool B            │                  │
│  │   (user's data)     │          │   (shared service)  │                  │
│  │                     │          │                     │                  │
│  │   IAM: user impersonation    │   IAM: service account │                │
│  └─────────────────────┘          └─────────────────────┘                  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**IAM patterns for agents**:

| Pattern | Use Case | Implementation |
|---------|----------|----------------|
| **Service account per agent** | Isolate agent permissions | Each agent type has dedicated SA with minimal roles |
| **Workload Identity** | GKE-based agents | Map K8s SA → GCP SA; no key files |
| **Short-lived credentials** | Tool access | Use `generateAccessToken` for 1-hour tokens |
| **User impersonation** | Access user's data | Agent acts on behalf of user with their permissions |
| **VPC Service Controls** | Data exfiltration prevention | Restrict which APIs can be called from agent VPC |

```python
# Pseudocode: IAM-aware tool execution
from google.auth import impersonated_credentials
from google.auth import default

class IAMAwareToolExecutor:
    def __init__(self):
        # Agent's base credentials
        self.agent_credentials, self.project = default()
    
    def get_user_impersonated_credentials(self, user_email: str, scopes: list[str]):
        """Get credentials that act as the user (requires domain-wide delegation or user consent)."""
        target_principal = f"user:{user_email}"
        
        # Create impersonated credentials
        impersonated = impersonated_credentials.Credentials(
            source_credentials=self.agent_credentials,
            target_principal=target_principal,
            target_scopes=scopes,
            lifetime=3600  # 1 hour
        )
        return impersonated
    
    def get_scoped_service_credentials(self, target_service_account: str, scopes: list[str]):
        """Get short-lived credentials for a specific service account."""
        impersonated = impersonated_credentials.Credentials(
            source_credentials=self.agent_credentials,
            target_principal=target_service_account,
            target_scopes=scopes,
            lifetime=3600
        )
        return impersonated
    
    async def execute_tool_as_user(self, tool: str, params: dict, user_context: dict):
        """Execute tool with user's permissions."""
        user_email = user_context["email"]
        allowed_scopes = user_context["allowed_scopes"]
        
        # Get user-scoped credentials
        user_creds = self.get_user_impersonated_credentials(user_email, allowed_scopes)
        
        # Execute tool with user credentials
        return await self.tools[tool].execute(params, credentials=user_creds)
```

#### Apigee + GKE Inference Gateway for LLM Policies

Source: [Apigee Operator for Kubernetes and GKE Inference Gateway Integration](https://developers.googleblog.com/apigee-operator-for-kubernetes-and-gke-inference-gateway-integration-for-auth-and-aillm-policies/)

For enterprise agent deployments, **Apigee** provides API management capabilities specifically designed for LLM traffic:

```yaml
# Apigee policy for LLM inference (conceptual)
apiVersion: apigee.cloud.google.com/v1alpha1
kind: APIProduct
metadata:
  name: agent-api-product
spec:
  displayName: "Agent API"
  approvalType: auto
  attributes:
    - name: llm-policies
      value: "enabled"
  
  # Rate limiting per developer/app
  quota:
    limit: 1000
    interval: 1
    timeUnit: day
  
  # LLM-specific policies
  llmPolicies:
    # Token budget per request
    maxInputTokens: 32000
    maxOutputTokens: 8000
    
    # Model allowlist
    allowedModels:
      - gemini-2.0-flash
      - gemini-2.0-pro
    
    # Content safety
    contentFiltering: strict
    
    # Cost tracking
    costTracking:
      enabled: true
      alertThreshold: 100.00  # USD per day
```

**What Apigee + GKE Inference Gateway provides**:

| Capability | Description |
|------------|-------------|
| **OAuth/API key auth** | Standard API authentication before reaching agent |
| **Rate limiting** | Per-developer, per-app, per-endpoint quotas |
| **Token budgets** | Enforce max input/output tokens per request |
| **Model allowlisting** | Restrict which models can be called |
| **Cost tracking** | Real-time cost monitoring and alerts |
| **Content filtering** | Apply safety policies at the gateway level |
| **Audit logging** | Full request/response logging for compliance |

#### VPC Service Controls for Data Protection

Prevent data exfiltration by restricting which APIs agents can access:

```yaml
# VPC Service Controls perimeter (conceptual)
accessPolicy:
  name: "agent-data-perimeter"
  
  servicePerimeters:
    - name: "agent-perimeter"
      perimeterType: PERIMETER_TYPE_REGULAR
      
      # Resources inside the perimeter
      resources:
        - "projects/my-agent-project"
      
      # Restricted services (agent can only call these)
      restrictedServices:
        - "aiplatform.googleapis.com"      # Vertex AI
        - "bigquery.googleapis.com"         # BigQuery
        - "storage.googleapis.com"          # Cloud Storage
        - "secretmanager.googleapis.com"    # Secret Manager
      
      # Block external APIs
      vpcAccessibleServices:
        enableRestriction: true
        allowedServices:
          - "RESTRICTED-SERVICES"
      
      # Ingress rules (who can access the perimeter)
      ingressPolicies:
        - ingressFrom:
            identityType: ANY_SERVICE_ACCOUNT
            sources:
              - resource: "projects/my-agent-project"
          ingressTo:
            operations:
              - serviceName: "aiplatform.googleapis.com"
                methodSelectors:
                  - method: "*"
```

#### Confidential Computing for Sensitive AI

Source: [Building End-to-End Confidential Applications on Google Cloud](https://developers.googleblog.com/dont-trust-verify-building-end-to-end-confidential-applications-on-google-cloud/)

For highly sensitive workloads (healthcare, finance), **Confidential Computing** encrypts data in use:

| Feature | Description |
|---------|-------------|
| **Confidential VMs** | Memory encrypted with AMD SEV or Intel TDX |
| **Confidential GKE Nodes** | Run agent containers in confidential VMs |
| **Attestation** | Cryptographic proof that code runs in trusted environment |
| **Use case** | Process sensitive data without exposing to cloud provider |

```yaml
# Confidential GKE node pool (conceptual)
apiVersion: container.cnrm.cloud.google.com/v1beta1
kind: ContainerNodePool
metadata:
  name: confidential-agent-pool
spec:
  nodeConfig:
    machineType: n2d-standard-4  # AMD EPYC (required for SEV)
    confidentialNodes:
      enabled: true
    shieldedInstanceConfig:
      enableSecureBoot: true
      enableIntegrityMonitoring: true
```

#### Agent Security Checklist (Google Cloud)

| Layer | Service | Purpose |
|-------|---------|---------|
| **Network** | Cloud Armor | WAF, DDoS protection, rate limiting |
| **Network** | VPC Service Controls | Data exfiltration prevention |
| **API Gateway** | Apigee | Auth, rate limiting, LLM-specific policies |
| **Identity** | IAM | Service accounts, workload identity, impersonation |
| **Secrets** | Secret Manager | API keys, credentials, certificates |
| **Audit** | Cloud Audit Logs | Who did what, when |
| **Data** | Cloud DLP | PII detection and redaction |
| **Compute** | Confidential VMs/GKE | Encryption in use for sensitive workloads |
| **Model** | Vertex AI safety settings | Content filtering at the model level |

**EXAM TIP:** When questions mention "secure agent deployment" or "enterprise agent architecture" → think **IAM (least privilege) + Secret Manager (no hardcoded creds) + Cloud Armor (WAF/rate limits) + VPC Service Controls (data perimeter) + audit logging**.

---

### CI/CD for AI Systems

Agent systems require specialized CI/CD practices beyond traditional software.

#### The AI/ML CI/CD pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CI/CD PIPELINE FOR AGENTS                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │  CODE    │ →  │  BUILD   │ →  │  TEST    │ →  │  EVAL    │         │
│  │  COMMIT  │    │  & LINT  │    │  (unit)  │    │  (AI)    │         │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘         │
│                                                                        │
│                        ↓                                               │
│                                                                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │  DEPLOY  │ ←  │  APPROVE │ ←  │  STAGING │ ←  │  SECURITY│         │
│  │  (prod)  │    │  (HITL)  │    │  (canary)│    │  SCAN    │         │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘         │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

#### What's different for AI systems

| Stage                 | Traditional CI/CD        | Agent CI/CD                                            |
| --------------------- | ------------------------ | ------------------------------------------------------ |
| **Unit tests**        | Deterministic assertions | Fuzzy assertions; eval metrics; LLM-as-judge           |
| **Integration tests** | API contracts            | Tool behavior; multi-turn scenarios                    |
| **Evaluation**        | Not applicable           | Prompt quality; retrieval accuracy; agent trajectories |
| **Security scan**     | Code vulnerabilities     | Prompt injection; jailbreak; data leakage              |
| **Staging**           | Feature testing          | A/B prompt experiments; shadow mode                    |
| **Deployment**        | Blue/green or rolling    | Canary with eval metrics; automatic rollback           |

#### Clean code practices for agents

```python
# ❌ BAD: Prompt as string concatenation
def bad_agent_prompt(user_query, context):
    return f"""You are a helpful assistant.
    Context: {context}
    User: {user_query}
    Please respond helpfully."""

# ✅ GOOD: Prompts as structured, versioned artifacts
from dataclasses import dataclass
from typing import Optional

@dataclass
class PromptConfig:
    version: str
    system_instruction: str
    few_shot_examples: list[dict]
    output_schema: Optional[dict] = None

    def render(self, context: dict, user_query: str) -> list[dict]:
        messages = [
            {"role": "system", "content": self.system_instruction},
        ]
        for example in self.few_shot_examples:
            messages.append({"role": "user", "content": example["user"]})
            messages.append({"role": "assistant", "content": example["assistant"]})

        if context:
            messages.append({"role": "user", "content": f"Context: {context}"})

        messages.append({"role": "user", "content": user_query})
        return messages

# Prompt versioning
PROMPTS = {
    "research_agent_v1": PromptConfig(
        version="1.0.0",
        system_instruction="""You are a research assistant...""",
        few_shot_examples=[...],
    ),
    "research_agent_v2": PromptConfig(
        version="2.0.0",
        system_instruction="""You are a research assistant with improved...""",
        few_shot_examples=[...],
    ),
}
```

#### Evaluation-driven CI/CD

```python
# Pseudocode: eval gate in CI/CD
from typing import Literal

class EvalResult:
    metric: str
    value: float
    threshold: float
    passed: bool

async def run_eval_suite(agent_config: dict, eval_dataset: list[dict]) -> list[EvalResult]:
    results = []

    # Retrieval quality (for RAG agents)
    retrieval_scores = await eval_retrieval(agent_config, eval_dataset)
    results.append(EvalResult(
        metric="retrieval_precision@5",
        value=retrieval_scores["precision@5"],
        threshold=0.7,
        passed=retrieval_scores["precision@5"] >= 0.7
    ))

    # Response quality (LLM-as-judge)
    response_scores = await eval_responses(agent_config, eval_dataset)
    results.append(EvalResult(
        metric="response_quality",
        value=response_scores["average"],
        threshold=4.0,  # out of 5
        passed=response_scores["average"] >= 4.0
    ))

    # Trajectory evaluation (for multi-step agents)
    trajectory_scores = await eval_trajectories(agent_config, eval_dataset)
    results.append(EvalResult(
        metric="trajectory_success_rate",
        value=trajectory_scores["success_rate"],
        threshold=0.8,
        passed=trajectory_scores["success_rate"] >= 0.8
    ))

    # Safety checks
    safety_scores = await eval_safety(agent_config, eval_dataset)
    results.append(EvalResult(
        metric="safety_pass_rate",
        value=safety_scores["pass_rate"],
        threshold=0.99,
        passed=safety_scores["pass_rate"] >= 0.99
    ))

    return results

def should_deploy(eval_results: list[EvalResult]) -> bool:
    return all(r.passed for r in eval_results)
```

---

### API Design for Agentic Calls

Agent APIs differ from traditional REST APIs in key ways.

#### Streaming vs request/response

| Pattern                      | Use Case                           | Implementation                         |
| ---------------------------- | ---------------------------------- | -------------------------------------- |
| **Request/response**         | Simple queries, internal tools     | Standard REST; wait for full response  |
| **Server-Sent Events (SSE)** | Chat UIs; token-by-token streaming | `text/event-stream`; partial responses |
| **WebSockets**               | Bidirectional; long-running tasks  | Real-time status updates; cancelation  |
| **Long-polling**             | Legacy clients; firewalls block WS | Repeated requests with timeout         |

#### Agent API design patterns

```python
# FastAPI example: streaming agent responses
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import asyncio
import json

app = FastAPI()

@app.post("/v1/agent/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    session_id = body.get("session_id")
    message = body.get("message")
    stream = body.get("stream", False)

    if stream:
        return EventSourceResponse(
            stream_agent_response(session_id, message),
            media_type="text/event-stream"
        )
    else:
        response = await run_agent(session_id, message)
        return {"response": response}

async def stream_agent_response(session_id: str, message: str):
    """Stream agent response as SSE events."""
    async for event in agent.stream(session_id, message):
        if event.type == "token":
            yield {"event": "token", "data": json.dumps({"text": event.text})}
        elif event.type == "tool_call":
            yield {"event": "tool_call", "data": json.dumps({
                "tool": event.tool_name,
                "args": event.args,
                "status": "started"
            })}
        elif event.type == "tool_result":
            yield {"event": "tool_result", "data": json.dumps({
                "tool": event.tool_name,
                "result": event.result,
                "status": "completed"
            })}
        elif event.type == "done":
            yield {"event": "done", "data": json.dumps({
                "full_response": event.full_response,
                "usage": event.usage
            })}

# Async task pattern for long-running agents
@app.post("/v1/agent/tasks")
async def create_task(request: Request):
    body = await request.json()
    task_id = await queue_agent_task(body)
    return {"task_id": task_id, "status": "queued"}

@app.get("/v1/agent/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = await get_task(task_id)
    return {
        "task_id": task_id,
        "status": task.status,  # queued, running, completed, failed
        "result": task.result if task.status == "completed" else None,
        "progress": task.progress,  # 0.0 - 1.0
    }
```

#### API versioning and deprecation

```python
# Version-aware routing
from fastapi import APIRouter, Header

v1_router = APIRouter(prefix="/v1")
v2_router = APIRouter(prefix="/v2")

@v1_router.post("/agent/chat")
async def chat_v1(request: Request):
    # Legacy behavior
    ...

@v2_router.post("/agent/chat")
async def chat_v2(request: Request):
    # New behavior (streaming by default, new response schema)
    ...

# Deprecation header
@app.middleware("http")
async def add_deprecation_header(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/v1/"):
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = "2026-06-01"
        response.headers["Link"] = '</v2/agent/chat>; rel="successor-version"'
    return response
```

---

### Traffic Management & Scaling

Agent workloads have unique scaling characteristics.

#### Characteristics of agent traffic

| Characteristic        | Impact                                         | Mitigation                                         |
| --------------------- | ---------------------------------------------- | -------------------------------------------------- |
| **High latency**      | 5-60+ seconds per request                      | Async processing; streaming; timeout handling      |
| **Variable cost**     | 100x difference between simple/complex queries | Cost prediction; tiered pricing; budgets           |
| **Bursty**            | Viral content can spike traffic 100x           | Auto-scaling; queue-based processing               |
| **Long connections**  | Streaming ties up connections                  | Connection pooling; horizontal scaling             |
| **Stateful sessions** | Must route to correct session state            | External session store; sticky sessions (cautious) |

#### Scaling architecture

```yaml
# Kubernetes HPA for agent workloads (conceptual)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-deployment
  minReplicas: 3
  maxReplicas: 100
  metrics:
    # Scale on concurrent requests (not CPU)
    - type: Pods
      pods:
        metric:
          name: http_requests_in_flight
        target:
          type: AverageValue
          averageValue: 10 # 10 concurrent requests per pod
    # Also consider queue depth
    - type: External
      external:
        metric:
          name: pubsub_subscription_num_undelivered_messages
          selector:
            matchLabels:
              subscription: agent-tasks-sub
        target:
          type: AverageValue
          averageValue: 50 # Scale up when queue grows
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300 # Slow scale-down (agent workloads are bursty)
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
```

#### Cost management

```python
# Cost-aware request handling
from dataclasses import dataclass

@dataclass
class CostEstimate:
    input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    tier: str  # "cheap", "standard", "expensive"

class CostAwareRouter:
    def __init__(self, budget_per_user_per_day: float = 1.0):
        self.budget = budget_per_user_per_day

    async def route_request(self, user_id: str, request: dict) -> str:
        # Estimate cost
        estimate = self.estimate_cost(request)

        # Check user's remaining budget
        spent_today = await self.get_user_spend(user_id)
        if spent_today + estimate.estimated_cost_usd > self.budget:
            raise BudgetExceededError("Daily budget exceeded")

        # Route to appropriate model tier
        if estimate.tier == "cheap":
            return "gemini-1.5-flash"
        elif estimate.tier == "standard":
            return "gemini-2.0-flash"
        else:
            return "gemini-2.0-pro"

    def estimate_cost(self, request: dict) -> CostEstimate:
        input_tokens = self.count_tokens(request)
        # Heuristic: output ~2x input for agents
        estimated_output = input_tokens * 2

        # Rough pricing (adjust for actual model)
        cost_per_1k_input = 0.00035
        cost_per_1k_output = 0.00105
        estimated_cost = (
            (input_tokens / 1000) * cost_per_1k_input +
            (estimated_output / 1000) * cost_per_1k_output
        )

        tier = "cheap" if estimated_cost < 0.01 else "standard" if estimated_cost < 0.10 else "expensive"

        return CostEstimate(
            input_tokens=input_tokens,
            estimated_output_tokens=estimated_output,
            estimated_cost_usd=estimated_cost,
            tier=tier
        )
```

#### Observability for agents

```python
# Key metrics to track
AGENT_METRICS = {
    # Latency
    "agent_latency_seconds": Histogram(
        buckets=[0.5, 1, 2, 5, 10, 30, 60, 120]
    ),
    "llm_call_latency_seconds": Histogram(),
    "tool_call_latency_seconds": Histogram(labels=["tool_name"]),

    # Throughput
    "agent_requests_total": Counter(labels=["status", "model"]),
    "tool_calls_total": Counter(labels=["tool_name", "status"]),

    # Cost
    "tokens_used_total": Counter(labels=["model", "direction"]),  # input/output
    "estimated_cost_usd": Counter(labels=["model", "user_tier"]),

    # Quality (from evals)
    "response_quality_score": Histogram(),
    "retrieval_precision": Gauge(),

    # Errors
    "agent_errors_total": Counter(labels=["error_type"]),
    "guardrail_blocks_total": Counter(labels=["guardrail_type"]),
}

# Structured logging for agent traces
def log_agent_trace(trace: AgentTrace):
    logger.info(
        "agent_trace",
        session_id=trace.session_id,
        request_id=trace.request_id,
        user_id=trace.user_id,
        latency_ms=trace.latency_ms,
        tokens_in=trace.tokens_in,
        tokens_out=trace.tokens_out,
        tool_calls=[t.name for t in trace.tool_calls],
        model=trace.model,
        success=trace.success,
        error=trace.error if not trace.success else None,
    )
```

---

### Production Readiness Checklist

| Category          | Item                                                           | Status |
| ----------------- | -------------------------------------------------------------- | ------ |
| **Context**       | Tiered context architecture (working/session/memory/artifacts) | ☐      |
| **Context**       | Context compilation pipeline (explicit processors)             | ☐      |
| **Context**       | Multi-agent context scoping                                    | ☐      |
| **Security**      | Input guardrails (PII, injection, safety)                      | ☐      |
| **Security**      | Output guardrails (PII redaction, safety)                      | ☐      |
| **Security**      | Tool-level permissions and audit logging                       | ☐      |
| **Security**      | Cloud Armor or WAF configured                                  | ☐      |
| **Resiltic**      | Circuit breakers on external calls                             | ☐      |
| **Resilience**    | Idempotency for tool calls                                     | ☐      |
| **Resilience**    | Graceful degradation on LLM failures                           | ☐      |
| **CI/CD**         | Prompt versioning and testing                                  | ☐      |
| **CI/CD**         | Eval suite in pipeline                                         | ☐      |
| **CI/CD**         | Canary deployment with eval metrics                            | ☐      |
| **Scaling**       | Auto-scaling on concurrent requests                            | ☐      |
| **Scaling**       | Cost budgets per user/session                                  | ☐      |
| **Scaling**       | Queue-based processing for long tasks                          | ☐      |
| **Observability** | Latency, throughput, error metrics                             | ☐      |
| **Observability** | Token/cost tracking                                            | ☐      |
| **Observability** | Structured agent traces                                        | ☐      |

**EXAM TIP:** Production agent questions often combine multiple concerns. When you see "production-ready agent system" → think **context engineering + guardrails + eval-driven CI/CD + cost controls + observability**.

---

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
