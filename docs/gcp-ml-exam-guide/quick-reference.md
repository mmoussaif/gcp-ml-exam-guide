## QUICK REFERENCE

A friendly, organized guide to common ML engineering and GenAI/agent problems and their solutions on Google Cloud.

This quick reference is split into two tracks:

- **Track A (Predictive ML)**: Parts I–V (mostly)
- **Track B (GenAI/Agentic AI)**: Parts V.5–VII (mostly)

### Table of Contents

- [Track A — Predictive ML](#track-a--predictive-ml)
- [Track B — GenAI & Agents](#track-b--genai--agents)
- [Data & Feature Engineering](#data--feature-engineering)
- [Model Training & Optimization](#model-training--optimization)
- [Data Quality & Validation](#data-quality--validation)
- [Deployment & Serving](#deployment--serving)
- [Pipelines & Automation](#pipelines--automation)
- [Monitoring & Operations](#monitoring--operations)
- [Security & Compliance](#security--compliance)
- [Generative AI & Agents](#generative-ai--agents)
- [Common Exam Traps](#common-exam-traps)
- [Study Resources](#study-resources)

---

## Track A — Predictive ML

Use this section when you’re studying the **core exam track** (Parts I–V).

## Track B — GenAI & Agents

Use this section when you’re studying **LLMs/RAG/agents** (Parts VI–VII) and **GenAI infra** (Part V.5).

- Track A entry page: `track-predictive-ml.md`
- Track B entry page: `track-genai-agentic.md`

---

## Data & Feature Engineering

### Data Handling & Storage

| Problem                                 | Solution                            |
| --------------------------------------- | ----------------------------------- |
| Data doesn't fit in RAM                 | Use `tf.data.Dataset` for streaming |
| 5TB single CSV is slow                  | Split files + parallel interleave   |
| 100 billion records in CSV              | Convert to sharded TFRecords        |
| GPU waiting for data                    | Parallel interleaving               |
| Need cheap storage for huge log archive | Cloud Storage Archive class         |
| Data lake (cheap raw storage)           | Cloud Storage                       |
| Data warehousing (analytics)            | BigQuery                            |

### Data Splitting & Leakage

| Problem                                  | Solution                                                                      |
| ---------------------------------------- | ----------------------------------------------------------------------------- |
| RAND() split has overlapping records     | Use hash-based split (`MOD(FARM_FINGERPRINT(id), 10)`) or stored random value |
| Time series accuracy drops in production | Use time-based split (train on past, test on future)                          |
| Avoid author leakage in NLP datasets     | Split by author/user (group-based split), not by sentence                     |
| 99% AUC with no effort                   | **Data leakage** - check for features correlated with target or future data   |

### Feature Engineering

| Problem                                   | Solution                                                      |
| ----------------------------------------- | ------------------------------------------------------------- |
| Different feature ranges causing issues   | Normalization (Min-Max, Z-Score, Log Scaling)                 |
| 100K categories                           | One-hot hash buckets (not standard one-hot)                   |
| Missing categorical with predictive power | Add 'MISSING' class + binary indicator feature                |
| Predict profit by location                | Feature cross lat×long + binning (not raw coordinates)        |
| Multi-class with integer labels           | Use **Sparse Categorical Cross-Entropy** (not Categorical CE) |
| Multi-class with one-hot labels           | Use **Categorical Cross-Entropy**                             |

**EXAM TIP:** Feature Cross is for **TABULAR** data, NOT for CNN (CNN uses convolution kernels).

---

## Model Training & Optimization

### Training Issues

| Problem                            | Solution                                                |
| ---------------------------------- | ------------------------------------------------------- |
| Loss oscillating during training   | Lower the learning rate                                 |
| OOM during training                | Decrease batch size                                     |
| OOM during prediction              | Smaller request batch (not quota increase)              |
| < 1% positive class won't converge | Oversample minority OR downsample with upweighting      |
| Model accuracy declining over time | **Data drift** - set up skew alarms + retrain           |
| Reduce training costs              | Preemptible VMs **WITH checkpoints** (or lose progress) |
| Daily retraining minimize cost     | AI Platform + GPUs + Cloud Storage                      |
| Speed up AI Platform training      | Modify scale-tier (affects speed, not quality)          |
| TPU input-bound                    | Interleave + prefetch=batch_size                        |

### Model Selection

| Problem                              | Solution                                      |
| ------------------------------------ | --------------------------------------------- |
| Inventory prediction with seasons    | RNN/LSTM (not CNN) - sequential/temporal data |
| CNN code without Estimator           | GPU (not TPU)                                 |
| PyTorch multi-GPU training           | Setuptools + pre-built container + Vertex AI  |
| PyTorch hyperparameter tuning        | AI Platform HP tuning + custom containers     |
| Compare multiple model architectures | Kubeflow experiments                          |

### Model Optimization

| Problem                                    | Solution                               |
| ------------------------------------------ | -------------------------------------- |
| Reduce latency 50% without retraining      | Dynamic range quantization             |
| Need slight latency win without retraining | Quantization / lower precision first   |
| Explain ensemble predictions               | Sampled Shapley                        |
| Which feature influenced prediction        | AI Explanations with 'explain' keyword |

---

## Data Quality & Validation

| Problem                                      | Solution                          |
| -------------------------------------------- | --------------------------------- |
| Third-party data format changes              | TFDV schema validation            |
| Validate input schema + detect anomalies     | TFDV / TFX ExampleValidator       |
| Prevent training-serving skew for transforms | TensorFlow Transform (TFT) in TFX |
| Measure bias across subgroups                | TFMA + Fairness Indicators        |
| Data quality accuracy                        | Correctness of data values        |
| Dataset completeness metric                  | Missing data percentage           |
| Measure dataset spread                       | Variance / standard deviation     |
| Detect skewness in a distribution            | Skewness coefficient              |

---

## Deployment & Serving

### Serving Options

| Problem                                           | Solution                                            |
| ------------------------------------------------- | --------------------------------------------------- |
| Fully managed low-latency online serving + canary | Vertex AI Endpoints (traffic splitting)             |
| Low-latency online features (<10ms)               | Vertex Feature Store (online serving API)           |
| Latency on CPU-only GKE                           | Recompile TF Serving for CPU                        |
| Minimum latency Dataflow inference                | Model directly in Dataflow job (not external calls) |
| 300ms@p99 with user context                       | Memorystore for context storage                     |
| End-of-day batch processing                       | Batch prediction functionality                      |

### Productionization

| Problem                                   | Solution                           |
| ----------------------------------------- | ---------------------------------- |
| Productionize Keras notebook              | TFX pipeline + Vertex AI Pipelines |
| Tens of millions records daily serverless | BigQuery ML                        |
| PySpark on BQ data taking 12+ hours       | Convert to BigQuery SQL (faster)   |

---

## Pipelines & Automation

### Pipeline Setup

| Problem                                         | Solution                                        |
| ----------------------------------------------- | ----------------------------------------------- |
| New data triggers training automatically        | Cloud Storage → Pub/Sub → Cloud Function        |
| Preprocessing at prediction time                | Pub/Sub → Cloud Function → AI Platform          |
| Orchestrate workflows (Airflow)                 | Cloud Composer (managed Airflow)                |
| Query BigQuery in Kubeflow                      | BigQuery Query Component from GitHub            |
| Handle flaky third-party API in pipeline        | KFP retries + exponential backoff               |
| Auto-retrain on new ground truth + quality gate | Continuous Training (CT) in Vertex AI Pipelines |
| Recreate pipeline months later (lineage)        | Vertex ML Metadata + Artifact Registry          |

### Data Pipelines

| Problem                                      | Solution                                     |
| -------------------------------------------- | -------------------------------------------- |
| Data transformation (ETL/ELT)                | Dataflow / Dataproc / Cloud Data Fusion      |
| Stream app events into BigQuery in real time | Pub/Sub → Dataflow (streaming) → BigQuery    |
| Managed Spark on GCP                         | Dataproc                                     |
| ETL pipelines on GCP                         | Dataflow                                     |
| Batch processing benefit                     | High-throughput processing                   |
| Fault tolerance in distributed systems       | Data replication                             |
| MapReduce purpose                            | Parallelize work by splitting into sub-tasks |
| Autoscale Dataflow jobs                      | Dataflow autoscaling                         |
| Pub/Sub delivery guarantee                   | At-least-once delivery (handle duplicates)   |
| Streaming window                             | Time-based segmentation for aggregation      |

### CI/CD & Automation

| Problem                                | Solution                                                   |
| -------------------------------------- | ---------------------------------------------------------- |
| Automate unit tests on push            | Cloud Build trigger (not Cloud Logging sink)               |
| Reliability of pipeline changes        | CI/CD + version control + data validation tests            |
| Schedule pipeline tasks                | Cloud Scheduler                                            |
| Primary benefit of pipeline automation | Less manual effort + faster iteration + fewer human errors |

---

## Monitoring & Operations

### Model Monitoring

| Problem                                | Solution                                                    |
| -------------------------------------- | ----------------------------------------------------------- |
| Monitor deployed model versions        | Continuous Evaluation (mAP)                                 |
| Prevent prediction drift               | Model monitoring: **10% sampling every 24 hours** (not 90%) |
| Feature distribution changes over time | Vertex AI Model Monitoring: drift detection                 |
| Query experiment metrics via API       | Vertex AI Pipelines + MetadataStore                         |

### Logging & Observability

| Problem                             | Solution                                        |
| ----------------------------------- | ----------------------------------------------- |
| Monitor pipeline status/performance | Cloud Monitoring + alert policies               |
| Centralized pipeline logs           | Cloud Logging                                   |
| Runtime error aggregation/alerts    | Error Reporting                                 |
| One-time EDA report                 | Vertex AI Workbench notebooks (not Data Studio) |
| Stakeholder dashboard on BigQuery   | Looker Studio                                   |

### Resource Management

| Problem                             | Solution                                    |
| ----------------------------------- | ------------------------------------------- |
| 50+ data scientists organizing work | Labels on resources (not separate projects) |
| Find table among thousands          | Data Catalog (not lookup tables)            |

---

## Security & Compliance

### Data Protection

| Problem                                     | Solution                                      |
| ------------------------------------------- | --------------------------------------------- |
| Train ML with PII, need all columns         | DLP + Dataflow + Format Preserving Encryption |
| PII in streaming data                       | Quarantine bucket → DLP scan → route          |
| Need customer-controlled encryption + audit | CMEK via Cloud KMS                            |
| Verify object integrity                     | Cloud Storage checksums/hashes (CRC32C/MD5)   |
| Key management                              | Cloud KMS                                     |
| Audit access for compliance                 | Cloud Audit Logs                              |

### Privacy & Compliance

| Problem                        | Solution                                          |
| ------------------------------ | ------------------------------------------------- |
| Regulated industry compliance  | Traceability, reproducibility, explainability     |
| GDPR (EU)                      | Privacy regulation incl. portability + erasure    |
| CCPA (California)              | Consumer control over personal data               |
| HIPAA                          | Health information privacy/security               |
| PCI-DSS                        | Payment card industry security                    |
| GDPR max fine                  | 4% global turnover or €20M (whichever higher)     |
| Data minimization              | Collect only what's necessary                     |
| Data portability (GDPR)        | Export personal data in structured, common format |
| Data anonymization             | Remove/transform PII to protect privacy           |
| Confidentiality                | Access only for authorized users                  |
| Encryption in transit          | Protect data while transmitted                    |
| Biometric without storing data | Federated learning                                |

---

## Generative AI & Agents

### RAG & Grounding

| Problem                                             | Solution                                                                 |
| --------------------------------------------------- | ------------------------------------------------------------------------ |
| Ground LLM answers on proprietary documents         | Vertex AI Search + (optional) Vertex AI RAG Engine + check grounding API |
| Improve retrieval relevance without re-embedding    | Add reranking (bi-encoder retrieve → cross-encoder rerank)               |
| Query is sparse/ambiguous in RAG                    | HyDE: generate hypothetical answer → embed → retrieve                    |
| Reduce retrieval calls for stable knowledge         | CAG: cache stable knowledge + retrieve only for fresh/dynamic info       |
| RAG feels "almost right" but misses details         | Revisit chunking + overlap + semantic/recursive chunking                 |
| Measure retrieval quality for RAG                   | precision@k + recall@k + nDCG@k (use BEIR/MTEB-style eval sets)          |
| Embeddings seem "incompatible" after a model change | Keep embedding model/version consistent; re-embed corpus on upgrade      |
| Traditional RAG misses context / needs refinement   | Agentic RAG: agent decides when to retrieve, multi-hop + query rewriting |

### LLM Optimization

| Problem                                              | Solution                                                                         |
| ---------------------------------------------------- | -------------------------------------------------------------------------------- |
| Estimate LLM cost / check context window fit         | Count tokens (Vertex AI `countTokens`, tiktoken, HF tokenizer)                   |
| Input silently truncated / weird special chars       | Tokenization issue — use model's matched tokenizer, handle truncation explicitly |
| Multilingual text uses way more tokens than expected | Non-Latin scripts often 2–3x tokens — test with representative samples           |
| LLM output stuck repeating filler                    | Tune temperature/top-K/top-P + stop conditions + lower max tokens                |
| Cache long shared prompts / large-doc chat           | Context caching (prefix caching) to reuse KV cache                               |
| Speed up decoding without changing outputs           | Speculative decoding (drafter + verifier)                                        |

### Agent Development

| Problem                                            | Solution                                                                              |
| -------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Build agent on GCP with code-first control         | ADK (Agent Development Kit) → deploy to Vertex AI Agent Engine                        |
| Need role-based multi-agent team quickly           | CrewAI (Agents + Tasks + Crew with sequential/hierarchical process)                   |
| Need explicit state machine control for agents     | LangGraph (StateGraph + conditional edges + checkpointer)                             |
| Complex workflow needs specialist delegation       | Supervisor pattern (manager agent + specialist agents)                                |
| Agent forgets previous turns / loses context       | Session management: persist messages + checkpointing (LangGraph, ADK sessions)        |
| Context window fills up in long conversations      | Sliding window / summarization / token budget / hybrid memory strategies              |
| Need to remember user across sessions              | Long-term memory: vector store + structured DB for facts, preferences, episodes       |
| Resume conversation after app restart              | Checkpointer (LangGraph) or session service (ADK) with durable backend                |
| Context window explodes in long agent runs         | Context engineering: tiered storage (session/memory/artifacts) + compiled views       |
| Multi-agent causes context explosion               | Scoped handoffs: agents see minimum context; use "agents as tools" or scoped transfer |
| Agent chooses wrong tools / gets stuck             | Tool unit tests + trajectory evaluation + tool-selection evaluation                   |
| Decide agent tool access strategy                  | Specialist tool list (predictable) vs generalist (flexible) vs dynamic selection      |
| Agent needs multi-step reasoning + tool use        | ReAct loop (Thought → Action → Observation)                                           |
| Agent's first answer is often wrong / needs polish | Reflection pattern (self-review + iterate)                                            |
| Route requests to the right tool/agent             | Router pattern (model selects among predefined paths/tools)                           |
| Break a complex task into steps/subgoals           | Planning pattern (task decomposition + roadmap)                                       |
| Desktop tool for agent dev with browser automation | **Google Antigravity** (`antigravity.google`) — agent manager + browser-in-the-loop   |

### Agent Production & Security

| Problem                                            | Solution                                                                             |
| -------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Agent API needs streaming for chat UX              | SSE (EventSource) for token-by-token; WebSocket for bidirectional                    |
| Protect agent API from abuse                       | Cloud Armor WAF + rate limits per IP/session + input guardrails                      |
| Agent CI/CD: how to test non-deterministic outputs | Eval suite (LLM-as-judge + retrieval metrics) as quality gate; canary deploy         |
| Agent costs spiral out of control                  | Token budgets per user/session; cost-aware routing to cheaper models                 |
| Agent needs API keys / credentials                 | **Secret Manager** (never in prompts/code); use `latest` version for rotation        |
| Prevent agent privilege escalation                 | **IAM**: separate service account per agent; short-lived tokens; user impersonation  |
| Enterprise auth + rate limiting for agent API      | **Apigee** + GKE Inference Gateway (LLM-specific policies)                           |
| Prevent agent from exfiltrating data               | **VPC Service Controls** perimeter around agent project                              |
| Highly sensitive data (healthcare/finance)         | **Confidential Computing** (Confidential VMs/GKE); data encrypted in use             |
| Block prompt injection / jailbreaks at agent level | **Model Armor** (LLM-level guardrails); Cloud Armor is HTTP-level, not semantic      |
| Agent leaking PII / sensitive data in responses    | **Model Armor** output filters + Cloud DLP for redaction                             |
| Gemini 3 agents losing reasoning in long sessions  | Use **thought signatures** (pass back to maintain state); `thinking_level` parameter |
| Need tool interoperability/context standardization | Model Context Protocol (MCP)                                                         |
| Scale/govern an "agent workforce"                  | Google Agentspace (managed orchestration)                                            |
| Evaluate a genAI app beyond manual prompt tests    | Vertex AI gen AI evaluation service + automated eval criteria                        |
| Need to test agents beyond final answer            | Multi-turn evals + component-level evals (retrieval/tool selection) + red teaming    |
| Improve genAI evaluation at scale                  | Rubric-based eval + LLM autorater + calibrate vs humans (meta-eval)                  |
| Route "tool use" for security/ops constraints      | Prefer function calling (client-side exec) over agent-side API execution             |

### GenAI Models & Tools

| Problem                                          | Solution                                                                             |
| ------------------------------------------------ | ------------------------------------------------------------------------------------ |
| Fast, cost-effective image generation for agents | **Nano Banana Pro** (efficient Imagen variant); use Imagen for max quality marketing |

---

## Common Exam Traps

### Algorithm Classification

- **KNN** = non-parametric **AND** lazy learning (stores all training data)
- **Neural Networks** = parametric (NOT non-parametric)
- **Decision Tree** alone is NOT ensemble; **Random Forest** IS ensemble
- **L1 regularization** keeps original features; **PCA** transforms them

### Data Issues

- **RAND()** regenerates per query - records appear in both train/test sets
- **99% AUC easily** = **DATA LEAKAGE**, not just overfitting
- **Model accuracy declining over time** = **DATA DRIFT**, need retraining
- **EXTEND** test dataset with new products (don't replace entirely)

### Training & Optimization

- **OOM during training** → decrease batch size (not learning rate)
- **OOM during prediction** → smaller request batch, NOT quota increase
- **Loss oscillating** → lower learning rate (not increase)
- **CNN without Estimator** → GPU, not TPU
- **scale-tier** affects training SPEED, not model quality
- **Oversample AND downsample** both valid for class imbalance
- **Bayesian optimization** → SMALL number of parallel trials
- **scikit-learn** doesn't benefit from GPU
- **GPU not found** = REGION availability, not quota

### Model Selection

- **RNN** for sequential/temporal data, **CNN** for spatial/image data
- **Sparse Categorical CE** for integer labels, **Categorical CE** for one-hot
- **Don't convert PyTorch to TensorFlow** just for AI Platform (use custom containers)
- **Preemptible VMs REQUIRE checkpoints** or you lose progress

### Services & Tools

- **AutoML NL** = custom training. **Cloud NL API** = pre-trained.
- **Cloud Build trigger** for automated testing, not Cloud Logging sink
- **Data Catalog** for data discovery, not lookup tables
- **Firebase Cloud Messaging** for user notifications, not Pub/Sub per user
- **PyTorch HP tuning** → custom containers (don't convert to TF)
- **Fraud detection** → maximize AUC PR, not AUC ROC
- **Sampled Shapley** for ensembles, **Integrated Gradients** for NNs
- **Vertex AI Workbench** for EDA reports, not Data Studio
- **TFX + Vertex AI Pipelines** for productionizing notebooks
- **Daily retraining** → AI Platform + Cloud Storage, not DLVM
- **Parallel interleaving** for GPU input bottlenecks

### GenAI & Agents

- **"Tokenization" has two meanings**: (1) DLP/privacy = replace PII with surrogate tokens; (2) NLP/LLM = split text into sub-word units for model input
- **Fine-tuning is NOT grounding**; use RAG/grounding when you need "up-to-date" or "enterprise doc" answers
- **"Out-of-the-box RAG / semantic retrieval over docs"** → Vertex AI Search (not just "a vector DB")
- **"Vibe testing prompts" is not production readiness**: add evals + monitoring + guardrails (AgentOps mindset)
- **Agents that call tools increase security risk**: apply least privilege IAM + input/output guardrails + prompt-injection defenses
- **Token limit does NOT make output "more concise"** — it only truncates; you still need explicit prompt constraints
- **Gemini** is the MODEL family; "Gemini app" / "Gemini in Gmail/BigQuery/Looker" are product surfaces using the model
- **Multimodal GenAI** = uses multiple modalities together (e.g., video/audio + text), not "any GenAI that outputs images"
- **LLMs are a specialized type of foundation model** (LLM ⊂ foundation model); not all foundation models are LLMs

### General ML Concepts

- **Not everything needs ML** - optimization problems use algorithms
- **ML benefit over rules** = identifying PHRASES, not just more keywords
- **SQL in BigQuery faster than PySpark** for BigQuery data
- **Cloud Storage trigger** → Pub/Sub → Cloud Function for auto-retrain
- **Quarantine bucket pattern** for PII streaming
- **Format Preserving Encryption** for ML with PII (not AES-256)
- **Continuous Evaluation** uses Mean Average Precision
- **Model monitoring**: 10% sampling every 24 hours (not 90%)
- **Minimum latency in Dataflow** → model IN the job (not external calls)
- **Memorystore** for ultra-low latency user context
- **Recompile TF Serving** for CPU-only latency issues
- **eager_execution=no is NOT valid** - use tf.function
- **tf.data.Iterator doesn't connect to BigQuery**
- **BQML doesn't support CNN**
- **AutoML doesn't support clustering**
- **Regulated industries** need traceability, NOT federated learning

---

## Study Resources

- **Official Certification Page**: <a href="https://cloud.google.com/certification/machine-learning-engineer">cloud.google.com/certification/machine-learning-engineer</a>

### Verification Checklist (Manual)

Automated web search in this environment is currently unreliable, so verify these against official docs:

- **Certification logistics** (duration/cost/format): Google Cloud Certification page
- **BQML model support list** (especially DNN\_\* and CNN not supported): BigQuery ML docs
- **Vertex AI endpoints, batch prediction, monitoring**: Vertex AI docs
- **Cloud DLP format-preserving encryption**: Cloud DLP docs
- **TFX component definitions**: TFX docs

---

**Tip**: Use Ctrl+F (or Cmd+F) to quickly search this document for specific problems or solutions!
