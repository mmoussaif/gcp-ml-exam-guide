## QUICK REFERENCE

### Problem → Solution

| Problem                                             | Solution                                                                          |
| --------------------------------------------------- | --------------------------------------------------------------------------------- |
| Data doesn't fit in RAM                             | tf.data.Dataset                                                                   |
| 99% AUC with no effort                              | Data leakage - check features                                                     |
| Model accuracy declining over time                  | Data drift - set up skew alarms + retrain                                         |
| Time series accuracy drops in production            | Use time-based split, not random                                                  |
| RAND() split has overlapping records                | Use hash-based split or stored random                                             |
| < 1% positive class won't converge                  | Oversample minority OR downsample with upweighting                                |
| Different feature ranges causing issues             | Normalization                                                                     |
| Loss oscillating during training                    | Lower the learning rate                                                           |
| 5TB single CSV slow                                 | Split files + parallel interleave                                                 |
| 100 billion records in CSV                          | Convert to sharded TFRecords                                                      |
| GPU waiting for data                                | Parallel interleaving                                                             |
| OOM during training                                 | Decrease batch size                                                               |
| OOM during prediction                               | Smaller request batch                                                             |
| Reduce latency 50% no retrain                       | Dynamic range quantization                                                        |
| 100K categories                                     | One-hot hash buckets                                                              |
| Third-party data format changes                     | TFDV schema validation                                                            |
| Missing categorical with predictive power           | Add missing class + binary indicator                                              |
| Multi-class with integer labels                     | Sparse Categorical Cross-Entropy                                                  |
| Inventory prediction with seasons                   | RNN/LSTM (not CNN)                                                                |
| Predict profit by location                          | Feature cross lat×long + binning                                                  |
| Train ML with PII, need all columns                 | DLP + Dataflow + Format Preserving Encryption                                     |
| New data triggers training automatically            | Cloud Storage → Pub/Sub → Cloud Function                                          |
| Preprocessing at prediction time                    | Pub/Sub → Cloud Function → AI Platform                                            |
| PII in streaming data                               | Quarantine bucket → DLP scan → route                                              |
| Regulated industry compliance                       | Traceability, reproducibility, explainability                                     |
| CNN code without Estimator                          | GPU (not TPU)                                                                     |
| TPU input-bound                                     | Interleave + prefetch=batch_size                                                  |
| Speed up AI Platform training                       | Modify scale-tier                                                                 |
| Compare multiple model architectures                | Kubeflow experiments                                                              |
| Query experiment metrics via API                    | Vertex AI Pipelines + MetadataStore                                               |
| Monitor deployed model versions                     | Continuous Evaluation (mAP)                                                       |
| Prevent prediction drift                            | Model monitoring: 10% sampling every 24 hours                                     |
| Query BigQuery in Kubeflow                          | BigQuery Query Component from GitHub                                              |
| Latency on CPU-only GKE                             | Recompile TF Serving for CPU                                                      |
| Minimum latency Dataflow inference                  | Model directly in Dataflow job                                                    |
| 300ms@p99 with user context                         | Memorystore for context storage                                                   |
| 50+ data scientists organizing work                 | Labels on resources                                                               |
| PySpark on BQ data taking 12+ hours                 | Convert to BigQuery SQL                                                           |
| Tens of millions records daily serverless           | BigQuery ML                                                                       |
| End-of-day batch processing                         | Batch prediction functionality                                                    |
| New products released                               | Extend test dataset                                                               |
| Increase precision                                  | Decrease recall (raise threshold)                                                 |
| Biometric without storing data                      | Federated learning                                                                |
| Reduce training costs                               | Preemptible VMs WITH checkpoints                                                  |
| Daily retraining minimize cost                      | AI Platform + GPUs + Cloud Storage                                                |
| Find table among thousands                          | Data Catalog                                                                      |
| Notify users of predictions                         | Firebase Cloud Messaging                                                          |
| Classify calls by product                           | AutoML Natural Language                                                           |
| General sentiment (no domain)                       | Cloud Natural Language API                                                        |
| Automate unit tests on push                         | Cloud Build trigger                                                               |
| Productionize Keras notebook                        | TFX pipeline + Vertex AI Pipelines                                                |
| PyTorch multi-GPU training                          | Setuptools + pre-built container + Vertex AI                                      |
| PyTorch hyperparameter tuning                       | AI Platform HP tuning + custom containers                                         |
| Fraud detection optimization                        | Maximize AUC PR (not AUC ROC)                                                     |
| Explain ensemble predictions                        | Sampled Shapley                                                                   |
| Which feature influenced prediction                 | AI Explanations with 'explain' keyword                                            |
| One-time EDA report                                 | Vertex AI Workbench notebooks                                                     |
| Content moderation metrics                          | Precision/recall on flagged messages                                              |
| Gaming model business metric                        | User participation (games per day)                                                |
| Anti-spam ML benefit over rules                     | Identify problematic phrases                                                      |
| Unreliable Wi-Fi + fastest defect detection         | AutoML Vision Edge `mobile-low-latency-1`                                         |
| Stream app events into BigQuery in real time        | Pub/Sub → Dataflow (streaming) → BigQuery                                         |
| Avoid author leakage in NLP datasets                | Split by author/user (group-based split)                                          |
| Evaluation step OOM in pipeline                     | Run TFMA evaluation on DataflowRunner                                             |
| Need slight latency win without retraining          | Quantization / lower precision first                                              |
| No user-event history for recommender               | Start with heuristics; collect events for ML later                                |
| Video content moderation + object detection         | Pub/Sub → Cloud Function → Video Intelligence API + Cloud Logging                 |
| Imbalanced classification in BQML                   | `AUTO_CLASS_WEIGHTS = TRUE`                                                       |
| Need labels for object detection                    | Data Labeling Service + AutoML Vision Object Detection                            |
| Speed up HP tuning safely                           | Early stopping + narrower parameter ranges                                        |
| Primary benefit of pipeline automation              | Less manual effort + faster iteration + fewer human errors                        |
| Orchestrate workflows (Airflow)                     | Cloud Composer (managed Airflow)                                                  |
| Data warehousing                                    | BigQuery                                                                          |
| Data transformation (ETL/ELT)                       | Dataflow / Dataproc / Cloud Data Fusion                                           |
| Data governance                                     | Policies/standards for access, security, compliance                               |
| Fully managed low-latency online serving + canary   | Vertex AI Endpoints (traffic splitting)                                           |
| Low-latency online features (<10ms)                 | Vertex Feature Store (online serving API)                                         |
| Feature distribution changes over time              | Vertex AI Model Monitoring: drift detection                                       |
| Recreate pipeline months later (lineage)            | Vertex ML Metadata + Artifact Registry                                            |
| Prevent training-serving skew for transforms        | TensorFlow Transform (TFT) in TFX                                                 |
| Validate input schema + detect anomalies            | TFDV / TFX ExampleValidator                                                       |
| Measure bias across subgroups                       | TFMA + Fairness Indicators                                                        |
| Handle flaky third-party API in pipeline            | KFP retries + exponential backoff                                                 |
| Auto-retrain on new ground truth + quality gate     | Continuous Training (CT) in Vertex AI Pipelines                                   |
| Cheapest storage for huge log archive               | Cloud Storage Archive                                                             |
| Need customer-controlled encryption + audit         | CMEK via Cloud KMS                                                                |
| Stakeholder dashboard on BigQuery                   | Looker Studio                                                                     |
| Data quality accuracy                               | Correctness of data values                                                        |
| Dataset completeness metric                         | Missing data percentage                                                           |
| Measure dataset spread                              | Variance / standard deviation                                                     |
| Detect skewness in a distribution                   | Skewness coefficient                                                              |
| Data lake (cheap raw storage)                       | Cloud Storage                                                                     |
| Managed Spark on GCP                                | Dataproc                                                                          |
| ETL pipelines on GCP                                | Dataflow                                                                          |
| Batch processing benefit                            | High-throughput processing                                                        |
| Fault tolerance in distributed systems              | Data replication                                                                  |
| MapReduce purpose                                   | Parallelize work by splitting into sub-tasks                                      |
| Reliability of pipeline changes                     | CI/CD + version control + data validation tests                                   |
| Monitor pipeline status/performance                 | Cloud Monitoring + alert policies                                                 |
| Centralized pipeline logs                           | Cloud Logging                                                                     |
| Runtime error aggregation/alerts                    | Error Reporting                                                                   |
| Schedule pipeline tasks                             | Cloud Scheduler                                                                   |
| Autoscale Dataflow jobs                             | Dataflow autoscaling                                                              |
| Pub/Sub delivery guarantee                          | At-least-once delivery (handle duplicates)                                        |
| Streaming window                                    | Time-based segmentation for aggregation                                           |
| Audit access for compliance                         | Cloud Audit Logs                                                                  |
| Key management                                      | Cloud KMS                                                                         |
| GDPR (EU)                                           | Privacy regulation incl. portability + erasure                                    |
| CCPA (California)                                   | Consumer control over personal data                                               |
| HIPAA                                               | Health information privacy/security                                               |
| PCI-DSS                                             | Payment card industry security                                                    |
| GDPR max fine                                       | 4% global turnover or €20M (whichever higher)                                     |
| Data minimization                                   | Collect only what’s necessary                                                     |
| Data portability (GDPR)                             | Export personal data in structured, common format                                 |
| Data anonymization                                  | Remove/transform PII to protect privacy                                           |
| Confidentiality                                     | Access only for authorized users                                                  |
| Encryption in transit                               | Protect data while transmitted                                                    |
| Verify object integrity                             | Cloud Storage checksums/hashes (CRC32C/MD5)                                       |
| Ground LLM answers on proprietary documents         | Vertex AI Search + (optional) Vertex AI RAG Engine + check grounding API          |
| Build a code-first AI agent on Google Cloud         | Agent Development Kit (ADK) + deploy via Vertex AI Agent Engine                   |
| Scale/govern an “agent workforce”                   | Google Agentspace (managed orchestration)                                         |
| Evaluate a genAI app beyond manual prompt tests     | Vertex AI gen AI evaluation service + automated eval criteria                     |
| Need tool interoperability/context standardization  | Model Context Protocol (MCP)                                                      |
| Agent chooses wrong tools / gets stuck              | Tool unit tests + trajectory evaluation + tool-selection evaluation               |
| Decide agent tool access strategy                   | Specialist tool list (predictable) vs generalist (flexible) vs dynamic selection  |
| Agent needs multi-step reasoning + tool use         | ReAct loop (Thought → Action → Observation)                                       |
| Agent’s first answer is often wrong / needs polish  | Reflection pattern (self-review + iterate)                                        |
| Route requests to the right tool/agent              | Router pattern (model selects among predefined paths/tools)                       |
| Break a complex task into steps/subgoals            | Planning pattern (task decomposition + roadmap)                                   |
| Complex workflow needs multiple specialists         | Multi-agent supervisor/manager pattern (delegate + merge results)                 |
| Embeddings seem “incompatible” after a model change | Keep embedding model/version consistent; re-embed corpus on upgrade               |
| Measure retrieval quality for RAG                   | precision@k + recall@k + nDCG@k (use BEIR/MTEB-style eval sets)                   |
| LLM output stuck repeating filler                   | Tune temperature/top-K/top-P + stop conditions + lower max tokens                 |
| Cache long shared prompts / large-doc chat          | Context caching (prefix caching) to reuse KV cache                                |
| Speed up decoding without changing outputs          | Speculative decoding (drafter + verifier)                                         |
| Improve genAI evaluation at scale                   | Rubric-based eval + LLM autorater + calibrate vs humans (meta-eval)               |
| Route “tool use” for security/ops constraints       | Prefer function calling (client-side exec) over agent-side API execution          |
| Improve retrieval relevance without re-embedding    | Add reranking (bi-encoder retrieve → cross-encoder rerank)                        |
| Query is sparse/ambiguous in RAG                    | HyDE: generate hypothetical answer → embed → retrieve                             |
| Reduce retrieval calls for stable knowledge         | CAG: cache stable knowledge + retrieve only for fresh/dynamic info                |
| RAG feels “almost right” but misses details         | Revisit chunking + overlap + semantic/recursive chunking                          |
| Need to test agents beyond final answer             | Multi-turn evals + component-level evals (retrieval/tool selection) + red teaming |

### Common Exam Traps

- KNN = non-parametric AND lazy learning
- Neural Networks = parametric (NOT non-parametric)
- Decision Tree alone is NOT ensemble; Random Forest IS
- RAND() regenerates per query - records appear in both train/test
- 99% AUC easily = DATA LEAKAGE, not just overfitting
- Model accuracy declining over time = DATA DRIFT, need retraining
- L1 keeps original features; PCA transforms them
- Bayesian optimization → SMALL number of parallel trials
- scikit-learn doesn't benefit from GPU
- GPU not found = REGION availability, not quota
- Don't convert PyTorch to TensorFlow just for AI Platform (use custom containers)
- OOM during training → decrease batch size (not learning rate)
- OOM during prediction → smaller request batch, NOT quota increase
- Loss oscillating → lower learning rate (not increase)
- CNN without Estimator → GPU, not TPU
- scale-tier affects training SPEED, not model quality
- Oversample AND downsample both valid for imbalance
- EXTEND test dataset with new products (don't replace)
- Regulated industries need traceability, NOT federated learning
- Not everything needs ML - optimization problems use algorithms
- ML benefit over rules = identifying PHRASES, not just more keywords
- SQL in BigQuery faster than PySpark for BigQuery data
- Cloud Storage trigger → Pub/Sub → Cloud Function for auto-retrain
- Quarantine bucket pattern for PII streaming
- Format Preserving Encryption for ML with PII (not AES-256)
- Continuous Evaluation uses Mean Average Precision
- Model monitoring: 10% sampling every 24 hours (not 90%)
- Minimum latency in Dataflow → model IN the job (not external calls)
- Memorystore for ultra-low latency user context
- Recompile TF Serving for CPU-only latency issues
- eager_execution=no is NOT valid - use tf.function
- tf.data.Iterator doesn't connect to BigQuery
- BQML doesn't support CNN
- AutoML doesn't support clustering
- Fine-tuning is NOT grounding; use RAG/grounding when you need “up-to-date” or “enterprise doc” answers
- “Out-of-the-box RAG / semantic retrieval over docs” → Vertex AI Search (not just “a vector DB”)
- “Vibe testing prompts” is not production readiness: add evals + monitoring + guardrails (AgentOps mindset)
- Agents that call tools increase security risk: apply least privilege IAM + input/output guardrails + prompt-injection defenses
- Safety controls often show up as: content filtering + safety attribute scoring + confidence thresholds (especially in Vertex AI Studio / GenAI APIs)
- Token limit does NOT make output “more concise” — it only truncates; you still need explicit prompt constraints
- Gemini is the MODEL family; “Gemini app” / “Gemini in Gmail/BigQuery/Looker” are product surfaces using the model
- Multimodal GenAI = uses multiple modalities together (e.g., video/audio + text), not “any GenAI that outputs images”
- LLMs are a specialized type of foundation model (LLM ⊂ foundation model); not all foundation models are LLMs
- GenAI strategy: don’t start with the tech — start with business priorities, run safe bottom-up experiments, and scale with top-down guardrails
- Augment vs automate: use GenAI to augment human judgment/strategy, and automate repetitive tasks — keep humans-in-the-loop for data, prompts, QA, and monitoring
- AI-assisted captioning pattern: generate a time-coded draft artifact (captions/transcript) + human review/QA gate; track edits as feedback/eval data
- Why GenAI took off (around 2022): LLM capability jump + more accessible compute/tooling → much stronger “out-of-the-box” usefulness
- Definitions: AI ⊃ ML ⊃ GenAI (GenAI creates new content; ML learns a model from data; AI is the broader goal)
- DL/foundation/GenAI: deep learning (neural nets) underpins foundation models; GenAI is an application of these models to generate new content
- GenAI layers (leader mental model): infrastructure → models → platform → agents → applications (apps are just the top layer)
- Infrastructure layer: compute (GPUs/TPUs), storage, networking; if no managed platform, plan infra for data prep + training + deployment + refinement + monitoring
- Edge vs cloud: edge for low-latency/privacy/offline (cars, surgical devices); cloud for large-scale centralized workloads (chatbots at scale, city traffic analytics)
- Edge tooling: LiteRT (on-device runtime) + Gemini Nano (compact on-device model) + AI Edge SDK; often train/tune/eval in Vertex AI then convert/deploy/monitor at edge
- Platform layer (Vertex AI): the “glue” that unifies infra + models + deployment + MLOps (pipelines/metadata/monitoring) + security/IAM
- Vertex AI MLOps: model registry + evaluation + pipelines (orchestration) + monitoring (skew/drift/perf degradation)
- Model layer (Vertex AI): Model Garden for ready models (Google/partners/open) + tuning; choose custom training when you need full control
- AutoML objectives: image (classification/OD), video (action recognition/classification/tracking), tabular (classification/regression/forecasting)
- Pattern: AutoML (specialist extractor → structured metadata) + Gemini (LLM → summaries/Q&A) + Vertex AI Pipelines (orchestration) for domain artifacts like CAD
- Agents vs apps: the app is the user-facing product/framework; the agent is the “actor” inside it that reasons + uses tools + takes actions (often multi-agent)
- Agents beyond models: reasoning loop (observe → interpret → plan → act) + tools (APIs/data/actions) enable multi-step work; conversational vs workflow agents are common patterns
- Quiz recall: infrastructure = compute/storage foundation; agents layer defines the “actions/tasks” inside an app (filter/summarize/recommend); applications layer = user-facing UI/experience
- Data for ML/GenAI: relevance + quality (accuracy/completeness/representativeness/consistency) + accessibility (availability/cost/format) often matter more than “fancier models”
- Data types: structured (tables) vs unstructured (text/images/audio/video) vs semi-structured (JSON/nested) drive what modeling approaches and tooling make sense
- Learning types: supervised = labeled targets; unsupervised = unlabeled pattern discovery; reinforcement = interaction + rewards/penalties
- Semi-supervised learning: combine a small labeled set with a large unlabeled set (common when labels are expensive)
- Diffusion models: generate media by iteratively denoising noise into structured outputs (common for image generation)
- Model selection factors: modality, context window, security, availability/reliability, cost, performance, tuning/customization, ease of integration
- Gemini vs Gemma: Gemini = multimodal model family; Gemma = lightweight open models for local/specialized deployments
- Imagen vs Veo: Imagen = text-to-image diffusion; Veo = text/image-to-video generation
- Foundation model limitations: hallucinations + knowledge cutoff + bias/fairness + edge cases + data dependency; grounding/RAG ties answers to trusted sources
- Grounding purpose (quiz-style): improve accuracy/reliability by connecting outputs to verifiable sources (enterprise docs, KBs, databases)
- Secure AI (SAIF mindset): defend across the lifecycle (data→train→deploy→operate); key risks include data poisoning, model theft, adversarial/prompt-injection attacks; use IAM least privilege + encryption + logging/monitoring + Security Command Center
- Responsible AI: security foundation + transparency + privacy + bias/fairness + accountability/explainability (+ evolving legal requirements); Vertex Explainable AI helps interpret model outputs
- Lifecycle mapping: ingest (Pub/Sub/GCS/SQL/Spanner) → prepare (BigQuery + Data Catalog) → train/deploy/manage (Vertex AI + Pipelines + monitoring/versioning + IAM)
- Consistency matters: inconsistent formats/labels confuse models and hinder learning (it’s not “just a data engineering issue”)
- ML lifecycle order (core): data ingestion+preparation → model training → model deployment → model management
- Ingestion+prep goal: collect, clean, and transform raw data into a usable training/prediction dataset
- Embeddings from different models/versions are not comparable; plan safe upgrades (re-embed + regression tests)
- Feature Cross is for TABULAR data, not CNN
- Location features: feature cross lat×long + binning (not raw lat/long)
- AutoML NL = custom training. Cloud NL API = pre-trained.
- Preemptible VMs REQUIRE checkpoints or you lose progress
- RNN for sequential/temporal data, CNN for spatial/image data
- Sparse Categorical CE for integer labels, Categorical CE for one-hot
- Cloud Build trigger for automated testing, not Cloud Logging sink
- Data Catalog for data discovery, not lookup tables
- Firebase Cloud Messaging for user notifications, not Pub/Sub per user
- PyTorch HP tuning → custom containers (don't convert to TF)
- Fraud detection → maximize AUC PR, not AUC ROC
- Sampled Shapley for ensembles, Integrated Gradients for NNs
- Vertex AI Workbench for EDA reports, not Data Studio
- TFX + Vertex AI Pipelines for productionizing notebooks
- Daily retraining → AI Platform + Cloud Storage, not DLVM
- Parallel interleaving for GPU input bottlenecks

### Study resources

- `cloud.google.com/certification/machine-learning-engineer`

### Verification checklist (manual)

Automated web search in this environment is currently unreliable, so verify these against official docs:

- **Certification logistics** (duration/cost/format): Google Cloud Certification page.
- **BQML model support list** (especially DNN\_\* and CNN not supported): BigQuery ML docs.
- **Vertex AI endpoints, batch prediction, monitoring**: Vertex AI docs.
- **Cloud DLP format-preserving encryption**: Cloud DLP docs.
- **TFX component definitions**: TFX docs.
