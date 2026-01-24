## PART III: GCP ML SERVICES

### Official docs (high-signal starting points)

- **Vertex AI**: `https://cloud.google.com/vertex-ai/docs`
- Vertex AI Training: `https://cloud.google.com/vertex-ai/docs/training/overview`
- Vertex AI Prediction: `https://cloud.google.com/vertex-ai/docs/predictions/overview`
- Vertex AI Model Monitoring: `https://cloud.google.com/vertex-ai/docs/model-monitoring/overview`
- Vertex AI Feature Store: `https://cloud.google.com/vertex-ai/docs/featurestore/overview`
- Vertex AI Explainable AI: `https://cloud.google.com/vertex-ai/docs/explainable-ai/overview`
- **BigQuery ML**: `https://cloud.google.com/bigquery/docs/bqml-introduction`
- BQML Model Types: `https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create`
- **AutoML**: `https://cloud.google.com/vertex-ai/docs/beginner/beginners-guide`
- AutoML Vision: `https://cloud.google.com/vertex-ai/docs/image-data/classification/train-model`
- AutoML Natural Language: `https://cloud.google.com/vertex-ai/docs/text-data/classification/prepare-data`
- **Pre-trained APIs**:
  - Vision API: `https://cloud.google.com/vision/docs`
  - Video Intelligence API: `https://cloud.google.com/video-intelligence/docs`
  - Natural Language API: `https://cloud.google.com/natural-language/docs`
  - Speech-to-Text: `https://cloud.google.com/speech-to-text/docs`
- **Cloud DLP**: `https://cloud.google.com/dlp/docs`
- **Data Catalog**: `https://cloud.google.com/data-catalog/docs`

### 3.1 BIGQUERY ML

Train and deploy ML models using SQL syntax directly in BigQuery.

#### Supported Models

| Model                   | BQML Syntax                              | Supported                 |
| ----------------------- | ---------------------------------------- | ------------------------- |
| Linear Regression       | LINEAR_REG                               | YES                       |
| Logistic Regression     | LOGISTIC_REG                             | YES                       |
| K-Means (auto clusters) | KMEANS                                   | YES - can auto-optimize K |
| Time Series (ARIMA)     | ARIMA_PLUS                               | YES                       |
| Boosted Trees           | BOOSTED_TREE_CLASSIFIER/REGRESSOR        | YES                       |
| DNN (Wide & Deep)       | DNN_CLASSIFIER/REGRESSOR                 | YES                       |
| AutoML Tables           | AUTOML_CLASSIFIER/REGRESSOR              | YES                       |
| Matrix Factorization    | MATRIX_FACTORIZATION                     | YES                       |
| Import TensorFlow model | CREATE MODEL ... OPTIONS(model_path=...) | YES                       |
| CNN                     | —                                        | NO - not supported        |

**COMMON TRAP:** BQML does NOT support CNN (Convolutional Neural Networks).

#### Key BQML Functions

| Function           | Purpose                             | Example                                                |
| ------------------ | ----------------------------------- | ------------------------------------------------------ |
| ML.EVALUATE        | Evaluate model                      | `SELECT * FROM ML.EVALUATE(MODEL my_model)`            |
| ML.PREDICT         | Make predictions                    | `SELECT * FROM ML.PREDICT(MODEL my_model, TABLE data)` |
| ML.EXPLAIN_PREDICT | Predictions with explanations       | Feature attributions for each prediction               |
| EXCEPT             | Select all columns except specified | `SELECT * EXCEPT(label) FROM table`                    |
| FEATURE_CROSS      | Combine categorical features        | `STRUCT(country, language)`                            |
| QUANTILE_BUCKETIZE | Bucket into quantiles               | Bucket income into 5 classes                           |

**EXAM TIP:** Select all features except the label → Use EXCEPT.  
**EXAM TIP:** Bucket income into 5 classes → Use QUANTILE_BUCKETIZE.

#### BQML Limitations

- No automatic deployment/serving - must export to Vertex AI
- No CNN support
- Limited hyperparameter tuning compared to Vertex AI

#### Handling class imbalance in BigQuery ML (highly tested)

When your positive class is rare (fraud, failures, defects), train with class weights so the model doesn’t ignore positives.

- **Option**: `AUTO_CLASS_WEIGHTS = TRUE` (BigQuery ML automatically reweights classes)
- **Works well with**: Logistic regression / boosted trees classification use cases

**EXAM TIP:** Rare event prediction in BigQuery with logistic regression → enable `AUTO_CLASS_WEIGHTS = TRUE` (don’t “multiply feature values”).  
**COMMON TRAP:** Normalization (z-score) helps optimization; it does **not** solve class imbalance.

#### SQL Normalization in BigQuery

For efficiency, translate normalization logic into SQL rather than using Dataflow:

- **Benefit**: Reduces computation time and manual intervention
- **When to Use**: Preprocessing data that's already in BigQuery

**EXAM TIP:** Minimize computation for Z-score normalization on BigQuery data → Translate to SQL (not Dataflow).

#### Importing TensorFlow Models

- **Capability**: Import trained TensorFlow model into BQML and run predictions with SQL
- **Use Case**: Batch prediction on millions of BigQuery records with custom TF model

**EXAM TIP:** Batch prediction on 100M BigQuery records with TensorFlow model → Import TF model to BQML + ML.PREDICT.

**EXAM TIP:** Text model in TF + batch predictions on text stored in BigQuery with minimal infra → import to **BQML** and use `ML.PREDICT` (serverless), rather than building Dataflow inference or managing batch prediction compute.

#### BigQuery ML at Scale

BigQuery ML is ideal for large-scale, serverless ML on structured data.

- **Scalability**: Handles tens of millions of records daily without infrastructure management
- **Serverless**: No cluster management, automatic scaling
- **SQL Interface**: Train models using familiar SQL syntax

**EXAM TIP:** Tens of millions of sensor records daily, minimal development effort → BigQuery ML regression.  
**EXAM TIP:** Serverless ML on structured data already in BigQuery → BigQuery ML.

#### SQL vs PySpark Performance

For data already in BigQuery, SQL transformations are often FASTER than PySpark:

| Approach                     | Speed              | When to Use                                  |
| ---------------------------- | ------------------ | -------------------------------------------- |
| BigQuery SQL transformations | Fast (optimized)   | Data already in BigQuery, SQL expertise      |
| PySpark on Dataproc          | Slower for BQ data | Complex transformations, existing Spark code |
| Dataflow                     | Good               | Streaming data, Apache Beam                  |

- **Pattern**: Load data into BigQuery → Transform with SQL → Write to new table
- **Benefit**: Eliminates data movement, leverages BigQuery's optimization

**EXAM TIP:** PySpark pipeline taking 12+ hours on BigQuery data → Convert to BigQuery SQL (much faster).  
**COMMON TRAP:** Don't use PySpark just because it's familiar if data is already in BigQuery.

### 3.2 VERTEX AI / AI PLATFORM

#### Training Options

| Option               | Control | When to Use                                |
| -------------------- | ------- | ------------------------------------------ |
| AutoML               | Low     | Quick baseline, no ML expertise needed     |
| Pre-built containers | Medium  | Standard frameworks (TF, PyTorch, sklearn) |
| Custom containers    | High    | Any framework, custom dependencies         |

- **Custom Containers**: Support ANY framework - Keras, PyTorch, Theano, scikit-learn, custom libraries

**EXAM TIP:** Team uses many frameworks (Keras, PyTorch, Theano, sklearn, custom) → Custom containers.

**EXAM TIP:** “Replace complex self-managed training backend with a managed service” → use **Vertex AI Training** (custom containers for any framework), not self-managed schedulers or bespoke VM image libraries.

#### PyTorch Training on Vertex AI

For PyTorch models requiring GPUs:

- **Pattern**: Package code with Setuptools + use pre-built PyTorch container + Vertex AI custom tier
- **GPUs**: Specify custom tier with required GPUs (e.g., 4 V100s)
- **NOT**: TFJob (TensorFlow-specific), manual Compute Engine setup

**EXAM TIP:** PyTorch ResNet50 with 4 V100 GPUs → Setuptools + pre-built container + Vertex AI custom tier.  
**COMMON TRAP:** Don't create manual Compute Engine VMs for training - use Vertex AI Training.

#### PyTorch Hyperparameter Tuning

For tuning PyTorch models:

- **Solution**: Run HP tuning job on AI Platform using CUSTOM CONTAINERS
- **NOT**: Convert to TensorFlow/Keras
- **NOT**: Use Katib on Kubeflow

**EXAM TIP:** PyTorch hyperparameter tuning → AI Platform HP tuning with custom containers.  
**COMMON TRAP:** Don't convert PyTorch to Keras/TF just for hyperparameter tuning.

#### Scale-Tier Parameter

Controls the resources allocated to a training job.

- **Effect**: Higher scale-tier = more resources = faster training (with suitable code)
- **Options**: BASIC, STANDARD_1, PREMIUM_1, BASIC_GPU, CUSTOM

**COMMON TRAP:** To speed up training, modify scale-tier (not epochs, batch_size, or learning_rate).

#### Hyperparameter Tuning

| Parameter Type          | Recommended Scaling | Reason                                         |
| ----------------------- | ------------------- | ---------------------------------------------- |
| Learning rate           | UNIT_LOG_SCALE      | Varies over orders of magnitude (0.001 to 0.1) |
| Embedding dimension     | UNIT_LINEAR_SCALE   | Integer with linear relationship to quality    |
| Number of hidden layers | UNIT_LINEAR_SCALE   | Small integer range                            |

- **Parallel Trials**: For Bayesian optimization, use SMALL number of parallel trials
- **Reason**: Bayesian optimization learns from completed trials

**EXAM TIP:** Bayesian optimization + maximize accuracy → small parallel trials + LOG_SCALE for learning rate, LINEAR_SCALE for embedding dimension.

#### Speeding up hyperparameter tuning (without ruining results)

If tuning is taking too long and blocks downstream pipeline steps:

- **Early stopping**: enable early stopping so bad trials terminate sooner
- **Narrow the search space**: tighten parameter ranges based on what you’ve learned
- **Cap trials**: reduce max trials in later tuning phases once you’re near the optimum

**EXAM TIP:** Speed up tuning safely → enable early stopping + narrow ranges (avoid switching to random search unless explicitly asked).

#### Continuous Evaluation

- **Purpose**: Monitor deployed model performance over time
- **Metric**: Mean Average Precision (mAP)

**EXAM TIP:** Monitor model versions performance over time → Continuous Evaluation feature (mAP).

#### Prediction Types

| Type              |       Latency | Use Case                    | GCP Service                |
| ----------------- | ------------: | --------------------------- | -------------------------- |
| Online Prediction |       < 100ms | Real-time user requests     | Vertex AI Endpoints        |
| Batch Prediction  | Minutes-hours | Processing accumulated data | Vertex AI Batch Prediction |
| Streaming         |       Seconds | Continuous data streams     | Dataflow + Model           |

**EXAM TIP:** End-of-day processing of scanned forms → Batch Prediction functionality.

#### Designing “actionable” predictions for business teams (no-code consumers)

Often the requirement is not just accuracy, but making outputs usable by non-technical stakeholders.

- **Serving design**: Return simple outputs (e.g., customer_id + score) that downstream apps can consume
- **Integration**: Embed predictions into existing workflows (CRM, ticketing, marketing tools)

**EXAM TIP:** “Business team needs actionable outputs without coding” → focus on **serving/integration design**, not only model evaluation.

#### Low-Latency Inference in Streaming Pipelines

When running inference in Dataflow and latency is critical:

| Approach                       | Latency              | When to Use                         |
| ------------------------------ | -------------------- | ----------------------------------- |
| Model DIRECTLY in Dataflow job | Lowest               | Minimize latency, TensorFlow models |
| Vertex AI endpoint call        | Higher (network hop) | Managed serving, model updates      |
| TFServing on GKE               | Medium               | Custom serving infrastructure       |
| Cloud Run                      | Higher               | Serverless, cold starts             |

**EXAM TIP:** Anomaly detection with minimum latency → Incorporate model directly into Dataflow job.  
**COMMON TRAP:** Calling external endpoints adds network latency.

#### Ultra-Low Latency with In-Memory Caching

For prediction pipelines with strict latency requirements (e.g., 300ms@p99):

- **Memorystore**: In-memory data store (Redis/Memcached) for fastest read/write
- **Use Case**: Store user navigation context for real-time ad/banner prediction
- **Pattern**: Client → App Engine → Memorystore (context) → AI Platform (prediction)

| Storage        | Latency         | Use Case                                           |
| -------------- | --------------- | -------------------------------------------------- |
| Memorystore    | Sub-millisecond | Session context, user state, 300ms@p99 requirement |
| Cloud Bigtable | Single-digit ms | Large-scale, high-throughput                       |
| Firestore      | Higher          | Document storage, not optimized for latency        |
| Cloud SQL      | Higher          | Relational data, not real-time                     |

**EXAM TIP:** 300ms@p99 latency with user context → Memorystore for context storage + AI Platform prediction.

#### Common Prediction Errors

- **Out of Memory (OOM)**: Send smaller batch of instances in the request

**EXAM TIP:** OOM during online prediction → Send smaller batch (not quota increase).

#### Explainability

Vertex Explainable AI provides feature attributions for predictions.

| Method               | Best For                             | Model Types                       |
| -------------------- | ------------------------------------ | --------------------------------- |
| Sampled Shapley      | Ensembles, non-differentiable models | Random Forest, XGBoost, any model |
| Integrated Gradients | Neural networks                      | Differentiable models only        |
| XRAI                 | Image models                         | Region-based explanations         |

- **Local Feature Importance**: Explains SPECIFIC predictions
- **Global Feature Importance**: Overall model feature importance

**EXAM TIP:** Explain specific loan rejection → LOCAL feature importance.

**EXAM TIP:** Explain an image classifier’s decision (pixel/feature attributions) → **Integrated Gradients** (for neural nets).

#### Using AI Explanations in Predictions

- **Method**: Submit prediction request with 'explain' keyword
- **Response**: Returns feature attributions using configured method

**EXAM TIP:** Which feature influenced THIS prediction most? → AI Explanations with 'explain' keyword + Sampled Shapley.

#### Choosing Explanation Method

| Scenario                            | Method                          |
| ----------------------------------- | ------------------------------- |
| Ensemble of trees + neural networks | Sampled Shapley                 |
| Deep neural network only            | Integrated Gradients            |
| Image classification                | XRAI                            |
| Any model type, tabular data        | Sampled Shapley (safest choice) |

**EXAM TIP:** Ensemble model explanation → Sampled Shapley.

#### Vertex AI Logging

- **Container Logging**: Logs from your training/serving container
- **Access Logging**: API request logs
- **Important**: CANNOT enable dynamically - must undeploy and redeploy to change logging settings

**COMMON TRAP:** To enable logging on a deployed model, you must undeploy and redeploy.

#### Traffic Splitting

Deploy multiple model versions to the same endpoint with different traffic percentages.

- **Use Case**: A/B testing, canary deployments
- **Method**: Deploy both versions to same endpoint, update traffic split percentage

**EXAM TIP:** New model version serving 10% traffic → Deploy to same endpoint + update traffic split.

#### Vertex AI Feature Store (online features)

Use Feature Store when you need:

- Consistent feature definitions for training and serving
- **Sub-10ms** online feature retrieval
- High-volume reads for real-time inference

**EXAM TIP:** Low-latency access to precomputed features for online prediction → **Vertex Feature Store (online serving API)** (not BigQuery).

#### Vertex AI Model Monitoring: drift vs skew

- **Prediction drift**: changes in prediction outputs or feature distributions over time
- **Training-serving skew**: differences between training feature stats and serving feature stats

**EXAM TIP:** “Input feature distribution is changing over time” → configure monitoring for **drift** (feature/prediction drift alerts).  
**COMMON TRAP:** Skew = training vs serving mismatch; drift = time-based change.

#### Vertex AI Endpoints (online predictions)

Use Vertex AI Endpoints for fully managed, low-latency online serving with:

- Managed infrastructure
- Canary/traffic splitting
- Monitoring/logging hooks

**EXAM TIP:** “Fully managed online endpoint + low latency + canary rollouts” → **Vertex AI Endpoints** (not Cloud Functions, not DIY GKE unless required).

#### Vertex ML Metadata

- **Purpose**: Store and compare model versions, experiments, and performance metrics
- **Use Case**: Track performance across different time periods (e.g., seasonal models)

**EXAM TIP:** Compare model versions across seasons/years → Vertex ML Metadata.

#### Reproducibility and lineage (end-to-end)

For long-term auditability (recreate code + deps + data snapshot + artifacts):

- **Vertex ML Metadata**: lineage of runs, datasets, models, metrics
- **Artifact Registry**: version and store container images/dependencies used by training/serving

**EXAM TIP:** “Recreate and audit months later” → use **Vertex ML Metadata** + **Artifact Registry**.

#### Experiment Tracking with API Access

When data science teams need to track experiments and query metrics programmatically:

| Solution                            | API                    | Best For                                |
| ----------------------------------- | ---------------------- | --------------------------------------- |
| Vertex AI Pipelines + MetadataStore | Vertex AI API          | GCP-native, managed, full ML lifecycle  |
| Kubeflow Pipelines                  | Kubeflow Pipelines API | Self-managed, export metrics file       |
| BigQuery logging                    | BigQuery API           | Custom solution, SQL queries            |
| Cloud Monitoring                    | Monitoring API         | Infrastructure metrics, not ML-specific |

**EXAM TIP:** Quick experimentation with API to query metrics → Vertex AI Pipelines + MetadataStore.  
**EXAM TIP:** Self-managed experiment tracking with API → Kubeflow Pipelines API.  
**COMMON TRAP:** Don't use Google Sheets for experiment tracking in production.

#### Vertex AI Generative AI + agents (service mapping)

GenAI questions often look like “how do we make outputs reliable/actionable with enterprise data?” The platform mapping is:

- **Vertex AI Search**: managed retrieval / “out-of-the-box RAG” over your content
- **Vertex AI RAG Engine**: framework for context-augmented LLM applications (RAG pipelines / context management)
- **Vertex AI Studio**: prototyping + built-in safety controls (content filtering / safety attribute scoring) to set confidence thresholds per use case
- **Vertex AI Endpoints**: managed online serving (low latency + traffic splitting) when the model is not “just a chat UI”
- **Gen AI evaluation service (Vertex AI)**: evaluate/benchmark genAI apps/models against criteria (not only manual prompt testing)

**EXAM TIP:** “Need grounded answers over internal docs quickly” → **Vertex AI Search** (and/or **Vertex AI RAG Engine**), not fine-tuning.  
**COMMON TRAP:** Vector storage (embeddings DB) is only one part of RAG; managed **search/retrieval** is usually what the question is asking for.

### 3.3 AUTOML

#### Supported Data Types

| Data Type    | Tasks                                        | Supported                       |
| ------------ | -------------------------------------------- | ------------------------------- |
| Tabular      | Classification, Regression, Forecasting      | YES                             |
| Image        | Classification, Detection, Segmentation      | YES                             |
| Text         | Classification, Entity Extraction, Sentiment | YES                             |
| Video        | Classification, Object Tracking              | YES                             |
| Cluster Data | —                                            | NO - unsupervised not supported |

**COMMON TRAP:** AutoML does NOT support clustering or other unsupervised tasks.

#### AutoML Tables (No-Code Classification)

- **When to Use**: Classification/regression on structured data without writing code
- **Features**: Automatic feature engineering, model selection, hyperparameter tuning

**EXAM TIP:** Classification over structured datasets without writing code → AutoML Tables.

#### AutoML Tables with Time Signal

When your data has a time component:

- **TIME COLUMN**: Specify the time column
- **Effect**: AutoML uses recent data for validation/testing, avoiding leakage

**EXAM TIP:** AutoML Tables with time-spread data → Indicate Time column, let AutoML split by time signal.

#### AutoML Vision Export Options

| Export Format   | Target Platform            |
| --------------- | -------------------------- |
| Core ML         | iOS applications           |
| TensorFlow Lite | Android, mobile, embedded  |
| TensorFlow.js   | Web browsers               |
| Coral/Edge TPU  | Edge devices with Edge TPU |

**EXAM TIP:** Face detection for iOS app → AutoML Vision with Core ML export.  
**EXAM TIP:** Mobile app deployment → TensorFlow Lite export.

#### AutoML Vision Object Detection (when you need labels + fastest delivery)

If you have many images and they are not labeled yet:

- **Labeling**: Use **Google Cloud Data Labeling Service** (human labeling workflow)
- **Training/Deployment**: Use **AutoML Vision Object Detection** to train and deploy quickly

**EXAM TIP:** “Images not labeled yet + need object detection” → Data Labeling Service + AutoML Object Detection.

#### AutoML Vision Edge (On-Device / Offline Inference)

When you need inference to run **on-device** (factory floor, unreliable Wi‑Fi, strict latency), use **AutoML Vision Edge** models.

- **Why**: No network hop, works offline, predictable latency
- **Tradeoff**: Edge models are optimized for either latency or accuracy

| Edge model               | Best for          | Typical scenario                               |
| ------------------------ | ----------------- | ---------------------------------------------- |
| `mobile-low-latency-1`   | Fastest inference | Speed is the priority; unreliable connectivity |
| `mobile-high-accuracy-1` | Best quality      | Accuracy is priority; latency is acceptable    |
| `mobile-versatile-1`     | Balanced          | Unsure / mixed constraints                     |

**EXAM TIP:** Defect detection with unreliable Wi‑Fi + fastest response → AutoML Vision **Edge** `mobile-low-latency-1` (not hosted AutoML Vision).  
**COMMON TRAP:** If the question mentions unreliable connectivity/offline requirement, hosted endpoints are usually the wrong choice.

#### AutoML Natural Language vs Cloud Natural Language API

| Service                    | Type        | Training          | Use Case                                                                |
| -------------------------- | ----------- | ----------------- | ----------------------------------------------------------------------- |
| Cloud Natural Language API | Pre-trained | None required     | General sentiment, entities, NO domain-specific terms                   |
| AutoML Natural Language    | Custom      | Your labeled data | Domain-specific classification, custom entities, product categorization |

- **Key Decision**: Does your use case have domain-specific terminology or custom categories?

**EXAM TIP:** Classify support tickets by product → AutoML Natural Language.  
**EXAM TIP:** General sentiment analysis (no domain terms) → Cloud Natural Language API.  
**EXAM TIP:** Classify calls by product for contact center → AutoML Natural Language (custom entities).

#### Transfer Learning on AI Platform

When you have TensorFlow expertise and want to build on existing models:

- **Pattern**: Deploy pre-existing text classification model on AI Platform for transfer learning
- **Benefit**: Leverage existing model, fine-tune on your data

**EXAM TIP:** TensorFlow expertise + leverage existing resources → Transfer learning with pre-existing model on AI Platform.

#### AutoML Optimization Objectives

| Objective                      | Best For                             | Scenario                              |
| ------------------------------ | ------------------------------------ | ------------------------------------- |
| Maximize AUC PR                | Imbalanced data (fraud, rare events) | Fraud detection with 1% positive rate |
| Maximize AUC ROC               | Balanced classification              | General binary classification         |
| Maximize Precision at Recall X | Specific tradeoff                    | When recall threshold is fixed        |
| Minimize Log Loss              | Calibrated probabilities             | When probability accuracy matters     |

**EXAM TIP:** Fraud detection (1% fraud) → Maximize AUC PR.  
**COMMON TRAP:** Don't use AUC ROC for highly imbalanced data - AUC PR is more appropriate.

### 3.4 PRE-TRAINED APIs

| API                    | Purpose                            | Example Use Case                          |
| ---------------------- | ---------------------------------- | ----------------------------------------- |
| Vision API             | Image analysis, OCR, labels        | General image classification              |
| Video Intelligence API | Video analysis, object tracking    | Security surveillance, content moderation |
| Natural Language API   | Text analysis, sentiment, entities | Analyzing customer reviews                |
| Speech-to-Text         | Audio transcription                | Converting voice to text                  |
| Text-to-Speech         | Speech synthesis                   | Voice assistants                          |
| Dialogflow             | Conversational AI                  | Voice commands, chatbots                  |
| Document AI            | Document extraction                | Invoice processing, form extraction       |

**EXAM TIP:** Voice commands for mobile app → Dialogflow (not just Speech-to-Text).  
**EXAM TIP:** Video surveillance for unauthorized access → Video Intelligence API.

#### Video content moderation (objects + inappropriate content alerts)

For video analysis that needs object detection and content moderation signals with minimal ML development:

- **Ingestion/trigger**: Pub/Sub → Cloud Function
- **Video analysis**: **Video Intelligence API** (pre-trained)
- **Ops**: Cloud Logging (audit, debugging)

**EXAM TIP:** “Analyze video + identify objects + alert on inappropriate content” → Pub/Sub + Cloud Function + **Video Intelligence API** + Cloud Logging.  
**COMMON TRAP:** Vision API is for images/frames; Video Intelligence API is designed for video.

### 3.5 CLOUD DLP (Data Loss Prevention)

#### PII Handling Pattern

For streaming data that may contain Personally Identifiable Information:

1. Create THREE buckets: Quarantine, Sensitive, Non-sensitive
2. Write ALL incoming data to Quarantine bucket first
3. Periodically scan Quarantine bucket with DLP API
4. Move files to Sensitive or Non-sensitive bucket based on scan results

**EXAM TIP:** Real-time PII streaming → Quarantine bucket → DLP scan → route.  
**COMMON TRAP:** DON'T write directly to Sensitive/Non-sensitive - scan FIRST in Quarantine.

#### De-identification Techniques

| Technique                    | Description                          | Use Case                     |
| ---------------------------- | ------------------------------------ | ---------------------------- |
| Masking                      | Replace with characters (\*\*\*)     | Hide data, preserve format   |
| Replacement                  | Replace with placeholder             | Pseudonymization             |
| Tokenization                 | Replace with token, maintain mapping | Reversible de-identification |
| Format-preserving encryption | Encrypt while maintaining format     | Data integrity + security    |

**EXAM TIP:** Replace sensitive data with surrogate characters → Masking.

> **⚠️ Note on "tokenization" terminology**: In the Cloud DLP context, "tokenization" means replacing sensitive values (like SSNs) with non-sensitive surrogate tokens that can be reversed with a key. This is **different** from NLP/LLM tokenization (breaking text into sub-word units for model input). See Part VI § 6.1.1 for LLM tokenization.

#### “Secure but still usable for ML” (coarsening vs tokenization)

Sometimes you must protect data but still preserve numeric meaning:

- **Coarsening/generalization** (good for ML): bucketize age into quantiles, reduce lat/long precision, round values
- **Tokenization/hashing** (good for IDs): replaces meaning with tokens; can destroy numeric ordering/distance

**EXAM TIP:** Protect AGE + LAT/LONG for ML training → coarsen (quantiles / reduced precision), not tokenization of numeric values.

#### Format Preserving Encryption (FPE) for ML Training

When you need to train ML models on sensitive data while protecting PII:

- **Scenario**: All columns vital to model performance, but contain PII
- **Solution**: DLP API + Dataflow + Format Preserving Encryption
- **How it works**: FPE encrypts values while maintaining the same format (e.g., SSN stays 9 digits)
- **Benefit**: Model can still learn patterns from encrypted data

| Approach                 | Keeps Data Usable?  | Use Case                               |
| ------------------------ | ------------------- | -------------------------------------- |
| DLP + Dataflow + FPE     | YES                 | All columns needed for ML, protect PII |
| Remove sensitive columns | NO - loses data     | When columns not needed                |
| Scramble values          | NO - loses patterns | Not suitable for ML                    |
| AES-256 encryption       | NO - changes format | Storage security, not ML               |

**EXAM TIP:** Train ML model with PII but need all columns → DLP API + Dataflow + FPE.  
**COMMON TRAP:** Don't use AES-256 for ML training data - it changes the format and makes data unusable for ML.

### 3.6 PRIVACY-PRESERVING ML

#### Federated Learning

Train models across decentralized data without centralizing the data.

- **How it works**: Model trains locally on each device, only model updates (gradients) are shared
- **Use Case**: Biometric authentication - fingerprints never leave the device

**EXAM TIP:** Biometric authentication without storing fingerprints → Federated Learning.

#### Differential Privacy

Mathematical framework for measuring privacy guarantees.

- **Use Case**: Aggregate statistics without revealing individual data

### 3.7 REGULATED INDUSTRIES

When building ML for regulated industries (insurance, healthcare, finance), certain factors are critical:

| Factor          | Description                               | Why It Matters                  |
| --------------- | ----------------------------------------- | ------------------------------- |
| Traceability    | Track lineage of data and model decisions | Audit requirements              |
| Reproducibility | Ability to reproduce model results        | Verification and validation     |
| Explainability  | Understand why model made decisions       | Regulatory compliance, fairness |

**EXAM TIP:** Regulated insurance company ML model → Traceability, Reproducibility, Explainability.  
**COMMON TRAP:** Federated learning and differential privacy are for PRIVACY, not regulatory compliance.

### 3.8 DATA CATALOG

Enterprise-scale metadata management and data discovery service.

- **Purpose**: Find and understand data assets across the organization
- **Features**: Search by keywords, tag resources, view table descriptions and schemas
- **Use Case**: Finding the right BigQuery table among thousands of datasets

**EXAM TIP:** Need to find correct table among thousands → Data Catalog (search by keywords in table descriptions).  
**COMMON TRAP:** Don't maintain separate lookup tables - use Data Catalog for data discovery.

### 3.9 FIREBASE CLOUD MESSAGING

Push notification service for mobile and web applications.

- **Use Case**: Alert users when ML model predictions meet certain conditions
- **Pattern**: Register users with user ID, trigger notifications based on predictions

**EXAM TIP:** Notify users when account balance predicted to fall below $25 → Firebase Cloud Messaging.  
**COMMON TRAP:** Don't create separate Pub/Sub topics for each user - use Firebase Cloud Messaging with user IDs.
