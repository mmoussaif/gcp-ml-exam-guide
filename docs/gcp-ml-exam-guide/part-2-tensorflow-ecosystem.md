## PART II: TENSORFLOW ECOSYSTEM

### 2.1 DATA HANDLING

#### tf.data.Dataset

The standard way to handle data that doesn't fit in memory. Uses lazy evaluation and streaming.

- **Key Feature**: Streams data from disk - doesn't load everything into RAM
- **When to Use**: Data doesn't fit in memory, need efficient I/O
- **NOT Alternatives**: pandas DataFrame (loads all to memory), NumPy arrays (loads all to memory)

**EXAM TIP:** Data doesn't fit in RAM → tf.data.Dataset (not pandas or NumPy).

#### TFRecord Format

Binary format optimized for TensorFlow data loading.

- **tf.train.Feature**: Wrapper for data types (BytesList, FloatList, Int64List) to store in TFRecord
- **Use Case**: Mixed data types (images + CSV), efficient storage and reading

**EXAM TIP:** Load images + CSV data efficiently → Convert to TFRecord using tf.train.Feature.

#### TFRecord Sharding for Massive Datasets

For extremely large datasets (100 billion+ records), use sharded TFRecords:

- **Sharding**: Split data into multiple TFRecord files (shards)
- **Benefits**: Parallel reading, better I/O performance, enables distributed training
- **Storage**: Store sharded TFRecords in Cloud Storage

**EXAM TIP:** 100 billion records in CSV files → Convert to sharded TFRecords in Cloud Storage.
**COMMON TRAP:** Don’t pick HDFS for GCP-native pipelines. For managed, scalable storage in Google Cloud, the default answer is **Cloud Storage**.

#### Input Pipeline Optimization

When profiling shows your training is INPUT-BOUND (data loading is the bottleneck):

| Technique                                    | Effect                                      |
| -------------------------------------------- | ------------------------------------------- |
| Split large file into multiple smaller files | Enables parallel reading                    |
| Use interleave option                        | Read from multiple files in parallel        |
| Set prefetch = batch_size                    | Overlap data loading with model computation |
| Use TFRecord format                          | More efficient than CSV/JSON                |
| Cache dataset                                | Avoid re-reading same data                  |

**EXAM TIP:** TPU training is input-bound → Use interleave + set prefetch equal to batch_size.  
**EXAM TIP:** 5TB single CSV is slow → Split into multiple files + parallel interleave transformation.

#### GPU Input Pipeline Bottlenecks

When GPU utilization is low during training (GPU waiting for data):

- **Symptom**: Native synchronous version, data split into several files
- **Solution**: Introduce PARALLEL INTERLEAVING to the pipeline

**EXAM TIP:** GPU training, multiple data files, want to decrease execution time → Parallel interleaving.

#### Accessing BigQuery Data

| Method                                | Works? | Best For                                            |
| ------------------------------------- | ------ | --------------------------------------------------- |
| BigQuery Python client library        | YES    | Direct SQL queries                                  |
| TensorFlow I/O BigQuery Reader        | YES    | Large datasets, streaming                           |
| BigQuery I/O Connector                | YES    | Dataflow pipelines                                  |
| BigQuery cell magic (`%%bigquery df`) | YES    | Notebooks, small data (< 1GB)                       |
| tf.data.Iterator                      | NO     | This just traverses datasets, doesn't connect to BQ |

**EXAM TIP:** 500MB from BigQuery in notebook → Use BigQuery cell magic (`%%bigquery`).  
**EXAM TIP:** Millions of records from BigQuery → TensorFlow I/O BigQuery Reader or convert to TFRecords.  
**COMMON TRAP:** tf.data.Iterator is NOT suitable for accessing BigQuery data.

### 2.2 TENSORFLOW TOOLS

#### Vertex AI Workbench for Exploratory Data Analysis

For generating one-time reports with data exploration, visualizations, and statistical analysis:

- **Best Tool**: Vertex AI Workbench user-managed notebooks
- **Why**: Maximum flexibility, can visualize distributions, run sophisticated stats
- **Data Size**: Works well with medium-sized datasets (~10GB from BigQuery)

| Tool                          | Best For                              | Limitations                  |
| ----------------------------- | ------------------------------------- | ---------------------------- |
| Vertex AI Workbench notebooks | EDA, visualizations, one-time reports | Manual process               |
| Data Studio                   | Dashboards, ongoing reporting         | Limited ML-specific analysis |
| Dataprep                      | Data cleaning                         | Not for statistical analysis |
| TFDV on Dataflow              | Schema validation, data drift         | Not for ad-hoc exploration   |

**EXAM TIP:** One-time EDA report with visualizations and stats for ML engineers → Vertex AI Workbench notebooks.

#### TensorBoard

Visualization toolkit for TensorFlow training.

- **Features**: Loss curves, metrics, model graphs, histograms, image visualization, embedding projector

#### TFProfiler

Performance profiling tool to identify training bottlenecks.

- **Features**: GPU/TPU utilization, input pipeline analysis, operation-level timing
- **Cloud TPU Profiler**: TensorBoard plugin for analyzing TPU performance

**EXAM TIP:** Monitor model training performance and find bottlenecks → TFProfiler.

#### What-If Tool (WIT)

Interactive tool for exploring model behavior without writing code.

- **Features**: Feature importance, fairness analysis, counterfactual testing, ROC curve visualization

#### LIT (Language Interpretability Tool)

Interactive tool specifically for NLP model exploration.

- **Features**: Attention visualization, saliency maps, counterfactual editing
- **Use Case**: Demonstrating how NLP model captures sentiment to stakeholders

**EXAM TIP:** NLP model demo for stakeholders → LIT (Language Interpretability Tool).

### 2.3 TENSORFLOW LIBRARIES

#### TensorFlow Hub

Repository of pre-trained models and reusable model components.

- **Available Models**: BERT, R-CNN, EfficientNet, MobileNet, Universal Sentence Encoder
- **Use Case**: Transfer learning, quick prototyping with pre-trained models

**EXAM TIP:** Need pre-trained R-CNN model → TensorFlow Hub.

#### TensorFlow Probability

Library for probabilistic reasoning and statistical analysis within TensorFlow.

- **Features**: Bayesian inference, probabilistic layers, MCMC, distributions
- **Use Case**: Combining traditional statistical methods with ML/AI processes

**EXAM TIP:** Integrate statistical analysis with ML → TensorFlow Probability.

#### TensorFlow Enterprise

Enterprise-grade distribution with Google support.

- **Features**: Long-term support, cloud-optimized, engineer-to-engineer assistance
- **Key Benefit**: Direct support from both Google Cloud AND TensorFlow teams

**EXAM TIP:** Need engineer-to-engineer assistance from Google Cloud and TensorFlow teams → TensorFlow Enterprise.

#### TensorFlow I/O

Extensions for additional file formats and data sources.

- **Supported Formats**: Parquet, Avro, Kafka, AWS S3, Azure Blob
- **Benefit**: No additional infrastructure needed for these formats

**EXAM TIP:** Read Parquet files in TensorFlow without extra infrastructure → TensorFlow I/O.

### 2.4 EAGER VS GRAPH MODE

#### Understanding the Modes

- **Eager Mode (Default in TF2)**: Operations execute immediately. Great for debugging and development.
- **Graph Mode**: Operations are compiled into a computation graph. Faster, optimized for production.

#### Converting to Graph Mode for Production

Don't deploy models in eager mode. Convert to graph mode for better performance:

- Use tf.function decorator (most common approach)
- Create tf.Graph explicitly
- Export as SavedModel (uses graph internally)

**COMMON TRAP:** eager_execution=no is NOT a valid parameter.  
**EXAM TIP:** Production deployment → Use tf.function or tf.Graph (not eager mode).

### 2.5 DISTRIBUTED TRAINING STRATEGIES

| Strategy                    | Scope                                | When to Use                  |
| --------------------------- | ------------------------------------ | ---------------------------- |
| MirroredStrategy            | Single machine, multiple GPUs        | Most common multi-GPU setup  |
| MultiWorkerMirroredStrategy | Multiple machines, multiple GPUs     | Distributed cluster training |
| TPUStrategy                 | TPU pods                             | TPU-specific training        |
| ParameterServerStrategy     | Multiple workers + parameter servers | Very large models            |
| OneDeviceStrategy           | Single device                        | Testing/debugging only       |

**EXAM TIP:** Multiple GPUs on single machine → MirroredStrategy.

#### When multi-GPU doesn’t speed things up

If you switch from 1 GPU to multiple GPUs (e.g., MirroredStrategy) and training time does not improve:

- **Common cause**: Input pipeline is the bottleneck (GPU(s) waiting on data)
- **Common cause**: Global batch size is too small (too much overhead per step)
- **Fix**: Optimize `tf.data` (parallel interleave/prefetch/cache) and **increase batch size** (then tune learning rate if needed)

**EXAM TIP:** Multi-GPU with no speedup → usually **input-bound** or **batch too small**; optimize `tf.data` and increase batch size.

### 2.6 TENSORFLOW ESTIMATORS

High-level TensorFlow API that simplifies model training, evaluation, and serving.

- **Benefits**: Built-in distributed training, easy export to SavedModel, works with AI Platform
- **Migration**: TensorFlow Estimator code can be ported to AI Platform with minimal refactoring

**EXAM TIP:** TF Estimators on-premises, minimize code refactoring for GCP → AI Platform distributed training.  
**COMMON TRAP:** For TPU training, your code MUST be wrapped in Estimator or use tf.distribute.

### 2.7 MODEL OPTIMIZATION FOR INFERENCE

Techniques to reduce model inference latency without retraining:

| Technique                          | Description                               |         Speedup | Requires Retraining?       |
| ---------------------------------- | ----------------------------------------- | --------------: | -------------------------- |
| Dynamic Range Quantization         | FP32 → INT8 conversion                    |            2-4x | NO - first try for latency |
| Weight Pruning                     | Remove small/zero weights                 |        Variable | Sometimes                  |
| Model Distillation                 | Train smaller student from larger teacher | Depends on size | YES                        |
| Recompile TF Serving with CPU opts | CPU-specific optimizations                |        Variable | NO (recompile only)        |

**EXAM TIP:** Reduce latency 50% WITHOUT retraining → Try Dynamic Range Quantization first.  
**EXAM TIP:** Latency issues on CPU-only GKE pods → Recompile TensorFlow Serving with CPU-specific optimizations.

#### Quick latency reduction when the model uses high precision

If a TF model is using unnecessarily high precision (e.g., `tf.float64`) and you need a small latency win with minimal quality loss:

- **First try**: Reduce precision / apply post-training quantization (e.g., FP16 or dynamic range quantization) before changing serving hardware

**EXAM TIP:** Need a quick serving latency improvement without retraining → try **quantization / lower precision** first (don’t jump to GPU serving).

### 2.8 TFX (TensorFlow Extended)

End-to-end platform for production ML pipelines.

| Component        | Purpose             | Key Function                                        |
| ---------------- | ------------------- | --------------------------------------------------- |
| ExampleGen       | Data ingestion      | Reads and splits data                               |
| StatisticsGen    | Compute statistics  | Profile your data                                   |
| SchemaGen        | Infer schema        | Define expected data types and ranges               |
| ExampleValidator | Data validation     | Check for anomalies and skew                        |
| Transform        | Feature engineering | preprocessing_fn() - PREVENTS TRAINING-SERVING SKEW |
| Trainer          | Model training      | run_fn() executes training                          |
| Evaluator        | Model evaluation    | Validates model meets thresholds                    |
| Pusher           | Model deployment    | Deploys model if blessed by Evaluator               |

**CRITICAL - Transform Component:** Saves preprocessing as a TensorFlow graph. This ensures the SAME preprocessing is applied at both training AND serving time, preventing training-serving skew.

**EXAM TIP:** More control over entire ML lifecycle → TFX.

#### Mapping “continuous validation / drift / fairness” tools (tested)

- **TFDV / ExampleValidator**: schema/anomalies + drift/skew signals for input data
- **TFT / Transform**: embeds feature transformations into the model artifact (prevents training-serving skew)
- **TFMA**: model evaluation at scale; can compute slicing + fairness metrics (Fairness Indicators)

**EXAM TIP:** “Continuous validation of input schema + drift/skew detection” → **TFDV / ExampleValidator**.  
**EXAM TIP:** “Quantify bias across subgroups (FPR/FNR gaps)” → **TFMA + Fairness Indicators**.

#### TFX Model Validator

- **Purpose**: Validate that model meets performance thresholds before production deployment
- **Use Case**: Track performance on specific subsets before pushing to production

**EXAM TIP:** Validate model on specific data subsets before production → TFX ModelValidator.

#### Inference in Dataflow (tfx_bsl.public.beam.RunInference)

Run TensorFlow inference within a Dataflow (Apache Beam) pipeline.

- **Use Case**: Streaming prediction with preprocessing in Dataflow

**EXAM TIP:** BQML model inference in Dataflow pipeline → Export to TensorFlow + RunInference step.

#### TFMA evaluation at scale (and avoiding OOM)

If evaluation in a pipeline fails due to memory constraints but you want minimal infra overhead:

- Run TFMA evaluation as an Apache Beam job on **DataflowRunner**

**EXAM TIP:** Evaluation step OOM in pipeline → run TFMA on **DataflowRunner** (not custom VMs/GKE unless required).

### 2.9 TENSORFLOW DATA VALIDATION (TFDV)

Library for analyzing and validating ML data.

- **Features**: Schema inference, anomaly detection, data drift detection, skew detection
- **Use Case**: Third-party data with unreliable formatting - detect when schema changes

**EXAM TIP:** Third-party data broker doesn't notify of format changes → TFDV to detect schema anomalies.

**EXAM TIP:** “Detect and visualize data anomalies” in an ML workflow → TFDV (often surfaced via the ML platform tooling, e.g., Vertex/AI Platform).
