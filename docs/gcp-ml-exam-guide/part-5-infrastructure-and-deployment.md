## PART V: INFRASTRUCTURE & DEPLOYMENT

**Scope note (important):** This part covers **infrastructure** for _both_ classic predictive ML and GenAI, but they have different bottlenecks and patterns.

- **Predictive ML infrastructure (5.0–5.4)**: storage/security, hardware, cost, and classic model serving (e.g., TensorFlow Serving).
- **GenAI / LLM infrastructure (5.5)**: token-by-token generation, KV cache, batching/schedulers, and LLM-serving patterns.

### Table of Contents

- [Predictive ML infrastructure](#predictive-ml-infrastructure-50-54)
  - [5.0 STORAGE + SECURITY BASICS](#50-storage--security-basics-often-tested)
  - [5.1 HARDWARE SELECTION](#51-hardware-selection)
  - [5.2 COMMON ERRORS](#52-common-errors)
  - [5.3 COST OPTIMIZATION](#53-cost-optimization)
  - [5.4 TENSORFLOW SERVING OPTIMIZATION](#54-tensorflow-serving-optimization)
- [GenAI / LLM infrastructure](#genai--llm-infrastructure-55)
  - [5.5 LLM SERVING (GenAI production patterns)](#55-llm-serving-genai-production-patterns)

### Official docs (high-signal starting points)

- **Cloud Storage**: [cloud.google.com/storage/docs](https://cloud.google.com/storage/docs)
- Storage Classes: [Storage Classes Documentation](https://cloud.google.com/storage/docs/storage-classes)
- **Cloud KMS** (encryption keys): [cloud.google.com/kms/docs](https://cloud.google.com/kms/docs)
- **IAM**: [cloud.google.com/iam/docs](https://cloud.google.com/iam/docs)
- **Cloud Audit Logs**: [Audit Logs Documentation](https://cloud.google.com/logging/docs/audit)
- **GPU/TPU on Compute Engine**: [GPU Documentation](https://cloud.google.com/compute/docs/gpus)
- **TPU**: [cloud.google.com/tpu/docs](https://cloud.google.com/tpu/docs)
- **Deep Learning VM Images**: [Deep Learning VM Documentation](https://cloud.google.com/deep-learning-vm/docs)
- **Preemptible VMs**: [Preemptible Instances](https://cloud.google.com/compute/docs/instances/preemptible)
- **TensorFlow Serving**: [TensorFlow Serving Guide](https://www.tensorflow.org/tfx/guide/serving)
- **Triton Inference Server on GKE**: [Serve TensorFlow Model on GKE](https://cloud.google.com/kubernetes-engine/docs/tutorials/serve-tensorflow-model)

### Predictive ML infrastructure (5.0–5.4)

### 5.0 STORAGE + SECURITY BASICS (often tested)

#### Storage classes (Cloud Storage)

- **Nearline/Coldline**: infrequent access, lower cost, good for large training corpora when retrieval is occasional
- **Archive**: lowest cost, very infrequent access (long-term archival)

**EXAM TIP:** Multi-petabyte raw logs, cost is top priority, retrieval not critical → **Cloud Storage Archive** (or Coldline/Nearline if accessed more often).

#### Encryption keys

- **GMEK**: Google-managed (default)
- **CMEK (Cloud KMS)**: customer-managed rotation + auditability

**EXAM TIP:** Need control over rotation schedule + audit trail → **CMEK via Cloud KMS**.

#### Integrity verification (hashes/checksums)

To verify data integrity for objects in storage, use object checksums/hashes (commonly **CRC32C** / **MD5**) provided at the storage layer.

**EXAM TIP:** “Generate hash values for data verification” → Cloud Storage **object checksums** (not IAM/DLP; KMS is key management).

#### Auditing access (compliance)

- **Cloud Audit Logs**: audit access to services and resources for compliance tracking

**EXAM TIP:** “Audit access to data / track compliance” → **Cloud Audit Logs**.

#### Privacy & compliance quick map

- **GDPR (EU)**: personal data protection; includes rights like **data portability** and **right to be forgotten**; max fine commonly tested as **4% global annual turnover or €20M**
- **CCPA (California)**: gives California residents more control over personal data
- **HIPAA**: health information privacy/security
- **PCI-DSS**: payment card industry security standard

**EXAM TIP:** GDPR data portability → right to receive personal data in a structured, commonly used format and transfer it.  
**EXAM TIP:** “Right to be forgotten” → data erasure right under GDPR.

#### Privacy principles and controls

- **Data anonymization**: remove/transform PII so individuals are not identifiable
- **Data minimization**: collect only what is necessary for a specific purpose
- **Confidentiality**: only authorized individuals can access the data
- **Encryption in transit**: protects data while being transmitted

### 5.1 HARDWARE SELECTION

#### GPU vs TPU Selection

| Scenario                                    | Best Hardware | Reason                             |
| ------------------------------------------- | ------------- | ---------------------------------- |
| CNN without Estimator/tf.distribute wrapper | GPU           | TPU requires specific API patterns |
| TensorFlow with Estimator API               | TPU           | Optimal for TF at scale            |
| PyTorch training                            | GPU           | Limited TPU support                |
| scikit-learn training slow                  | CPU (DLVM)    | sklearn doesn't use GPU            |
| Complex matrix operations, weeks to train   | TPU           | Best for matrix-heavy workloads    |

**COMMON TRAP:** CNN code NOT wrapped in Estimator → Must use GPU, not TPU.  
**COMMON TRAP:** scikit-learn does NOT benefit from GPU.

**EXAM TIP:** Speed up scikit-learn → Deep Learning VM (DLVM) with NumPy/SciPy optimizations (not GPU).

#### Deep Learning VM (DLVM)

Pre-configured VM images with ML frameworks and dependencies.

- **Benefit**: All libraries pre-installed, optimized for ML workloads
- **When to Use**: Quick experimentation, need GPU but not complex infrastructure

**EXAM TIP:** CNN training, easiest setup → Deep Learning VM with GPU.

#### Extremely large models / large batch sizes (memory-driven hardware choice)

If the model is very large (multi‑GB weights/embeddings) and/or batch sizes are huge, your primary constraint is **device memory**:

- Prefer GPUs with large HBM (e.g., A100) when you need to fit very large models/batches
- TPU can be excellent for TF workloads, but custom/experimental ops and memory constraints often push you toward high-memory GPUs

**EXAM TIP:** Very large model + very large batch → choose **high-memory GPU** hardware (e.g., A100-class), not CPU-only machines.

### 5.2 COMMON ERRORS

#### GPU Not Found Error

- **Error**: "The resource nvidia-tesla-k80 was not found"
- **Cause**: The GPU type is not available in the selected REGION
- **Solution**: Check GPU availability in region, not just quota

**COMMON TRAP:** GPU not found error = region availability issue, NOT quota issue.

**EXAM TIP:** If the error explicitly says the _acceleratorType was not found_, pick a different **zone/region** or a different **GPU type** that is offered there.  
**COMMON TRAP:** Quota errors look different (they mention **quota**). “Not found” usually means **not offered** in that location.

#### Out of Memory (OOM) During Prediction

- **Cause**: Prediction request batch too large for available memory
- **Solution**: Send smaller batch of instances
- **NOT**: Quota increase

#### Out of Memory (OOM) During Training

When training a model on GPU and you encounter OOM errors:

- **Cause**: Batch size too large for GPU memory
- **Solution**: DECREASE BATCH SIZE
- **NOT**: Change learning rate
- **NOT**: Change optimizer

| OOM Context                 | Cause                                   | Solution                          |
| --------------------------- | --------------------------------------- | --------------------------------- |
| During TRAINING             | Batch size too large for GPU            | Decrease batch size               |
| During PREDICTION           | Request batch too large                 | Send smaller request batch        |
| During TRAINING with images | Image resolution too high + large batch | Decrease batch size OR image size |

**EXAM TIP:** OOM during training with GPU → Decrease batch size.

### 5.3 COST OPTIMIZATION

#### Training Cost Optimization

- Use AI Platform Training with appropriate scale-tier
- Use preemptible VMs for long-running, interruptible tasks
- Use preemptible TPUs for cost-effective TPU training

**EXAM TIP:** Long-running training + checkpoints + want lowest cost → **preemptible TPU/VM WITH checkpoints** (preemption is fine if you can resume).

#### Preemptible VMs with Checkpoints

Preemptible VMs are 60-80% cheaper than regular VMs but can be interrupted at any time.

**CRITICAL:** When using preemptible VMs, you MUST use checkpoints to save training progress.

- **Pattern**: Save model checkpoints periodically → VM preempted → Resume from last checkpoint
- **Without checkpoints**: All training progress is lost when VM is preempted

| Configuration                       | Cost           | Risk                                 |
| ----------------------------------- | -------------- | ------------------------------------ |
| Preemptible VMs WITH checkpoints    | 60-80% savings | Can resume training after preemption |
| Preemptible VMs WITHOUT checkpoints | 60-80% savings | LOSE ALL PROGRESS if preempted       |
| Regular VMs                         | Full price     | No interruption risk                 |

**EXAM TIP:** Reduce training costs for weeks-long training → Preemptible VMs WITH checkpoints.  
**COMMON TRAP:** NEVER use preemptible VMs without checkpoints for training jobs.

#### Training Infrastructure for Daily Retraining

When retraining models daily (e.g., EfficientNet image classifier):

| Option                                           | Cost               | Best For                        |
| ------------------------------------------------ | ------------------ | ------------------------------- |
| AI Platform Training + V100 GPUs + Cloud Storage | Optimal            | Daily retraining, minimize cost |
| Deep Learning VM + GPUs + Cloud Storage          | Higher (always on) | Interactive development         |
| Deep Learning VM + local storage                 | Data loss risk     | NOT recommended                 |
| GKE cluster + NFS                                | Complex            | Only if K8s expertise exists    |

**EXAM TIP:** Daily EfficientNet retraining, minimize cost → AI Platform Training + V100 GPUs + Cloud Storage.  
**COMMON TRAP:** Don't use local storage for training data - use Cloud Storage for durability.

#### Notebook Cost Optimization

- Treat notebooks as ephemeral instances
- Set up automatic shutdown routines
- Monitor GPU usage with alerts

### 5.4 TENSORFLOW SERVING OPTIMIZATION

#### CPU-Only Infrastructure

When you're running on CPU-only infrastructure (e.g., GKE with CPU pods) and have latency issues:

- **Solution**: Recompile TensorFlow Serving with CPU-specific optimizations
- **Additional**: Configure GKE to use appropriate CPU platform for serving nodes

**EXAM TIP:** Latency issues on CPU-only GKE pods → Recompile TF Serving for CPU optimizations.

#### Ultra-low latency online serving for large deep models

For very large deep models (hundreds of millions of parameters) with strict latency (e.g., <50ms):

- Use **GPU-enabled online endpoints** and optimized serving runtimes (e.g., NVIDIA Triton / optimized TF Serving)

**EXAM TIP:** “<50ms online predictions for large deep model” → GPU endpoint + optimized serving (not batch prediction, not Cloud Functions).

### GenAI / LLM infrastructure (5.5)

### 5.5 LLM SERVING (GenAI production patterns)

LLMs introduce new serving bottlenecks (token-by-token decode, KV cache growth, and latency/cost tradeoffs).

#### High-throughput LLM inference engines (concepts)

Even if the exam doesn’t name a specific engine, these patterns often appear:

- **Continuous batching**: batch requests dynamically to keep GPUs busy (throughput + cost wins).
- **KV cache management**: KV cache grows with context length; paging/eviction matters to avoid OOM.
- **Prefill vs decode scheduling**: prefill is throughput-heavy; decode is latency-sensitive; schedulers try to keep decode responsive.
- **Context caching (prefix caching)**: reuse KV caches for shared prefixes (system prompts, long docs) to reduce repeated prefill.

**EXAM TIP:** If the requirement is “serve LLM efficiently at scale” → look for answers that mention **batching + cache + scheduler** (not “just add bigger GPUs”).
