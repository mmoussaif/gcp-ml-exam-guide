# GenAI System Design Cheat Sheet

---

## 1. ML FOUNDATIONS

_AI is built on ML; skipping fundamentals leads to technical debt, bottlenecks, and broken apps. This section follows the article structure: **Intelligence & Models**, **3 Ways Computers Can Learn** (ML → DL → RL), **Data**, plus vocabulary and GenAI links._

### Intelligence & models

- **Intelligence** = having an internal model of the world that lets you make predictions. Better model → more accurate predictions.
- **Model** = something that lets you make predictions (e.g. a function with parameters). Computers learn models from data; humans learn from experience and others.
- In traditional software, programmers write explicit rules. In **ML**, programmers curate examples and the computer learns from them.

### Two phases of ML (Machine Learning)

| Phase         | What it is                                                                                                                                                                                            | GenAI analogue                                                     |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Training**  | Fit model to reality: choose parameters that minimize **loss** (discrepancy between predictions and data). Uses a **training dataset**, **loss function**, and **optimizer** (e.g. gradient descent). | Pretraining, SFT, RLHF training                                    |
| **Inference** | Apply the trained model to **new data** to make predictions. No parameter updates.                                                                                                                    | What happens when you call an LLM API or run a model in production |

**Core idea:** Training = find parameters (e.g. via gradient of loss) so predictions match reality. Inference = use those parameters on new inputs.

### Deep Learning (DL)

- **DL** = training **neural networks** that learn useful **features** from raw data without hand-crafted feature engineering.
- **Neuron** = linear combination of inputs + **nonlinear activation** (e.g. ReLU). Stacking neurons in **layers** lets NNs approximate complex functions.
- **Training NNs:** Loss is usually **non-convex** → use **gradient descent** (or variants) to iteratively update parameters; **optimizer** and **hyperparameters** (learning rate, batch size) guide the process.
- **Feature engineering** = manually defining input variables; important in classical ML, less so in DL (NNs learn features).

### Neurons, activations, layers, networks (zoo)

**Neurons**

| Name        | Description                                                                   |
| ----------- | ----------------------------------------------------------------------------- |
| **Vanilla** | Basic unit computing weighted sum + activation; used in FFNNs and CNNs        |
| **LSTM**    | Advanced neuron with memory and gates for long-term dependencies in sequences |

**Activations**

| Name        | Description                                                                           |
| ----------- | ------------------------------------------------------------------------------------- |
| **ReLU**    | max(0, x); fast, widely used in deep networks                                         |
| **Sigmoid** | S-shaped, output in (0, 1); used in binary classification                             |
| **Tanh**    | Like sigmoid but centered at 0; output in (-1, 1)                                     |
| **Softmax** | Outputs a probability distribution; used in final layer of multi-class classification |

**Layers**

| Name                | Description                                                                   |
| ------------------- | ----------------------------------------------------------------------------- |
| **Fully Connected** | Standard layer where each neuron connects to all inputs                       |
| **Recurrent**       | Maintains memory across timesteps; used in RNNs, LSTMs                        |
| **Convolutional**   | Extracts spatial features using filters; used in image data                   |
| **Attention**       | Computes weighted importance of different inputs; key in Transformers         |
| **Pooling**         | Downsamples spatial data; used in CNNs to reduce size and noise               |
| **Normalization**   | Stabilizes training by normalizing activations; includes BatchNorm, LayerNorm |
| **Dropout**         | Randomly deactivates neurons during training to prevent overfitting           |

**Networks**

| Name                   | Description                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| **Feedforward (FFNN)** | Basic architecture with no loops; used for static input–output tasks                        |
| **RNN**                | Handles sequential data using recurrence; remembers previous inputs                         |
| **CNN**                | Uses convolutions to process grid-like data such as images                                  |
| **Transformer**        | Uses self-attention to model sequences without recurrence; state-of-the-art in NLP & beyond |

### Reinforcement Learning (RL)

- **RL** = learning through **trial and error**; no ground-truth labels—only a **reward signal** (e.g. win/lose, human preference).
- **Key difference vs supervised:** In supervised we minimize loss vs explicit targets. In RL we **maximize** cumulative reward; updates use **gradient ascent** (e.g. REINFORCE).
- **Examples:** AlphaGo (learned by playing itself); **RLHF** (reward from human preferences to align LLMs); o1 / deep research (RL-based reasoning).

### Popular RL algorithms

| Name                                          | Key idea                                                                           | Objective / note                                                                                                                         |
| --------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **REINFORCE**                                 | Monte Carlo policy gradient using full returns                                     | Maximize \( J(\theta) = \mathbb{E}_t[\nabla_\theta \log \pi\_\theta(a_t \mid s_t) \cdot R_t] \)                                          |
| **TRPO** (Trust Region Policy Optimization)   | Constrain policy updates with a KL-divergence trust region for stable improvements | Subject to \( \mathbb{E}_t[D_{KL}(\pi*{\theta*{\text{old}}} \| \pi\_\theta)] \leq \delta \)                                              |
| **PPO** (Proximal Policy Optimization)        | Approximates TRPO with a clipped surrogate objective; practical and stable         | Clip ratio \( r*t(\theta) = \pi*\theta(a*t \mid s_t) / \pi*{\theta\_{\text{old}}}(a_t \mid s_t) \) to \([1-\varepsilon, 1+\varepsilon]\) |
| **GRPO** (Group Relative Policy Optimization) | Encourages exploration by optimizing relative advantages within action groups      | Used in LLM alignment (e.g. group-wise advantage); no global baseline                                                                    |

### Data: quantity and quality

- **Quantity:** More (good) data usually helps; **insufficient data** is a leading cause of **overfitting** (model memorizes training set, fails on new data).
- **Quality:** **Accuracy** (labels and values correct) and **diversity** (data covers the scenarios where the model must work). “Garbage in, garbage out.”
- **Train / validation / test:** Train = fit model; validation = tune hyperparameters / early stopping; test = final evaluation on held-out data.

### Summary: 3 ways computers learn

| Way    | What it is                                                                         | Key idea                                                              | GenAI analogue                           |
| ------ | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------- |
| **ML** | Learn tasks from data; two phases: training (minimize loss) + inference (predict). | Curate examples; computer fits model via loss + gradient.             | Pretraining, SFT; inference = API calls. |
| **DL** | ML using neural networks that learn features automatically.                        | NNs = stacked neurons + nonlinearity; optimize with gradient descent. | LLMs, diffusion models are deep nets.    |
| **RL** | Learn from trial and error using a reward signal (no ground truth).                | Maximize reward; e.g. REINFORCE, PPO.                                 | RLHF, preference-based alignment.        |

### By learning signal (supervised / unsupervised / reinforcement)

| Way               | What it is                                              | Data                  | Goal                             | GenAI analogue                                                     |
| ----------------- | ------------------------------------------------------- | --------------------- | -------------------------------- | ------------------------------------------------------------------ |
| **Supervised**    | Learn from labeled examples (input → correct output)    | Labeled pairs         | Minimize error on labels         | Fine-tuning (SFT): (prompt, response) pairs                        |
| **Unsupervised**  | Learn structure/patterns without labels                 | Unlabeled data        | Compression, clustering, density | Pretraining: next-token prediction on raw text (no “answer” label) |
| **Reinforcement** | Learn from trial and reward (agent acts, gets feedback) | Rewards / preferences | Maximize long-term reward        | RLHF: human preferences → reward model → policy update             |

**For builders:** (1) **Use a pre-trained model + prompting** — no training; fast. (2) **Fine-tune** — adapt with your data (SFT, LoRA). (3) **Train from scratch** — data curation, pretraining, scaling (rare; use only when necessary).

### Popular supervised learning techniques

| Name                             | Description                                                                           | Loss function                                                | Type           |
| -------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------ | -------------- |
| **Linear Regression**            | Predicts continuous output by fitting a linear relationship between inputs and output | Mean Squared Error (MSE)                                     | Regression     |
| **Logistic Regression**          | Models the probability of a binary outcome using a logistic (sigmoid) function        | Binary Cross-Entropy (Log Loss)                              | Classification |
| **Decision Tree**                | Splits data into branches based on feature values to make predictions                 | Impurity measures (e.g. Gini, Entropy, MSE)                  | Both           |
| **Random Forest**                | Ensemble of decision trees averaged (regression) or voted (classification)            | Same as Decision Tree                                        | Both           |
| **XGBoost**                      | Gradient boosting framework that builds trees sequentially to correct prior errors    | Customizable; often Log Loss or MSE                          | Both           |
| **SVM** (Support Vector Machine) | Finds the optimal hyperplane that separates classes or fits data                      | Hinge loss (classification), ε-insensitive loss (regression) | Both           |

### Intelligence and models (cheat-sheet view)

| Concept            | Meaning                                                                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Model**          | A parameterized function (e.g. neural net) whose parameters are **learned from data**; inference = run the function on new inputs. |
| **Training**       | Update parameters to minimize **loss** (error on data); needs data, compute, and an optimizer.                                     |
| **Inference**      | Run the trained model on new inputs; no parameter updates; this is what you pay for in APIs.                                       |
| **Generalization** | Model performs well on **unseen** data; the opposite of memorizing the training set.                                               |
| **Overfitting**    | Model fits training data too closely and fails on new data; fix with more data, regularization, or simpler models.                 |
| **Underfitting**   | Model too simple to capture patterns; high error on train and test; fix with more capacity or more training.                       |
| **Loss**           | Scalar that measures error (e.g. cross-entropy, MSE); training minimizes it.                                                       |
| **Optimizer**      | Algorithm that updates parameters from gradients (e.g. Adam, SGD); learning rate controls step size.                               |
| **Epoch**          | One full pass over the training data.                                                                                              |
| **Batch**          | Subset of data used for one gradient update; batch size trades off speed vs stability.                                             |

### Key ML vocabulary (cheat-sheet)

| Term                     | Meaning                                                                      |
| ------------------------ | ---------------------------------------------------------------------------- |
| **Parameters / weights** | Numbers inside the model learned during training; 7B model = 7B parameters.  |
| **Hyperparameters**      | Settings you choose (learning rate, batch size, layers); not learned.        |
| **Gradient**             | Direction to change parameters to reduce loss; backprop computes it.         |
| **Learning rate**        | Step size for parameter updates; too high = unstable; too low = slow.        |
| **Regularization**       | Penalties (e.g. weight decay, dropout) to reduce overfitting.                |
| **Train / val / test**   | Train = fit model; validation = tune/early-stop; test = final eval (unseen). |

### Common optimizers

| Name                                  | Description                                                          | Data sampled               | Key hyperparameters                                       |
| ------------------------------------- | -------------------------------------------------------------------- | -------------------------- | --------------------------------------------------------- |
| **Gradient Descent**                  | Updates parameters using the gradient on the entire dataset          | Full dataset (batch)       | Learning rate (γ)                                         |
| **SGD** (Stochastic Gradient Descent) | Updates parameters using the gradient from a single random example   | One data point at a time   | Learning rate (γ)                                         |
| **Mini-Batch Gradient Descent**       | Combines efficiency and noise reduction using small random batches   | Batch of data (e.g. 8, 16) | Learning rate (γ), batch size                             |
| **Adam** (Adaptive Moment Estimation) | Combines momentum and RMSProp; adaptive learning rates per parameter | Mini-batches (usually)     | Learning rate (γ), β₁ (momentum), β₂ (RMS), ε (stability) |

### Common hyperparameters

| Name                  | Description                                                                    |
| --------------------- | ------------------------------------------------------------------------------ |
| **Epoch**             | Number of times the entire dataset is passed through the model during training |
| **Learning rate (γ)** | Controls how much to adjust weights with respect to the gradient               |
| **Batch size**        | Number of samples used to compute each update to the model’s weights           |
| **Dropout**           | Fraction of neurons randomly set to 0 during training to prevent overfitting   |

### For builders (three options)

| Option                      | What you do                             | When to use                                 |
| --------------------------- | --------------------------------------- | ------------------------------------------- |
| **Pre-trained + prompting** | Use an existing model; no training      | Fast iteration, general tasks, limited data |
| **Fine-tune**               | Adapt with your data (SFT, LoRA, QLoRA) | Custom style/domain, RAG not enough         |
| **Train from scratch**      | Data curation, pretraining, scaling     | Rare; only when you need a new foundation   |

### Link to GenAI

- **Pretraining** = unsupervised-style (next-token prediction; no explicit labels). **SFT** = supervised (instruction/response pairs). **RLHF** = reinforcement (preference reward).
- When you **prompt** an LLM, you’re doing **inference**. When you **fine-tune**, you’re doing **training** (usually supervised). Keeping this straight avoids confusion about cost, latency, and where failures come from.

---

## 2. CORE COMPONENTS

| Component          | What It Is                                   | Purpose                              | Key Choice                   |
| ------------------ | -------------------------------------------- | ------------------------------------ | ---------------------------- |
| **LLM**            | Large language model (GPT, Gemini, Claude)   | Text generation, reasoning, tool use | Size vs cost vs latency      |
| **Embeddings**     | Dense vector representation of text/images   | Semantic similarity, retrieval       | Dimension (768-3072), model  |
| **Vector DB**      | Database optimized for similarity search     | Store & retrieve embeddings          | FAISS, Pinecone, Vertex AI   |
| **Tokenizer**      | Splits text into tokens (BPE, SentencePiece) | Model input preprocessing            | Vocab size, subword handling |
| **Prompt**         | Instructions + context sent to LLM           | Control model behavior               | System prompt, few-shot      |
| **Context Window** | Max tokens model can process (4K-1M)         | Limits input + output length         | Cost grows with length       |
| **KV Cache**       | Stores computed attention keys/values        | Avoid recomputation                  | Grows with context length    |

## 3. ARCHITECTURES

### Where they sit (taxonomy)

- **Transformer** = the base architecture (2017): stacked layers of **self-attention** + **feed-forward (FFN)**. Everything below is built on it.
- **Decoder-only, encoder-only, encoder-decoder** = **Transformer variants** (which stacks you use and how attention is masked):
  - They are **not** separate architectures; they are **configurations** of the Transformer: use only the decoder stack (causal), only the encoder stack (bidirectional), or both with cross-attention.
  - So: same building blocks (attention, FFN), different **layout** and **attention pattern**.
- **MoE** = a **technique** applied _inside_ a Transformer layer, not a replacement for it:
  - You keep a standard variant (usually **decoder-only**). In each layer, the **FFN block** is replaced by a **MoE block**: a small router + many “expert” FFNs; only top-K experts run per token.
  - So: MoE is a **submodel/module** (the FFN is replaced by router + experts); the rest (embedding, attention, prediction head) is unchanged. Mixtral is a **decoder-only Transformer with MoE FFNs**.

**One-line:** Transformer = base. Decoder/encoder/encoder-decoder = variants (which stack + attention pattern). MoE = technique that swaps the FFN for a sparse MoE block inside a Transformer layer.

```
Transformer (base)
├── Variants (which stack + attention)
│   ├── Decoder-only   → causal self-attention only (GPT, LLaMA, Claude)
│   ├── Encoder-only   → bidirectional self-attention only (BERT)
│   └── Encoder-decoder → encoder (bidir.) + decoder (causal + cross-attn) (T5)
└── Optional modification inside a variant
    └── MoE → replace dense FFN with router + expert FFNs (Mixtral, Gemini 1.5)
```

```
DECODER-ONLY          ENCODER-ONLY           ENCODER-DECODER         MoE (Mixture of Experts)
(GPT, LLaMA, Gemini)  (BERT, RoBERTa)        (T5, BART)              (Mixtral, Gemini 1.5)
─────────────────     ─────────────────      ─────────────────       ─────────────────────
Generates text        Understands text       Transforms text         Sparse activation
Causal attention      Bidirectional          Cross-attention         Top-K experts per token
→ Chatbots, code      → Classification       → Translation           → High capacity, low cost
```

### What each architecture does

| Architecture        | What it does                                                                                          | Attention                                                                | Best for                                           | Examples                              |
| ------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------- | ------------------------------------- |
| **Decoder-only**    | Generates output token-by-token; one stack of layers                                                  | **Causal**: each token sees only past tokens (no future)                 | Text generation, chatbots, code completion         | GPT-4, LLaMA, Claude, Gemini (text)   |
| **Encoder-only**    | Processes entire input at once; outputs a representation or label                                     | **Bidirectional**: each token sees all tokens                            | Classification, NER, embeddings, search, reranking | BERT, RoBERTa, DeBERTa                |
| **Encoder-decoder** | Encoder reads input; decoder generates output; they’re separate stacks                                | Encoder: bidirectional. Decoder: causal + **cross-attention** to encoder | Translation, summarization, Q&A as text-to-text    | T5, BART, mT5                         |
| **MoE**             | Decoder-only (or encoder-decoder) but FFN is replaced by many “experts”; router picks top-K per token | Same as base (causal); compute is **sparse** in the FFN                  | High capability without proportional compute cost  | Mixtral 8×7B, Gemini 1.5, DeepSeek-V2 |

**Attention in one line:** Causal = “see only the past” (for generation). Bidirectional = “see full sentence” (for understanding). Cross-attention = “decoder looks at encoder output” (for input→output tasks).

### Decoder-only (LLMs you use daily)

- **Flow:** Input tokens → embedding + position → many layers of (self-attention + FFN) → prediction head → next-token probabilities.
- **Self-attention:** Each position can attend only to positions **to the left** (causal mask). So the model can’t “see the future” and is suitable for autoregressive generation.
- **Use when:** Chat, code completion, any “generate the next token” task. This is the default for GPT, LLaMA, Claude, Gemini (for text).

### Encoder-only (understanding, not generation)

- **Flow:** Input tokens → embedding + position → many layers of (self-attention + FFN) → pooled or per-token representation (or classification head).
- **Self-attention:** Each position can attend to **all** positions (bidirectional). Full context in one pass.
- **Use when:** Classification, NER, embeddings (e.g. for RAG), or as a **cross-encoder** for reranking (query + document in one forward pass).

### Encoder-decoder (input → output transformation)

- **Flow:** Encoder: input → bidirectional self-attention → encoder output (context). Decoder: previous output tokens → causal self-attention → **cross-attention to encoder output** → FFN → next-token prediction.
- **Cross-attention:** Decoder token “asks” which encoder positions to focus on (e.g. “hello” → “bonjour”). Lets the decoder align output to input.
- **Use when:** Translation, summarization, or any task where input and output are different lengths and the model must “read then write.” T5 frames many tasks as text-to-text.

### MoE (Mixture of Experts) in more detail

- **Idea:** Keep one shared **attention** stack, but replace the single **FFN** with many small FFNs (“experts”). For each token, a small **router** (gating network) picks the **top-K experts** (e.g. K=2); only those run. Their outputs are combined by router weights.
- **Dense vs sparse:** Dense = every token uses the same full FFN. MoE = every token uses only a subset of experts → **capacity** (many params) with **cost** proportional to active params.
- **Numbers:** 8×7B with top-2 → 56B total params, ~14B active per token. So: “capacity of a large model, compute of a smaller one.”
- **Trade-offs:**
  - **Pros:** Higher quality at similar latency; better scaling.
  - **Cons:** Full model still in memory (all experts loaded); router can be imbalanced (load-balancing loss in training); fine-tuning is trickier.
- **When to choose:** Use **MoE** when you want more capability without a proportional latency increase. Use **dense** when memory is tight (e.g. edge) or you need simpler fine-tuning.
- **Notable MoE:** Mixtral 8×7B (8 experts, top-2, ~47B total, ~13B active); Mixtral 8×22B (~141B total, ~39B active); DeepSeek-V2 (236B total, ~21B active, 160 experts top-6); Gemini 1.5 (MoE-based); GPT-4 (rumored MoE).

**MoE key insight:** 8×7B model = 56B total params, but only ~14B active per token → capacity of large, cost of small. Memory is the catch: you still load all experts.

## 4. TRAINING STAGES

```
PRETRAINING              SFT                      RLHF
(Trillions tokens)       (10K-100K pairs)         (Human preferences)
─────────────────        ─────────────────        ─────────────────
Web text, books, code    (instruction, response)  Preference rankings
Next-token prediction    Same objective           Reward model + PPO
→ Raw capability         → Follows instructions   → Aligned & safe
```

| Stage           | Data                                   | Objective                        | Output                      |
| --------------- | -------------------------------------- | -------------------------------- | --------------------------- |
| **Pretraining** | Trillions of tokens (web, books, code) | Next-token prediction            | Base model (predicts)       |
| **SFT**         | 10K-100K (prompt, response) pairs      | Next-token on instruction format | Instruction-tuned           |
| **RLHF**        | Human preference rankings              | Maximize reward model score      | Aligned (helpful, harmless) |

## 5. KEY TECHNIQUES

| Technique              | What It Does                       | When to Use                  | Trade-off                 |
| ---------------------- | ---------------------------------- | ---------------------------- | ------------------------- |
| **RAG**                | Retrieve docs → inject into prompt | Ground answers in your data  | Latency vs accuracy       |
| **Fine-tuning**        | Train model on your data           | Specialized style/domain     | Cost vs customization     |
| **Prompt Engineering** | Craft instructions for behavior    | Quick iteration, no training | Limited vs flexible       |
| **Function Calling**   | LLM outputs structured tool calls  | Connect to APIs, DBs         | Reliability vs capability |
| **Agents**             | LLM in loop: think → act → observe | Multi-step tasks, tools      | Complexity vs power       |
| **Guardrails**         | Filter input/output for safety     | Block harmful content        | Safety vs helpfulness     |
| **Grounding**          | Verify claims against sources      | Reduce hallucination         | Latency vs accuracy       |

**RAG vs fine-tune (when):** RAG when data changes often or you need citations; fine-tune when you need custom style/domain or RAG isn’t accurate enough. LoRA/PEFT for cheaper fine-tune.

## 6. RAG PIPELINE (Most Common Pattern)

```
Query → Embed → Vector Search → Rerank → Top-K chunks → LLM + context → Answer
         │           │             │                          │
    text-embedding  HNSW/IVF   Cross-encoder            Cite sources
```

| RAG Component    | Options                             | Notes                  |
| ---------------- | ----------------------------------- | ---------------------- |
| **Chunking**     | 500 tokens, 200 overlap             | RecursiveTextSplitter  |
| **Embeddings**   | text-embedding-004, CLIP            | 768-3072 dimensions    |
| **Vector Index** | HNSW (fast), IVF (memory efficient) | HNSW for <100M vectors |
| **Retrieval**    | Dense, BM25, Hybrid (RRF)           | Hybrid often best      |
| **Reranking**    | Cross-encoder (ms-marco)            | Top-20 → Top-5         |

**RAG Evaluation (RAGAS)**:
| Metric | Measures | Target |
|--------|----------|--------|
| **Faithfulness** | % claims supported by context | > 0.9 |
| **Answer Relevancy** | Response addresses question | > 0.85 |
| **Context Precision** | % retrieved chunks relevant | > 0.8 |
| **Context Recall** | % relevant chunks retrieved | > 0.8 |

## 7. AGENT PATTERNS

```
SINGLE AGENT              MULTI-AGENT               ORCHESTRATION
─────────────             ───────────               ─────────────
LLM + Tools               Specialized agents        Sequential / Parallel / Debate
ReAct loop                Handoffs between          DAG of steps
→ Most common             → Complex tasks           → Pipelines
```

**Agent = LLM in a loop**: Reason → Act (call tool) → Observe (get result) → Repeat until done

**Reasoning Frameworks**:
| Framework | How It Works | Best For |
|-----------|--------------|----------|
| **CoT (Chain-of-Thought)** | "Think step by step" in prompt | Math, logic |
| **ReAct** | Reason + Act + Observe loop | Tool use, multi-step |

**Tool Types**:
| Type | Where Executed | Example |
|------|----------------|---------|
| **Function Calling** | Client-side | API calls, DB queries |
| **Code Execution** | Agent sandbox | Python, SQL |
| **MCP Tools** | External servers | Standardized tool protocol |

## 8. SERVING & OPTIMIZATION

| Concept                  | What It Is                            | Why It Matters                          |
| ------------------------ | ------------------------------------- | --------------------------------------- |
| **KV Cache**             | Stores computed attention keys/values | Avoid recomputation; grows with context |
| **Continuous Batching**  | Add new requests to running batch     | Better GPU utilization                  |
| **Quantization**         | Reduce precision (FP16→INT8→INT4)     | Faster, smaller, slight quality loss    |
| **Model Routing**        | Small model for easy, large for hard  | Cost optimization                       |
| **Caching**              | Response/semantic/KV cache            | Reduce redundant LLM calls              |
| **Tensor Parallelism**   | Split layers across GPUs              | Serve models > 1 GPU memory             |
| **Pipeline Parallelism** | Split layers sequentially             | Very large models                       |

**Quantization Levels**:
| Level | Memory | Speed | Quality |
|-------|--------|-------|---------|
| FP32 | 1× | 1× | Best |
| FP16/BF16 | 0.5× | 1.5× | ~Same |
| INT8 | 0.25× | 2× | 1-2% loss |
| INT4 | 0.125× | 3× | 3-5% loss |

## 9. IMAGE GENERATION

### Diffusion Models (DALL-E, Stable Diffusion, Imagen)

```
TRAINING:   Image → Add noise (T steps) → Noisy image
            Model learns to predict/remove noise at each step

INFERENCE:  Pure noise → Denoise (T steps) → Generated image
                              ↑
                    Conditioned on text (CLIP/T5 embeddings)
```

| Component                          | Purpose                                        |
| ---------------------------------- | ---------------------------------------------- |
| **Text Encoder**                   | CLIP or T5 → text embeddings                   |
| **U-Net / DiT**                    | Predicts noise to remove                       |
| **DDIM Sampler**                   | Faster sampling (20-50 steps vs 1000)          |
| **CFG (Classifier-Free Guidance)** | Balance text adherence vs diversity (w=7-15)   |
| **VAE**                            | Latent diffusion: compress 512× for efficiency |
| **Super-Resolution**               | Upscale 64×64 → 256×256 → 1024×1024            |

**Latent vs Pixel Diffusion**:
| Approach | Training | Quality | Examples |
|----------|----------|---------|----------|
| **Latent** | 512× faster | Good (VAE may lose details) | Stable Diffusion |
| **Pixel** | Expensive | Highest | Imagen |

### GANs (StyleGAN for faces)

```
Generator (G): Noise → Fake image
Discriminator (D): Real or fake?

Training: G tries to fool D, D tries to catch G → adversarial
```

| GAN Concept        | What It Is                           |
| ------------------ | ------------------------------------ |
| **Mode Collapse**  | G produces limited variety           |
| **Truncation (ψ)** | Trade diversity for quality          |
| **Latent Space**   | 512-dim noise vector → style control |

## 10. VIDEO GENERATION (Sora, Movie Gen)

```
Text → DiT with Temporal Layers → Low-res video → Spatial SR → Temporal SR → Final
              ↑
    3D patches + temporal attention for consistency
```

| Video Component               | Purpose                                  |
| ----------------------------- | ---------------------------------------- |
| **Temporal Attention**        | Consistency across frames                |
| **Temporal Convolution**      | Local motion patterns                    |
| **VAE Compression**           | 8× temporal + 8×8 spatial = 512× smaller |
| **Spatial Super-Resolution**  | 160×90 → 1280×720                        |
| **Temporal Super-Resolution** | 8fps → 24fps                             |

## 11. MULTIMODAL (Vision-Language)

```
IMAGE CAPTIONING:
Image → ViT/CLIP Encoder → Patch embeddings → Cross-attention → Text Decoder → Caption

VQA (Visual Q&A):
Image + Question → Encoder + Cross-attention → Decoder → Answer
```

| Model             | Architecture                    | Use Case        |
| ----------------- | ------------------------------- | --------------- |
| **BLIP-2**        | Frozen encoder + Q-Former + LLM | Captioning, VQA |
| **LLaVA**         | ViT + LLaMA                     | Open-source VQA |
| **Gemini Vision** | Multimodal encoder + decoder    | Everything      |

## 12. ALL METRICS YOU NEED TO KNOW

### Text Generation

| Metric         | Measures                      | Higher/Lower Better |
| -------------- | ----------------------------- | ------------------- |
| **Perplexity** | Model uncertainty             | Lower               |
| **BLEU**       | N-gram precision vs reference | Higher              |
| **ROUGE**      | Recall vs reference           | Higher              |
| **METEOR**     | Semantic similarity           | Higher              |

### RAG Quality

| Metric                | Measures                    | Target |
| --------------------- | --------------------------- | ------ |
| **Faithfulness**      | Claims supported by context | > 0.9  |
| **Answer Relevancy**  | Addresses the question      | > 0.85 |
| **Context Precision** | Retrieved chunks relevant   | > 0.8  |
| **Context Recall**    | Relevant chunks retrieved   | > 0.8  |

### Retrieval

| Metric          | Measures                              |
| --------------- | ------------------------------------- |
| **Precision@K** | % of top-K that are relevant          |
| **Recall@K**    | % of relevant in top-K                |
| **MRR**         | Rank of first relevant result         |
| **NDCG**        | Ranking quality with graded relevance |
| **Hit Rate**    | % queries with ≥1 relevant in top-K   |

### Image Generation

| Metric                   | Measures                               | Target |
| ------------------------ | -------------------------------------- | ------ |
| **FID**                  | Distribution similarity to real images | < 10   |
| **Inception Score (IS)** | Quality × diversity                    | > 10   |
| **CLIPScore**            | Text-image alignment                   | > 0.3  |
| **LPIPS**                | Perceptual similarity                  | Lower  |

### Video Generation

| Metric                   | Measures                              | Target      |
| ------------------------ | ------------------------------------- | ----------- |
| **FVD**                  | Fréchet Video Distance (I3D features) | < 300       |
| **FID (per-frame)**      | Average frame quality                 | < 15        |
| **Temporal Consistency** | Smoothness across frames              | Human > 4/5 |

### Captioning

| Metric     | Measures                           |
| ---------- | ---------------------------------- |
| **CIDEr**  | Consensus with multiple references |
| **BLEU-4** | 4-gram precision                   |
| **METEOR** | Semantic similarity                |

### LLM Benchmarks

| Benchmark      | Tests                             |
| -------------- | --------------------------------- |
| **MMLU**       | Multitask knowledge (57 subjects) |
| **HumanEval**  | Code generation                   |
| **GSM8K**      | Math reasoning                    |
| **TruthfulQA** | Factual accuracy                  |
| **HellaSwag**  | Commonsense reasoning             |

### Safety

| Metric                | Target |
| --------------------- | ------ |
| **Toxicity Rate**     | < 0.1% |
| **PII Leak Rate**     | 0%     |
| **Jailbreak Success** | < 1%   |

### Performance

| Metric                  | What It Measures                 |
| ----------------------- | -------------------------------- |
| **TTFT**                | Time to first token              |
| **Latency P50/P95/P99** | Response time distribution       |
| **Throughput**          | Requests/second or tokens/second |

## 13. COST FORMULA

```
Cost = (Input tokens × input price) + (Output tokens × output price)

Example: 2K input + 500 output @ Gemini Pro ($0.50/1M in, $1.50/1M out)
       = 2000 × 0.0000005 + 500 × 0.0000015 = $0.00175/request
```

**Cost Optimization Levers**:
| Lever | Savings | How |
|-------|---------|-----|
| **Response Cache** | 30-50% | Cache exact query matches |
| **Semantic Cache** | 20-30% | Cache similar queries |
| **Model Routing** | 40-60% | Small model for easy queries |
| **Prompt Optimization** | 10-30% | Shorter prompts |
| **Quantization** | 20-40% | INT8/INT4 models |

## 14. SECURITY & GUARDRAILS

### Threat Model

| Attack                        | Vector                         | Defense                          |
| ----------------------------- | ------------------------------ | -------------------------------- |
| **Direct Prompt Injection**   | User input                     | Input filter, spotlighting       |
| **Indirect Prompt Injection** | Retrieved docs contain attacks | Sanitize, instruction hierarchy  |
| **Jailbreaking**              | Trick model to bypass rules    | RLHF, guardrails                 |
| **Data Leakage**              | Model reveals training data    | Differential privacy, guardrails |
| **Tool Abuse**                | Agent misuses APIs             | Least privilege, validation      |

### Defense Layers

```
HTTP Layer → Auth → INPUT GUARDRAILS → LLM → OUTPUT GUARDRAILS → Response
                         ↑                          ↑
                   Injection detection         PII filter
                   Blocklist                   Toxicity check
                   Spotlighting                Hallucination check
```

**Spotlighting**: Mark external content → `<<DATA>>user docs here<</DATA>>` → model treats as data, not instructions. **Indirect injection**: Attacker puts instructions inside retrieved docs; defend with sanitization and instruction hierarchy (system prompt overrides doc content).

### Platform Services

| Purpose           | Google Cloud   | AWS                |
| ----------------- | -------------- | ------------------ |
| **Guardrails**    | Model Armor    | Bedrock Guardrails |
| **WAF**           | Cloud Armor    | AWS WAF            |
| **PII Detection** | Cloud DLP      | Amazon Macie       |
| **Secrets**       | Secret Manager | Secrets Manager    |

## 15. SCALABILITY PATTERNS

### GPU Quick Reference

| GPU         | Memory   | FP16 TFLOPS | Cost      | Best For                        |
| ----------- | -------- | ----------- | --------- | ------------------------------- |
| **V100**    | 16/32 GB | 125         | ~$2/hr    | Legacy, small models            |
| **A100**    | 40/80 GB | 312         | ~$4/hr    | Production standard             |
| **H100**    | 80 GB    | 990         | ~$8/hr    | Large training, high throughput |
| **L4**      | 24 GB    | 121         | ~$0.80/hr | Cost-effective inference        |
| **TPU v5e** | 16 GB    | —           | ~$1.20/hr | Google Cloud                    |

**Model → GPU sizing**: 7B = 14GB (1× L4) | 70B = 140GB (2× H100) | 405B = 810GB (8× H100)

### Inference Scaling

| Pattern                  | What It Does             | When              |
| ------------------------ | ------------------------ | ----------------- |
| **Horizontal**           | More replicas            | Stateless serving |
| **Tensor Parallelism**   | Split layers across GPUs | Model > 1 GPU     |
| **Pipeline Parallelism** | Sequential layer split   | Very large models |
| **Continuous Batching**  | Dynamic batch management | High throughput   |

### Training Scaling

| Pattern                      | What It Does                              |
| ---------------------------- | ----------------------------------------- |
| **Data Parallelism**         | Same model on each GPU, different data    |
| **Model/Tensor Parallelism** | Split model across GPUs                   |
| **Pipeline Parallelism**     | Split layers sequentially                 |
| **ZeRO/FSDP**                | Shard optimizer states, gradients, params |
| **Gradient Checkpointing**   | Trade compute for memory                  |
| **Mixed Precision (AMP)**    | FP16 forward, FP32 gradients              |

**ZeRO Levels**:
| Level | Shards | Memory Reduction |
|-------|--------|------------------|
| ZeRO-1 | Optimizer states | ~4× |
| ZeRO-2 | + Gradients | ~8× |
| ZeRO-3 | + Parameters | ~N× (N = GPUs) |

## 16. INTERVIEW FRAMEWORK (45 min)

| Phase               | Time      | What to Cover                                |
| ------------------- | --------- | -------------------------------------------- |
| **1. Requirements** | 5-10 min  | Token budget, latency, quality, cost, safety |
| **2. Architecture** | 10-15 min | Draw: API → Orchestration → LLM → Response   |
| **3. Deep Dive**    | 15-20 min | RAG, model choice, eval, security            |
| **4. Trade-offs**   | 5-10 min  | Quality vs cost, latency vs throughput       |

**Always do**: Back-of-envelope (tokens × price, latency breakdown)

## 17. QUICK DECISION GUIDE

| Need                 | Solution                                     |
| -------------------- | -------------------------------------------- |
| Ground in docs       | **RAG** (chunk + vector DB + rerank)         |
| Real-time tools      | **Function calling** / **Agents**            |
| Custom style         | **Fine-tuning** (SFT)                        |
| Ultra-low latency    | **Small model** + **edge** + **cache**       |
| Reduce hallucination | **RAG** + **grounding** + **citations**      |
| Long documents       | **Chunking** or **long-context model**       |
| Multi-step reasoning | **Agents** (ReAct)                           |
| Cost at scale        | **Routing** + **caching** + **quantization** |
| Generate images      | **Diffusion** (Stable Diffusion, Imagen)     |
| Generate video       | **DiT + temporal layers** (Sora)             |
| Generate faces       | **StyleGAN**                                 |

## 18. PLATFORM COMPARISON

|                | Google Cloud               | AWS                       |
| -------------- | -------------------------- | ------------------------- |
| **Models**     | Gemini (Vertex AI)         | Claude, Titan (Bedrock)   |
| **RAG**        | Vertex RAG Engine          | Bedrock Knowledge Bases   |
| **Agents**     | ADK, Conversational Agents | AgentCore, Bedrock Agents |
| **Vector DB**  | Vertex AI Vector Search    | OpenSearch, Pinecone      |
| **Guardrails** | Model Armor                | Bedrock Guardrails        |
| **Image Gen**  | Imagen 3                   | Titan Image, SD           |
| **Video Gen**  | Veo                        | -                         |

## 18b. TABLE OF TOOLS (by category)

Quick lookup: which tool to use for what. Aligned with full guide (E.1–E.10, D.2, F.1).

| Category                  | Tool / stack                                   | Purpose                                            | When to use                                 |
| ------------------------- | ---------------------------------------------- | -------------------------------------------------- | ------------------------------------------- |
| **LLM serving**           | **vLLM**                                       | OSS inference; PagedAttention, continuous batching | Self-hosted, high throughput, open models   |
|                           | **TGI** (Text Generation Inference)            | Hugging Face serving; similar to vLLM              | HF ecosystem, Inference Endpoints           |
|                           | **TensorRT-LLM**                               | NVIDIA-optimized inference                         | Max perf on NVIDIA GPUs                     |
|                           | **Vertex AI / Bedrock**                        | Managed model hosting                              | No infra; enterprise; Gemini/Claude         |
| **Orchestration**         | **LangChain**                                  | Chains, agents, tools, RAG helpers                 | Rapid prototyping; broad ecosystem          |
|                           | **LlamaIndex**                                 | RAG-focused; data loaders, indices                 | Document-heavy RAG, diverse data sources    |
|                           | **ADK** (Agent Development Kit)                | Google agent framework; workflows, multi-agent     | GCP; production agents; Vertex Agent Engine |
| **RAG / vector**          | **Vertex AI RAG Engine**                       | Managed RAG (ingest, index, query)                 | GCP; minimal RAG plumbing                   |
|                           | **Bedrock Knowledge Bases**                    | Managed RAG on AWS                                 | AWS; same idea as Vertex RAG                |
|                           | **FAISS**                                      | OSS vector search (HNSW, IVF)                      | Self-hosted; library inside your app        |
|                           | **Pinecone / Weaviate / Qdrant**               | Managed or OSS vector DBs                          | Production vector search; scale             |
|                           | **Vertex AI Vector Search**                    | Managed vector index (GCP)                         | GCP; managed HNSW                           |
|                           | **pgvector**                                   | Postgres extension for vectors                     | Already on Postgres; smaller scale          |
| **Embeddings**            | **text-embedding-004** (Vertex)                | Google embeddings                                  | GCP RAG                                     |
|                           | **OpenAI text-embedding-3**                    | OpenAI embeddings                                  | When using OpenAI models                    |
|                           | **e5 / BGE** (OSS)                             | Open-source embed models                           | Self-hosted; no API cost                    |
| **Document parsing**      | **Document AI** (GCP)                          | PDF/docs → text, tables, structure                 | GCP; high-quality parsing                   |
|                           | **Textract** (AWS)                             | Same idea on AWS                                   | AWS                                         |
|                           | **PyMuPDF / pdfplumber**                       | Rule-based PDF extraction                          | Simple, consistent layouts                  |
| **Chunking**              | **RecursiveCharacterTextSplitter** (LangChain) | Split by separators, token-aware                   | Default in many RAG tutorials               |
| **Guardrails**            | **Model Armor** (GCP)                          | Input/output safety; injection, jailbreak          | Vertex AI apps                              |
|                           | **Bedrock Guardrails**                         | Same on AWS                                        | Bedrock apps                                |
| **Evaluation**            | **RAGAS**                                      | RAG metrics (faithfulness, relevancy, context)     | RAG quality without gold answers            |
| **Dev / experimentation** | **Google AI Studio**                           | Play with Gemini; quick prototypes                 | Learning; no production                     |
|                           | **Vertex AI Studio**                           | Enterprise; fine-tune, eval, deploy                | Move from AI Studio to production           |

**Stack snapshots (full guide):** Real-world examples in F.1 combine these (e.g. Document AI + LangChain chunking + Vertex Vector Search + Gemini + RAGAS; or vLLM on GKE + LangChain + Pinecone).

## 19. MODEL QUICK REFERENCE

| Model                | Type             | Params                 | Notes              |
| -------------------- | ---------------- | ---------------------- | ------------------ |
| **GPT-4o**           | Decoder-only     | ~1.7T (rumored MoE)    | Multimodal, SOTA   |
| **Gemini 1.5 Pro**   | MoE              | Undisclosed            | 1M context         |
| **Claude 3.5**       | Decoder-only     | Undisclosed            | Strong reasoning   |
| **LLaMA 3**          | Decoder-only     | 8B-405B                | Open-source        |
| **Mixtral 8x22B**    | MoE              | 141B total, 39B active | Open-source        |
| **Stable Diffusion** | Latent Diffusion | ~1B                    | Open-source images |
| **Sora**             | DiT + temporal   | Undisclosed            | Video generation   |

## 20. RED FLAGS IN INTERVIEWS

| Red Flag              | Better Answer                 |
| --------------------- | ----------------------------- |
| "No RAG needed"       | How prevent hallucination?    |
| "One model for all"   | What about cost optimization? |
| "Skip guardrails"     | Prompt injection? PII?        |
| "No eval plan"        | How know it's working?        |
| "Unbounded context"   | Cost and latency?             |
| "Train from scratch"  | Why not fine-tune?            |
| "Real-time video gen" | Minutes is realistic          |

---

## 21. GLOSSARY (ADAPTED)

Short definitions for terms used in the full guide. **Why** = why it matters for system design.

### Fundamentals & core

| Term                        | Definition                                                                                  | Why                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **AI / ML / GenAI**         | AI = broad field. ML = learn from data. GenAI = create content (text, image, video).        | GenAI = LLMs + diffusion + agents; design for generation, not just classification. |
| **LLM**                     | Large language model; predicts next token; billions of params (GPT, Gemini, Claude, LLaMA). | Core of most GenAI systems; cost and latency are token-driven.                     |
| **Token**                   | Smallest unit of text (~4 chars, ~0.75 words). Models charge and limit by tokens.           | Cost, context window, and latency all scale with tokens.                           |
| **Context window**          | Max tokens in one request (prompt + response). 4K–2M depending on model.                    | Limits how much you can retrieve and inject; larger = more cost/latency.           |
| **Inference**               | Running a trained model (prompt → response).                                                | Most cost is inference, not training.                                              |
| **Attention / Transformer** | Mechanism to focus on relevant input; architecture behind LLMs.                             | Enables long context and reasoning; KV cache is the main memory cost.              |
| **Encoder / Decoder**       | Encoder = understand (BERT). Decoder = generate (GPT). Encoder-decoder = transform (T5).    | Decoder-only for chat/code; encoder-decoder for translation/summarization.         |

### Classical ML, NN & RL acronyms

| Term              | Definition                                                                               | Why                                                                         |
| ----------------- | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **MSE**           | Mean Squared Error; loss for regression (average of squared residuals).                  | Default loss for linear regression; differentiable, penalizes large errors. |
| **Log Loss**      | Binary Cross-Entropy; loss for binary classification (log probability of correct class). | Standard for logistic regression and binary classifiers.                    |
| **SVM**           | Support Vector Machine; finds optimal separating hyperplane (or fit in regression).      | Strong baseline for tabular data; hinge/ε-insensitive loss.                 |
| **XGBoost**       | Extreme Gradient Boosting; gradient boosting with regularization.                        | Dominant for tabular ML; sequential trees correct prior errors.             |
| **FFNN**          | Feedforward Neural Network; layers with no loops, static input → output.                 | Basic deep net; used in MLPs and as sub-blocks in Transformers.             |
| **CNN**           | Convolutional Neural Network; uses convolution filters for grid-like data (e.g. images). | Standard for vision; spatial feature extraction.                            |
| **RNN**           | Recurrent Neural Network; recurrence over time; maintains hidden state.                  | Classic sequence model; superseded by Transformers for long context.        |
| **LSTM**          | Long Short-Term Memory; RNN with gates to capture long-range dependencies.               | Reduces vanishing gradients in RNNs; used in older sequence models.         |
| **ReLU**          | Rectified Linear Unit; activation max(0, x).                                             | Default activation in deep nets; fast, avoids some vanishing gradients.     |
| **Sigmoid**       | S-shaped activation, output in (0, 1).                                                   | Binary classification output; probability.                                  |
| **Tanh**          | Hyperbolic tangent; like sigmoid but output in (-1, 1), centered at 0.                   | Used in gates (e.g. LSTM); zero-centered.                                   |
| **Softmax**       | Converts logits to a probability distribution over classes.                              | Final layer for multi-class classification.                                 |
| **BatchNorm**     | Batch Normalization; normalizes activations over the batch dimension.                    | Stabilizes training; standard in CNNs.                                      |
| **LayerNorm**     | Layer Normalization; normalizes over feature dimension (per token).                      | Standard in Transformers; stable for variable batch/length.                 |
| **NLP**           | Natural Language Processing; models and systems for text/speech.                         | Transformers are state-of-the-art in NLP.                                   |
| **SGD**           | Stochastic Gradient Descent; parameter updates from (mini-)batch gradient.               | Base optimizer; learning rate γ.                                            |
| **Adam**          | Adaptive Moment Estimation; combines momentum and per-parameter adaptive learning rates. | Default optimizer for deep learning; γ, β₁, β₂, ε.                          |
| **RMSProp**       | Root Mean Square Propagation; adaptive learning rate from squared gradient.              | Component of Adam; good for non-stationary objectives.                      |
| **REINFORCE**     | Monte Carlo policy gradient; maximize return using full episode gradient.                | Basic RL policy gradient; high variance.                                    |
| **TRPO**          | Trust Region Policy Optimization; policy updates constrained by KL to old policy.        | Stable RL; used in robotics and alignment.                                  |
| **PPO**           | Proximal Policy Optimization; clipped surrogate objective approximating TRPO.            | Default for RLHF and many RL tasks; stable and practical.                   |
| **GRPO**          | Group Relative Policy Optimization; relative advantages within groups (e.g. for LLMs).   | Used in LLM alignment; group-wise baselines.                                |
| **KL-divergence** | Kullback–Leibler divergence; measure of difference between two distributions.            | Used in TRPO/PPO to limit policy change; in RLHF for regularization.        |

### Tokens & generation

| Term               | Definition                                                                   | Why                                                             |
| ------------------ | ---------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Tokenization**   | Text → tokens (BPE, SentencePiece, WordPiece).                               | Same text can be different token count per model; affects cost. |
| **Autoregressive** | Generate one token at a time; each depends on previous.                      | Latency scales with output length.                              |
| **Temperature**    | Randomness in token choice. 0 = deterministic; 1 = sample; >1 = more random. | Low for factual, high for creative.                             |
| **Top-p / Top-k**  | Restrict candidate tokens by cumulative probability (top-p) or top k.        | Control diversity and weird outputs.                            |
| **Sampling**       | How next token is chosen: greedy vs random (with temp/top-p/top-k).          | Greedy = reproducible; sampling = varied.                       |

### Memory & caching

| Term               | Definition                                                         | Why                                                                  |
| ------------------ | ------------------------------------------------------------------ | -------------------------------------------------------------------- |
| **KV cache**       | Stores attention keys/values for previous tokens; avoid recompute. | Without it, generation is O(N²); with it, memory grows with context. |
| **PagedAttention** | KV cache in non-contiguous pages (vLLM).                           | Less fragmentation, 2–4× more throughput.                            |
| **Semantic cache** | Cache responses by embedding similarity.                           | Cuts cost 20–50% for similar queries; risk of stale/wrong answers.   |
| **Prompt cache**   | Cache KV for common prefixes (e.g. system prompt).                 | Saves compute and TTFT.                                              |

### RAG

| Term                           | Definition                                                                        | Why                                                    |
| ------------------------------ | --------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **RAG**                        | Retrieve docs → inject into prompt → generate.                                    | Ground answers in your data without fine-tuning.       |
| **Embedding**                  | Dense vector for semantic meaning (768–1536 dims).                                | Similar text → close vectors; enables semantic search. |
| **Vector DB**                  | Store embeddings; nearest-neighbor search (Pinecone, FAISS, Vertex).              | Needed for retrieval at scale; use HNSW or IVF.        |
| **Chunking**                   | Split docs into 200–1000 token chunks; overlap 50–100.                            | Too small = no context; too large = noise.             |
| **HNSW / IVF**                 | HNSW = graph ANN, best recall/speed. IVF = cluster, search few clusters (nprobe). | HNSW default; IVF when memory-constrained.             |
| **nprobe**                     | In IVF: how many clusters to search per query.                                    | Higher = better recall, slower. Typical 10–50.         |
| **Reranking**                  | After retrieval: cross-encoder re-scores top-K (e.g. 20→5).                       | More accurate than embedding-only; slower.             |
| **Bi-encoder / Cross-encoder** | Bi: query and doc embedded separately, compare. Cross: query+doc in one pass.     | Bi for retrieval; cross for reranking.                 |
| **Hybrid search**              | Vector + keyword (BM25); merge with RRF.                                          | Better than dense or keyword alone.                    |
| **BM25**                       | Keyword ranking by term frequency.                                                | Good for exact matches; combine with vector.           |
| **Grounding**                  | Tie model answers to retrieved sources; cite.                                     | Reduces hallucination; verifiable.                     |
| **RAGAS**                      | RAG eval: Faithfulness, Answer Relevancy, Context Precision/Recall.               | Standard RAG quality metrics.                          |

### Fine-tuning

| Term                  | Definition                                                         | Why                                                              |
| --------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------- |
| **Fine-tuning / SFT** | Train on (input, output) pairs; model learns your task/style.      | When RAG isn’t enough; need 100s–1000s of examples.              |
| **LoRA / QLoRA**      | Train small adapters (rank 8–64); base frozen. QLoRA = 4-bit base. | 10–100× cheaper than full fine-tune; swap adapters at inference. |
| **PEFT**              | Parameter-efficient fine-tuning (LoRA, adapters, prefix tuning).   | Same idea: train a small part.                                   |
| **RLHF**              | Reward model from human preferences; RL to maximize reward.        | How chat models get “helpful, harmless.”                         |
| **DPO**               | Direct preference optimization; no separate reward model.          | Simpler than RLHF for alignment.                                 |

### Agents & tools

| Term                 | Definition                                                            | Why                                    |
| -------------------- | --------------------------------------------------------------------- | -------------------------------------- |
| **Agent**            | LLM in loop: perceive → decide → act (tool) → observe → repeat.       | Multi-step tasks, APIs, DBs, code.     |
| **ReAct**            | Reason + Act + Observe; interleave reasoning and tool calls.          | Default pattern for tool-using agents. |
| **CoT**              | Chain-of-thought: “think step by step” in the prompt.                 | Improves math/reasoning.               |
| **Function calling** | LLM outputs structured tool/API call; system executes and returns.    | How agents talk to external systems.   |
| **ADK**              | Agent Development Kit (Google); workflows, tools, multi-agent.        | Build agents on GCP.                   |
| **MCP**              | Model Context Protocol; standard for tools and context.               | Interop between agents and tools.      |
| **Orchestration**    | Layer that runs LLM, retrieval, tools, control flow (LangChain, ADK). | Glue between your app and models.      |

### Serving & scaling

| Term                              | Definition                                             | Why                                                     |
| --------------------------------- | ------------------------------------------------------ | ------------------------------------------------------- |
| **TTFT**                          | Time to first token.                                   | Drives perceived latency; optimize for interactive use. |
| **Continuous batching**           | New requests join running batch as slots free.         | Much better GPU use than static batching.               |
| **Quantization**                  | FP16→INT8→INT4; smaller, faster, small quality loss.   | Deploy large models on limited hardware.                |
| **Tensor / Pipeline parallelism** | Split model across GPUs (per layer vs per stage).      | When model doesn’t fit on one GPU.                      |
| **vLLM / TGI**                    | OSS LLM serving; PagedAttention, continuous batching.  | Production inference.                                   |
| **ZeRO / FSDP**                   | Shard optimizer, grads, params across GPUs (training). | Train 70B+ models.                                      |
| **Speculative decoding**          | Draft model proposes tokens; big model verifies.       | 2–3× speedup when draft is good.                        |

### Image & video

| Term                       | Definition                                                              | Why                                                  |
| -------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------- |
| **Diffusion**              | Learn to denoise; generate by iteratively denoising from noise.         | Dominant for text-to-image (SD, DALL·E, Imagen).     |
| **Latent diffusion (LDM)** | Diffuse in compressed latent space; VAE encode/decode.                  | 64× cheaper; Stable Diffusion.                       |
| **U-Net / DiT**            | U-Net = hourglass denoiser. DiT = Transformer denoiser (Sora).          | DiT scales better for video.                         |
| **VAE**                    | Encoder/decoder for images; compress to latent.                         | Used inside latent diffusion.                        |
| **CFG**                    | Classifier-free guidance; amplify effect of text conditioning (w=7–15). | Better prompt adherence.                             |
| **DDIM**                   | Few-step sampling (20–50 vs 1000).                                      | Production image gen.                                |
| **FID / CLIPScore**        | FID = distribution match to real images. CLIPScore = text-image match.  | FID = quality; CLIPScore = alignment.                |
| **CIDEr**                  | Caption metric: consensus over multiple references (TF-IDF n-grams).    | Best correlation with human judgment for captioning. |
| **FVD / I3D**              | FVD = video quality (frames + motion); I3D = network used inside.       | Main metric for video gen.                           |

### Evaluation & safety

| Term                         | Definition                                                            | Why                                   |
| ---------------------------- | --------------------------------------------------------------------- | ------------------------------------- |
| **Hallucination**            | Model states false things confidently.                                | RAG, grounding, guardrails reduce it. |
| **Faithfulness / Relevancy** | Faithfulness = claims from context. Relevancy = answers the question. | Both needed for RAG eval.             |
| **LLM-as-judge**             | Use an LLM to score another LLM’s output.                             | Scalable eval; use strong model.      |
| **Guardrails**               | Input/output filters: toxicity, PII, jailbreak, policy.               | Required in production.               |
| **Spotlighting**             | Mark external content so model treats it as data, not instructions.   | Fights prompt injection.              |
| **Model Armor**              | Google Cloud guardrail service.                                       | Managed guardrails on GCP.            |

### Infrastructure & cost

| Term                    | Definition                                                 | Why                                   |
| ----------------------- | ---------------------------------------------------------- | ------------------------------------- |
| **Vertex AI / Bedrock** | GCP / AWS managed GenAI (models, RAG, agents, guardrails). | Deploy without building everything.   |
| **Per-token pricing**   | Charge by input + output tokens; output often 3–4× input.  | Design for token budgets and caching. |
| **Token budget**        | Max tokens per request or per user/session.                | Prevents runaway cost.                |
| **Model routing**       | Easy query → small/cheap model; hard → large.              | 40–60% cost savings.                  |

---

**Core Formula**: GenAI design = Traditional system design + Token economics + Quality evaluation + Safety

**Remember**: Always discuss metrics, always do back-of-envelope, always mention trade-offs
