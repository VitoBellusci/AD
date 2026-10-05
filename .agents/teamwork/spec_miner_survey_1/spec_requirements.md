# Academic Specification & Requirements Catalog: Tiny Text-Conditioned Avatar Diffusion from Scratch

**Course:** Deep Learning (M.D. in Computer Engineering) — Politecnico di Bari  
**Instructor:** Vito Walter Anelli, Ph.D.  
**Test Code:** 2026_VI (Released September 21st, 2026 | Deadline: October 22nd, 2026)  
**Authoritative Source:** `Deep_Learning_2026_VI 1.pdf` (supplemented by `ORIGINAL_REQUEST.md`)  
**Auditor:** `spec_miner_survey_1` (Specification Miner)  
**Audit Scope:** Read-Only Formal Requirements Mining & Verification Criteria Catalog

---

## Executive Summary & Core Research Question

The assignment challenges students to build and train a **pixel-space text-conditioned Denoising Diffusion Probabilistic Model (DDPM)** from scratch on cartoon avatars. The foundational requirement is that **no pretrained components** (e.g. CLIP, pretrained diffusion backbones, pretrained text models, or ready-made Diffusers pipelines) may be used. The entire architecture—including image tokenizer/vocabulary, text encoder, U-Net denoiser, conditioning mechanism, noise scheduling, and reverse sampling loop—must be implemented, trained, and evaluated in pure PyTorch.

### Central Research Question
> *"Can a very small diffusion model, whose denoiser and text encoder are both trained from scratch, generalize to combinations of avatar attributes that were not observed together during training?"*

This question establishes that standard random train/test splitting is strictly insufficient: the project demands a **compositional generalization benchmark** with systematically held-out attribute combinations.

---

## 1. Mandatory From-Scratch Constraints

The assignment sets rigid boundaries separating student-implemented deep learning components from external or pretrained tooling.

### 1.1 Prohibited Components & Models (Strict Zero-Tolerance List)

| Prohibited Component / Category | Specific Examples Explicitly Banned | Architectural Rationale & Enforcement Rule |
|---|---|---|
| **Pretrained Diffusion Checkpoints** | Stable Diffusion (v1.4, v1.5, v2.1), Tiny-SD, SDXL, Flux, DeepFloyd IF, Imagen, Midjourney models | Model weights must be randomly initialized and trained entirely on the provided avatar dataset. No fine-tuning or transfer learning from existing diffusion models. |
| **Pretrained Text Encoders** | CLIP (`openai/clip-vit-*`), T5 (`t5-small`, `t5-base`), BERT, RoBERTa, DistilBERT, GPT-2 | Conditioning representations cannot rely on pre-existing semantic spaces. The text representation must be learned ground-up. |
| **Pretrained Autoencoders / Latent Diffusion** | Stable Diffusion AutoencoderKL, VQ-VAE, VQ-GAN, pretrained latent pipelines | Model must operate strictly in **pixel space** ($32\times 32$ or $64\times 64$ RGB images), eliminating latent-space compression shortcuts. |
| **Ready-made Diffusers Pipelines as Black Boxes** | Hugging Face `diffusers.DDPMPipeline`, `diffusers.UNet2DModel`, `diffusers.DDPMScheduler`, `diffusers.DiffusionPipeline` | Black-box training loops or prepackaged forward/reverse scheduler classes are banned. All core diffusion math and forward passes must be explicitly written. |
| **Pretrained Embeddings** | GloVe, Word2Vec, FastText, pretrained ImageNet feature extractors as model representations | Input text embeddings and visual features must be learned from scratch. |

### 1.2 Mandatory From-Scratch Components (Must Be Student-Implemented)

1. **Deterministic Caption Generator:** Generates multi-attribute descriptive captions deterministically from dataset metadata.
2. **Vocabulary & Tokenizer:** Tokenizer vocabulary constructed strictly from the **training split only** (no validation/test text exposure).
3. **Transformer Text Encoder:** A 2–4 layer Transformer encoder (with learned embeddings, sinusoidal/learnable positional encodings, multi-head self-attention, and feed-forward networks) trained end-to-end.
4. **Compact Denoiser U-Net:** Pixel-space U-Net featuring downsampling blocks, bottleneck, upsampling blocks, skip connections, Group Normalization, and non-linearities.
5. **Conditioning Mechanism:** Cross-attention layers (or alternative mechanisms such as AdaGN/AdaLN/concatenation) mapping textual context into visual feature maps.
6. **Noise Scheduler & Mathematical Formulations:** Explicit calculation of variance schedule $\beta_t$, retention factor $\alpha_t = 1 - \beta_t$, cumulative product $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$, and forward closed-form diffusion $q(x_t|x_0)$.
7. **Diffusion Loss Objective:** Timestep-sampled mean squared error (MSE) predicting the injected noise $\epsilon$ ($\epsilon$-prediction) or velocity $v$ ($v$-prediction).
8. **Reverse Sampling Loop:** Generation loop $p_\theta(x_{t-1}|x_t)$ starting from pure Gaussian noise $x_T \sim \mathcal{N}(0, \mathbf{I})$, incorporating stochastic Langevin noise for $t > 0$, deterministic mean extraction at $t=0$, and Classifier-Free Guidance (CFG).

### 1.3 Permitted Libraries & Primitives
- Standard PyTorch (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data.DataLoader`).
- Primitive neural network modules (`nn.Conv2d`, `nn.Linear`, `nn.GroupNorm`, `nn.Embedding`, `nn.Dropout`, `nn.SiLU`, `nn.ReLU`).
- Standard optimization utilities (AdamW, Cosine Annealing learning rate schedulers, gradient clipping).
- Tensor manipulation and scientific packages (`numpy`, `math`, `csv`, `json`, `PIL`, `torchvision.transforms`).
- Metric computation tools (`torchmetrics` for FID/KID, `torchmetrics.image.lpip` or torchvision for perceptual evaluation).

---

## 2. Architectural Specifications & Parameter Budget

### 2.1 Parameter Envelope ("Tiny" / Compact Model Budget)
The assignment is calibrated for a single **NVIDIA T4 GPU (16 GB VRAM)** budget (e.g., standard Google Colab tier).

| Architectural Hyperparameter | PDF Recommendation Envelope | Target / Nominal Configuration |
|---|---|---|
| **Base Channels ($C_{\text{base}}$)** | 64 – 128 | 64 channels |
| **Spatial Resolution Stages** | 2 or 3 spatial resolutions | 3 levels (e.g., $64\times 64 \to 32\times 32 \to 16\times 16$) |
| **Channel Multipliers** | Typically $[1, 2, 4]$ | Level 0: 64, Level 1: 128, Level 2 / Bottleneck: 256 |
| **Text Encoder Layers** | 2 – 4 Transformer encoder layers | 3 layers |
| **Text Embedding Dimension ($d_{\text{model}}$)** | 64 – 128 | 128 |
| **Attention Heads ($h$)** | Divisible into $d_{\text{model}}$ (e.g., 4 or 8) | 4 heads ($d_k = 32$) or 8 heads ($d_k = 16$) |
| **Feed-Forward Dimension ($d_{ff}$)** | $2\times$ to $4\times d_{\text{model}}$ | 256 ($2\times$) |
| **Expected Parameter Count** | "Tiny" footprint | Total: $\approx 10\text{M} - 25\text{M}$ parameters |

### 2.2 U-Net Denoiser Architecture
- **Input Channels:** 3 (RGB noisy image $x_t \in \mathbb{R}^{B \times 3 \times H \times W}$, where $H, W \in \{32, 64\}$).
- **Output Channels:** 3 (predicted noise $\epsilon_\theta(x_t, t, c) \in \mathbb{R}^{B \times 3 \times H \times W}$).
- **Time Conditioning:**
  - Sinusoidal position embeddings: $\text{dim} = C_{\text{base}}$.
  - Time MLP: $\text{Linear}(C_{\text{base}}, 4 C_{\text{base}}) \to \text{SiLU} \to \text{Linear}(4 C_{\text{base}}, 4 C_{\text{base}})$.
  - Time embedding injected into every residual block via spatial broadcasting and addition (or scale-and-shift adaptive normalization).
- **Residual Blocks (`DoubleConv`):**
  - $\text{Conv2d}(3\times 3) \to \text{GroupNorm}(8) \to \text{SiLU} \to [+\text{time\_emb}] \to \text{Conv2d}(3\times 3) \to \text{GroupNorm}(8) \to \text{SiLU} + \text{ResidualConv}(1\times 1)$.
- **Downsampling Blocks:**
  - Spatial halving via `nn.MaxPool2d(2)` or strided convolutions.
- **Upsampling Blocks:**
  - Spatial doubling via `nn.Upsample(scale_factor=2, mode='bilinear')` or `nn.ConvTranspose2d`.
  - Skip connections concatenated channel-wise from corresponding encoder stages before convolution.
- **Spatial Attention Layers:**
  - Spatial Self-Attention: Applied at lower spatial resolutions ($16\times 16$, bottleneck) to capture global image coherence.
  - Spatial Cross-Attention: Injected across down/up stages or concentrated at the bottleneck to query the textual tokens $c \in \mathbb{R}^{B \times L \times d_{\text{model}}}$.

### 2.3 Mathematical DDPM Formulation

#### Variance Schedule ($\beta_t$)
The assignment encourages the **Cosine Noise Schedule** (Nichol & Dhariwal, 2021) over the linear schedule to avoid excessive corruption of low-resolution images in early steps:
$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad \text{where } f(t) = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$$
with offset $s = 0.008$ and $T = 1000$ timesteps.
$$\beta_t = \text{clip}\left(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, \, \beta_{\max} = 0.999\right), \quad \alpha_t = 1 - \beta_t$$

#### Forward Diffusion Process ($q(x_t | x_0)$)
Closed-form sampling at arbitrary timestep $t \sim \mathcal{U}(\{1, \dots, T\})$:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

#### Optimization Objective
Simplified MSE noise prediction loss:
$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon, c} \left[ \left\| \epsilon - \epsilon_\theta(x_t, t, c) \right\|^2 \right]$$
*(Alternative velocity target $v_t = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1 - \bar{\alpha}_t}x_0$ is explicitly noted as an acceptable target in PDF Section 7).*

#### Reverse Sampling ($p_\theta(x_{t-1} | x_t)$)
$$\mu_\theta(x_t, t, c) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \, \hat{\epsilon}_\theta(x_t, t, c) \right)$$
$$x_{t-1} = \mu_\theta(x_t, t, c) + \sigma_t z, \quad \text{where } z \sim \begin{cases} \mathcal{N}(0, \mathbf{I}) & \text{for } t > 1 \\ 0 & \text{for } t = 1 \end{cases}$$
with variance $\sigma_t^2 = \beta_t$ (or posterior variance $\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$).

#### Classifier-Free Guidance (CFG)
During training, conditioning $c$ is randomly replaced by unconditional token embedding $\emptyset$ with probability $p_{\text{uncond}} \in [0.1, 0.2]$.
During reverse sampling:
$$\hat{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + w \cdot \left(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)\right)$$
where guidance scale $w \ge 1.0$ (typically $w \in [2.0, 5.0]$).

---

## 3. Data & Dataset Requirements

### 3.1 Dataset Specification: Google Cartoon Set
- **Dataset Variants:** 10k images (acceptable fallback subset) or 100k images (recommended for extended configuration).
- **Target Spatial Resolutions:** $32\times 32$ (for initial smoke test & fast iteration) or $64\times 64$ (final demonstration resolution).
- **Metadata Structure:** Categorical attributes describing avatar components:
  - `face_shape` (7 variants), `face_color` (11 variants)
  - `hair` (111 variants), `hair_color` (10 variants)
  - `eye_color` (5 variants), `eye_angle` (3), `eye_lid` (2), `eye_lashes` (2), `eye_slant` (3)
  - `eyebrow_shape` (14), `eyebrow_thickness` (4), `eyebrow_weight` (2), `eyebrow_width` (3)
  - `glasses` (12 variants), `glasses_color` (7 variants)
  - `facial_hair` (15 variants), `chin_length` (3)
- **Image Normalization:** Pixels normalized consistently to range $[-1.0, 1.0]$ with $\mu = [0.5, 0.5, 0.5]$ and $\sigma = [0.5, 0.5, 0.5]$.

### 3.2 Caption Synthesis Requirements
- Captions must be **deterministic, multi-attribute text descriptions** constructed directly from image metadata attributes.
- Example from assignment specification:
  - *“a boy with blue colors, round eyes, and exaggerated proportions”*
  - or structured format: *“avatar with face 1, hair 98, eyes 4, glasses 11, and facial hair 3”*.
- Vocabulary must be derived strictly from the training split.

### 3.3 Strict Tokenizer & Vocabulary Isolation (Zero Data Leakage)
- **Prohibition:** Do **NOT** use validation or test captions, normalization statistics, or attribute frequencies when constructing the tokenizer vocabulary.
- Any word/token present in validation or test captions that did not occur in the training split must map to the out-of-vocabulary `<UNK>` token.
- Special tokens required: `<PAD>` (padding), `<UNK>` (unknown/unseen), `<SOS>` (start-of-sequence, optional), `<EOS>` (end-of-sequence, optional).

### 3.4 Compositional Generalization Split (Mandatory Requirement)
The evaluation of the central research question hinges entirely on the split design. **A purely random split violates the assignment specification.**

The dataset must be partitioned into **four distinct subsets**:
1. **Training Set ($\mathcal{D}_{\text{train}}$):** Contains images where all individual attributes are present, but specific blocked attribute combinations are completely absent.
2. **Validation Set ($\mathcal{D}_{\text{val}}$):** In-distribution hold-out sampled from the permissible combinations to tune hyperparameters and monitor training loss.
3. **Ordinary Test Set ($\mathcal{D}_{\text{test\_ind}}$):** In-distribution test examples with seen attribute combinations, used to establish ordinary generalization capability.
4. **Compositional Out-of-Distribution Test Set ($\mathcal{D}_{\text{test\_ood}}$):** Images exhibiting the specifically held-out attribute combinations (e.g., a specific combination of `face_color` and `hair_color`, or `hair` style and `glasses`).

#### Reporting Obligations for Split:
The assignment explicitly commands that the final report must document:
- Which individual attributes appear in training;
- Which specific attribute combinations are held out;
- Exact example counts for $\mathcal{D}_{\text{train}}$, $\mathcal{D}_{\text{val}}$, $\mathcal{D}_{\text{test\_ind}}$, and $\mathcal{D}_{\text{test\_ood}}$;
- Formal verification that the caption vocabulary was constructed without using validation or test text;
- Saved split definitions and preprocessing configurations enabling exact reproduction.

---

## 4. Evaluation & Metrics Specifications

The evaluation protocol must assess visual fidelity, conditioning controllability, diversity, and computational efficiency across both in-distribution and out-of-distribution prompts.

### 4.1 Quantitative Quality Metrics
- **Fréchet Inception Distance (FID) / Kernel Inception Distance (KID):**
  - Evaluated on generated samples vs. ground-truth real avatar images.
  - Mandatory documentation: feature layer used (e.g. Inception-v3 pool3 / 64 or 2048 dims), normalization range, sample count, and KID subset size (e.g. `subset_size=50`).
  - KID is strongly recommended due to its unbiased estimator properties on smaller evaluation sets.

### 4.2 Conditioning & Controllability Metrics
- **Compositional Generalization Comparison:**
  - Quantitative and qualitative comparison of model performance on ordinary in-distribution test prompts vs. held-out compositional OOD prompts.
  - Evaluation of whether the model generates the held-out attribute combination when prompted, or collapses to a seen training combination.

### 4.3 Diversity Across Seeds
- **Perceptual Diversity (LPIPS across seeds):**
  - Generates multiple samples (e.g., 4 or more random seeds) for the exact same text prompt.
  - Measures pairwise perceptual distance (Learned Perceptual Image Patch Similarity) across seeds to confirm that the model does not suffer from mode collapse.

### 4.4 Computational & Efficiency Metrics
The assignment specifies mandatory measurement and reporting of:
1. **Parameter Count:** Explicit breakdown of:
   - Denoiser U-Net parameters;
   - Text Encoder parameters;
   - Total trainable parameter count.
2. **Sampling Latency:** Wall-clock inference time (in seconds) required to generate an image or batch across all $T$ reverse steps.
3. **Memory Footprint:** Peak GPU VRAM allocated during training and during reverse sampling (in MB or GB).

### 4.5 Required Experimental Runs
The PDF lists two mandatory experimental baselines:
1. **Unconditional Neural Baseline:**
   - Diffusion model trained without text conditioning (or text conditioning always set to null/empty embedding). Establishes the baseline visual distribution modeling capacity.
2. **Conditional Diffusion Model:**
   - Diffusion model trained with the chosen text conditioning mechanism (e.g., cross-attention) and CFG dropout.
3. **Smoke Test Overfitting Verification:**
   - Preliminary run on $32\times 32$ resolution on a tiny subset to verify convergence before full-scale training.

---

## 5. Deliverables & Submission Requirements

| Deliverable Artifact | Description & Required Contents | Criteria for Evaluation |
|---|---|---|
| **Python Codebase** | Well-commented, modular implementation covering data preprocessing, model architectures, training loop, inference interface, and metric evaluation. | Code clarity, absence of forbidden pretrained weights, modularity, strict reproducibility. |
| **Comprehensive Technical Report** | Detailed document covering: preprocessing methodology, vocabulary construction, compositional split definition, model architectures, training dynamics, loss curves, experimental comparisons (unconditional vs conditional, in-distribution vs OOD), metrics tables, challenges encountered, and recommendations. | Theoretical soundness, thorough experimental rigor, clear answers to the research question. |
| **Oral Exam Presentation** | Presentation slide deck summarizing the project motivation, technical implementation, qualitative avatar generations, quantitative evaluation results, and architectural lessons. | Clarity of communication, mastery of deep learning concepts, defense during oral examination. |
| **Reproducibility Assets** | Checkpoint saving and resumption mechanisms, serialized split definitions, cached vocabularies, and fixed random seeds. | Independence of runs, exact replicability of splits and weights. |

---

## 6. Features Discovered

The following feature registry reflects all capabilities, components, and behavioral requirements identified across the authoritative PDF specification and project artifacts:

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|---|---|---|---|---|---|---|
| 1 | Constraint | From-Scratch Text Encoder | 2–4 layer Transformer encoder trained without pretrained weights | Token IDs $[B, L]$ | Context tensor $[B, L, d_{\text{model}}]$ | Fails if pretrained checkpoint loaded | PDF §2, §4, §5 |
| 2 | Constraint | From-Scratch Denoiser U-Net | Compact pixel-space U-Net with skip connections and GroupNorm | Noisy image $x_t$, timestep $t$, context $c$ | Predicted noise $\hat{\epsilon} \in \mathbb{R}^{B \times 3 \times H \times W}$ | Shape mismatch if channel dimensions misaligned | PDF §2, §5 |
| 3 | Constraint | Pure Pixel-Space DDPM | Diffusion operating directly on image pixels ($32\times 32$ or $64\times 64$), no VAE | RGB images $[-1, 1]$ | RGB images $[-1, 1]$ | Banned if latent diffusion / VAE used | PDF §2, §4 |
| 4 | Constraint | Ban on Pretrained Checkpoints | Absolute prohibition of Stable Diffusion, CLIP, T5, BERT, Diffusers black-boxes | Config / Checkpoint loading | Weight initialization | Immediate audit failure if detected | PDF §4 |
| 5 | Data | Deterministic Caption Generation | Converts avatar categorical attributes into structured descriptive text | Attribute dictionary / CSV row | Text prompt string | Fallback caption on missing keys | PDF §3, §4 |
| 6 | Data | Train-Only Vocabulary Extraction | Builds tokenizer vocabulary strictly from training partition text | Training texts list | JSON vocabulary dictionary | `<UNK>` assigned to out-of-vocabulary tokens | PDF §3, §4 |
| 7 | Data | Compositional Generalization Split | Partitions data based on blocked attribute combinations | Metadata dictionary list | 4 subsets: train, val, ordinary test, OOD test | Leakage error if OOD combination in training | PDF §4 |
| 8 | Architecture | Sinusoidal Timestep Embeddings | Computes frequency-based positional encodings for timestep $t$ | Timestep tensor $t \in [0, T-1]$ | Time embedding $[B, 4 C_{\text{base}}]$ | Assertion failure if negative timestep | PDF §5, Ho et al. |
| 9 | Architecture | Spatial Cross-Attention | Attends image visual patches to text token representations | Visual feature map $[B, C, H, W]$, text context $[B, L, d]$ | Conditioned feature map $[B, C, H, W]$ | Dimension error if $d \ne \text{context\_dim}$ | PDF §5 |
| 10 | Architecture | Spatial Self-Attention | Computes global self-attention across visual spatial positions | Visual feature map $[B, C, H, W]$ | Refined feature map $[B, C, H, W]$ | Memory explosion if applied at high resolution | PDF §5 |
| 11 | DDPM Math | Cosine Noise Schedule | Cosine-squared cumulative variance schedule $\bar{\alpha}_t$ | Timestep count $T$, offset $s=0.008$ | Precomputed $\bar{\alpha}_t, \alpha_t, \beta_t$ tensors | Clamped at $\beta_{\max}=0.999$ to avoid singularity | PDF §5, Nichol & Dhariwal |
| 12 | DDPM Math | Closed-Form Forward Noising | Direct addition of Gaussian noise at arbitrary timestep $t$ | Clean image $x_0$, noise $\epsilon$, timestep $t$ | Noisy image $x_t$ | Device mismatch if tensors on different devices | PDF §2, §7 |
| 13 | DDPM Math | Epsilon Prediction MSE Loss | Denoising regression objective comparing predicted to actual noise | $\hat{\epsilon}_\theta(x_t, t, c)$, injected noise $\epsilon$ | Scalar MSE loss | NaN/Inf if gradients explode | PDF §7 |
| 14 | DDPM Math | Stochastic Reverse Sampling | Iterative denoising loop from $x_T$ to $x_0$ with Langevin noise | Noisy image $x_t$, timestep $t$, context $c$ | Denoised state $x_{t-1}$ | Clamped variance to avoid $\sqrt{\le 0}$ | PDF §2, §7 |
| 15 | Sampling | Classifier-Free Guidance (CFG) | Linearly extrapolates conditional over unconditional noise | $x_t, t, c_{\text{cond}}, c_{\text{uncond}}, w$ | Guided noise prediction $\hat{\epsilon}$ | Degrades visual quality if $w$ excessively high | PDF §2, §7, Ho & Salimans |
| 16 | Experiment | Unconditional Neural Baseline | Baseline diffusion run without text conditioning for comparison | Unconditional prompt / null context | Baseline generated samples | N/A | PDF §7 |
| 17 | Experiment | Conditional Model Run | Main diffusion run conditioned on structured multi-attribute text | Text prompts, random seeds | Conditioned avatar samples | Blurry/generic output if conditioning fails | PDF §7 |
| 18 | Metric | Image Quality (FID / KID) | Computes Fréchet or Kernel Inception Distance against real set | Generated images, real test images | FID scalar score, KID mean $\pm$ std | Error if image count $<$ KID subset size | PDF §7 |
| 19 | Metric | Perceptual Diversity Across Seeds | Pairwise LPIPS distance across multiple seeds for identical prompt | Image batch $[N_{\text{seeds}}, 3, H, W]$ | Mean LPIPS diversity score | Requires $N \ge 2$ seeds | PDF §7 |
| 20 | Metric | Computational Efficiency Tracking | Tracks trainable parameters, latency, and peak VRAM | Model modules, sample function | Parameter counts, latency (s), VRAM (MB) | CUDA synchronization required for accurate timing | PDF §7 |
| 21 | Interface | Interactive Prompt & Seed Generation | User enters text prompt, chooses seed, inspects $32\times 32$ or $64\times 64$ avatar | String prompt, integer seed | Visual image plot & tensor | Out-of-vocabulary words mapped to `<UNK>` | PDF §1, §8 |
| 22 | Interface | Compositional OOD Evaluation Script | Batch generation across held-out OOD prompts across multiple seeds | List of OOD prompt strings, seed count | Comparative multi-seed plot grid | Verifies unseen attribute synthesis | PDF §1, §4, §8 |

---

## 7. Edge Cases & Boundary Conditions

| # | Feature / Subsystem | Edge Case Input / Condition | Expected / Observed Theoretical Behavior | Potential Failure Mode & Mitigation |
|---|---|---|---|---|
| 1 | Tokenizer Vocabulary | Words in test prompts not present in training vocabulary (e.g. "exaggerated", novel colors) | Tokenizer assigns `<UNK>` index; text encoder processes token without crashing | If tokenizer does not support `<UNK>`, key error occurs; model must be trained with `<UNK>` |
| 2 | Reverse Sampling | Final timestep $t=0$ in reverse sampling loop | Noise term $\sigma_0 z$ must be zeroed out; loop must return clean mean $\mu_\theta(x_1, 1)$ | Injecting noise at $t=0$ leaves residual grain/noise on final generated avatar |
| 3 | Classifier-Free Guidance | Guidance scale set to $w = 1.0$ vs $w > 5.0$ | At $w=1.0$, standard conditional sampling; at $w > 5.0$, image saturation / contrast distortion | Must provide configurable guidance scale ($w \approx 2.5 - 4.0$) with value clamping |
| 4 | Forward Noise Schedule | Step $t=T$ with cosine schedule near boundary | $\bar{\alpha}_T \to 0$, leading to $\beta_T \to 1.0$ and potential division by zero | Betas must be clamped: $\beta_t \le 0.999$, ensuring numerical stability |
| 5 | Compositional Splitting | Incomplete metadata or key mismatch (e.g., config using `color`/`proportion` while dataset uses `face_color`/`hair`) | Splitter fails to match blocked subsets, resulting in zero OOD samples (everything treated as train) | Splitter config keys must strictly match dataset CSV column names |
| 6 | Dataset Slicing | Attribute leakage where held-out attribute combination accidentally enters training split | Model observes the combination during training, completely invalidating the research question | Mandatory verification assertion: `verify_ood_isolation(train_metadata) == True` |
| 7 | KID Metric Computation | Evaluating on very small batch of samples (e.g. $N < \text{subset\_size}$) | `torchmetrics` KernelInceptionDistance raises exception if sample count $< \text{subset\_size}$ | Ensure evaluation batch size $\ge \text{subset\_size}$ (e.g., at least 50–100 samples) |
| 8 | Image Normalization | Mismatch between preprocessing normalization $[-1, 1]$ and metric expectations $[0, 1]$ | Metric yields distorted Inception feature activations or negative values | Explicit rescaling check: `(img + 1.0) / 2.0` before passing to FID/KID/LPIPS |
| 9 | Spatial Cross-Attention | High spatial resolution ($64\times 64$ with 4096 tokens) in early U-Net downsampling layers | Quadratic memory complexity $\mathcal{O}((HW)^2)$ can cause VRAM out-of-memory on 16GB GPU | Restrict heavy self-attention to bottleneck / low resolutions; use FlashAttention (`scaled_dot_product_attention`) |
| 10 | Unconditional Conditioning | Text dropout during training with `cfg_drop_rate=0.1` | Tensor with all `<PAD>` tokens passed through text encoder to establish unconditional representation | Ensure masking properly disables self-attention on padding tokens in Transformer encoder |

---

## 8. Requirements Traceability Matrix

This matrix maps every directive in `Deep_Learning_2026_VI 1.pdf` directly to verification criteria:

| PDF Section | Requirement Statement | Mandatory Criteria | Verification Target in Project |
|---|---|---|---|
| **§1 Task Overview** | Inspectable generation from prompt & seed | User prompt input, integer seed selection, visual output at $32\times 32$ or $64\times 64$ | `inference.py` interactive loop |
| **§1 Task Overview** | Generalize to unseen attribute combinations | Compositional generalization split and OOD prompt testing | `preprocessing/splitter.py`, `inference.py` |
| **§2 Sub-Objectives** | Reproducible preprocessing pipeline | Deterministic captioning, image resizing/normalization, cached splits | `preprocessing/` scripts & config |
| **§2 Sub-Objectives** | From-scratch text tokenizer & encoder | Built ground-up; 2–4 layer Transformer; no CLIP/BERT/T5 | `preprocessing/tokenizer.py`, `models/transformer.py` |
| **§2 Sub-Objectives** | Compact U-Net denoiser from scratch | Custom U-Net with skip connections, residual blocks, time embeddings | `models/unet.py`, `models/unet_parts.py` |
| **§2 Sub-Objectives** | Text conditioning mechanism | Cross-attention or modulation mapping text context to visual features | `models/unet_parts.py` `SpatialCrossAttention` |
| **§2 Sub-Objectives** | Complete DDPM formulation | Noise schedule, forward noising, reverse sampling, checkpointing | `models/diffusion.py`, `train.py` |
| **§3 Dataset** | Google Cartoon Set with deterministic captions | Multi-attribute captions generated from metadata; 10k or 100k version | `data/meta/`, `caption_generator.py` |
| **§3 Dataset** | Vocabulary from training split only | Zero vocabulary construction leakage from val or test sets | `preprocessing/tokenizer.py` |
| **§4 Compositional Split** | Split over attribute combinations | Controlled hold-out of attribute combinations absent from train | `preprocessing/splitter.py` |
| **§4 Compositional Split** | 4 subsets reported | Report train, validation, ordinary test, and compositional-OOD counts | Dataset splitting summary / report |
| **§4 Constraints** | Forbidden pretrained models | Zero tolerance for Stable Diffusion, CLIP, T5, pretrained VAE, Diffusers | Codebase dependencies & model definitions |
| **§5 Architecture** | Compact envelope | $C_{\text{base}} \in [64, 128]$, 2–3 resolutions, $d_{\text{model}} \in [64, 128]$, 2–4 layers | Model instantiation arguments |
| **§6 Hints** | 32x32 smoke test & reproducibility | Seed fixing, checkpoint save/resume, caching of vocab and splits | `main.py`, `train.py`, `preprocessing/` |
| **§7 Training** | Unconditional baseline & conditional model | Both unconditional baseline and conditional model trained and analyzed | `train.py` (`conditional=False` and `True`) |
| **§7 Metrics** | Image quality metric (FID/KID) | Quantitative FID or KID with all implementation details documented | `metrics.py` `compute_quality_metrics` |
| **§7 Metrics** | Diversity across seeds | LPIPS / perceptual diversity measured across seeds for identical prompts | `metrics.py` `compute_diversity_across_seeds` |
| **§7 Metrics** | Efficiency metrics | Total/U-Net/Text Encoder parameter count, sampling time, peak VRAM | `metrics.py` `compute_efficiency_metrics` |
| **§8 Deliverables** | Code, Report, Analysis, Presentation | Commented code, academic report, research question analysis, slides | Submission bundle |
