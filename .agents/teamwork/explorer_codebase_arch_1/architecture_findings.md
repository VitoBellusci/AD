# Comprehensive Architecture & Diffusion Mathematics Audit Report

**Audit Target**: Text-Conditioned Avatar Diffusion Project  
**Auditor**: `explorer_codebase_arch_1` (Teamwork Explorer Agent)  
**Date**: October 5, 2026  
**Reference Specification**: Politecnico di Bari - Deep Learning 2026, Test Code: 2026_VI (`Deep_Learning_2026_VI 1.pdf`)

---

## 1. Executive Summary & Compliance Matrix

This report delivers an exhaustive, line-by-line architectural and mathematical audit of the text-conditioned avatar diffusion model codebase (`models/unet.py`, `models/unet_parts.py`, `models/diffusion.py`, `models/transformer.py`, `train.py`, `main.py`, `inference.py`, `metrics.py`, and `preprocessing/`).

| Area | Requirement / Dimension | Codebase Implementation | Compliance Status | Key Notes / Severity |
| :--- | :--- | :--- | :--- | :--- |
| **From-Scratch** | No pretrained diffusion checkpoints (SD, Flux, etc.) | Custom U-Net (`models/unet.py`) | **COMPLIANT** | Zero pretrained generative weights |
| **From-Scratch** | No pretrained text models (CLIP, T5, BERT) | Custom Transformer (`models/transformer.py`) | **COMPLIANT** | Pure PyTorch modules trained from scratch |
| **From-Scratch** | No ready-made Diffusers / black-box pipelines | Custom DDPM engine (`models/diffusion.py`) | **COMPLIANT** | Forward & reverse processes written from scratch |
| **Architecture** | Compact U-Net ("Tiny" parameter budget) | 3-stage U-Net (~8.14M params) + Text Enc (~0.42M params) | **COMPLIANT** | Total ~8.56M parameters; conforms to assignment envelope |
| **Architecture** | Normalization & Activations | GroupNorm(8, C) + SiLU | **SOTA ALIGNED** | Consistent with modern diffusion standards |
| **Architecture** | Downsampling & Upsampling | MaxPool2d + Bilinear Upsample | **SUB-OPTIMAL** | MaxPool introduces spatial discontinuities vs strided Conv |
| **Diffusion Math** | Noise schedule | Cosine schedule (Nichol & Dhariwal 2021) | **COMPLIANT** | Mathematically sound with beta clipping |
| **Diffusion Math** | Forward process $q(x_t \vert x_0)$ | $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ | **COMPLIANT** | Exact analytical formulation |
| **Diffusion Math** | Reverse sampling $p_\theta(x_{t-1} \vert x_t)$ | Mean $\mu_\theta$ exact; variance $\sigma_t^2 = \beta_t$ | **COMPLIANT** | Correct DDPM formulation; noise suppressed at $t=0$ |
| **Diffusion Math** | Objective / Loss function | Epsilon prediction MSE Loss | **COMPLIANT** | $L_{\text{simple}} = \mathbb{E}[\|\epsilon - \epsilon_\theta\|^2]$ |
| **Conditioning** | Timestep embedding injection | Sinusoidal PE + 2-layer MLP + Additive projection | **SOTA ALIGNED** | Follows Ho et al. / ADM standards |
| **Conditioning** | Text conditioning mechanism | Multi-Head Spatial Cross-Attention | **FLAW IDENTIFIED** | Unmasked padding in cross-attention; attention at 64x64 |
| **Conditioning** | Classifier-Free Guidance (CFG) | Dropout during train (0.1) + Dual-pass sampling | **COMPLIANT** | Batch concatenation used at inference |
| **Bug / Flaw** | Attention pad masking in `transformer.py` | `attention_scores.masked_fill(mask == 0, -1e-9)` | **CRITICAL BUG** | Typed `-1e-9` instead of `-1e9` ($e^{-1e-9} \approx 1.0$) |
| **Bug / Flaw** | Vocabulary index collision in `tokenizer.py` | `word_count = 0` in `fit()` overwrites `<UNK>, <SOS>, <EOS>` | **CRITICAL BUG** | Clobbers special tokens; corrupts inverse mapping |
| **Bug / Flaw** | LayerNormalization implementation | Scaled by single scalar `nn.Parameter(torch.ones(1))` | **MINOR FLAW** | Not per-channel affine parameters |

---

## 2. Model Architecture & Parameter Budget Deep Dive

### 2.1 Exact Structure of the Custom U-Net (`models/unet.py`, `models/unet_parts.py`)

The denoiser model is implemented in `models/unet.py` (`class Unet(nn.Module)`). It operates on pixel-space images $[B, 3, H, W]$ at $64 \times 64$ (or $32 \times 32$) resolution.

```
                  [Input: x (3, 64, 64), t (scalar), context (B, 20, 128)]
                                         │
                                [time_mlp: 64 -> 256 -> 256]
                                         │
┌───────────────────────────────────────┴───────────────────────────────────────┐
│ ENCODER (Downsampling Path)                                                   │
│   • inc: DoubleConv(3 -> 64, time_emb=256)                                    │
│   • attn_inc: SpatialCrossAttention(query_dim=64, context_dim=128) -> skip1   │
│   • down1: MaxPool2d(2) -> DoubleConv(64 -> 128, time_emb=256) (32x32)       │
│   • attn_down1: SpatialCrossAttention(query_dim=128, context_dim=128) -> skip2│
│   • down2: MaxPool2d(2) -> DoubleConv(128 -> 256, time_emb=256) (16x16)      │
│   • self_attn_down2: SpatialSelfAttention(dim=256)                            │
│   • attn_down2: SpatialCrossAttention(query_dim=256, context_dim=128) -> skip3│
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │
┌───────────────────────────────────────┴───────────────────────────────────────┐
│ BOTTLENECK (Resolution 16x16)                                                 │
│   • bott1: DoubleConv(256 -> 256, time_emb=256)                               │
│   • self_attn_bott: SpatialSelfAttention(dim=256)                             │
│   • attn_bott1: SpatialCrossAttention(query_dim=256, context_dim=128)         │
│   • bott2: DoubleConv(256 -> 256, time_emb=256)                               │
└───────────────────────────────────────┬───────────────────────────────────────┘
                                        │
┌───────────────────────────────────────┴───────────────────────────────────────┐
│ DECODER (Upsampling Path)                                                     │
│   • up1: Upsample(bilinear, 16x16 -> 32x32) + concat skip2 (256+128 = 384 ch)  │
│          DoubleConv(384 -> 128, mid_channels=192, time_emb=256)               │
│   • self_attn_up1: SpatialSelfAttention(dim=128)                              │
│   • attn_up1: SpatialCrossAttention(query_dim=128, context_dim=128)          │
│   • up2: Upsample(bilinear, 32x32 -> 64x64) + concat skip1 (128+64 = 192 ch)   │
│          DoubleConv(192 -> 64, mid_channels=96, time_emb=256)                │
│   • attn_up2: SpatialCrossAttention(query_dim=64, context_dim=128)           │
│   • out: OutConv(64 -> 3, kernel_size=1)                                      │
└───────────────────────────────────────────────────────────────────────────────┘
```

#### Detailed Component Inspection:
1. **Residual Double-Convolution Block (`DoubleConv`)**:
   - `conv1`: `Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)`
   - `gn1`: `GroupNorm(8, mid_channels)`
   - `act1`: `nn.SiLU()`
   - `time_emb_proj`: `nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, mid_channels))`
   - `conv2`: `Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)`
   - `gn2`: `GroupNorm(8, out_channels)`
   - `act2`: `nn.SiLU()`
   - `residual_conv`: `Conv2d(in_channels, out_channels, kernel_size=1)` if `in_channels != out_channels` else `nn.Identity()`.
   - **Forward pass**: `residual = residual_conv(x)`, `x = conv1(x)`, `x = x + time_proj(t)[:, :, None, None]`, `x = conv2(x) + residual`.
2. **Downsampling (`Down`)**:
   - Uses `nn.MaxPool2d(2)` followed by `DoubleConv`.
   - *Architectural Critique*: While functionally operative, standard modern diffusion denoisers (DDPM, ADM, Stable Diffusion) replace MaxPool with strided convolutions (`Conv2d(..., stride=2)`) or average pooling (`AvgPool2d(2)`). MaxPool creates non-differentiable hard boundary selections, discarding 75% of pixel gradients and causing high-frequency aliasing in the pixel-space denoising trajectory.
3. **Upsampling (`Up`)**:
   - Implements `nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)` with channel concatenation and dimension alignment via `F.pad`.
   - Avoids checkerboard artifacts typical of transposed convolutions (`ConvTranspose2d`).
4. **Normalization & Activations**:
   - `GroupNorm(8, num_channels)` is strictly used across all convolutional blocks. Since channel depths (64, 96, 128, 192, 256) are all multiples of 8, GroupNorm computes stably without channel division remainder issues. GroupNorm avoids batch-size dependency during training, adhering to SOTA diffusion practices.
   - `nn.SiLU()` (Swish) activation is utilized throughout the model, ensuring smooth continuous gradients.

---

### 2.2 Exact Parameter Count Breakdown

The assignment specification (`Deep_Learning_2026_VI 1.pdf`, Section 5) specifies a **"Tiny"** model envelope:
- Base channels: 64–128
- Two or three spatial resolutions
- Residual blocks with time embeddings
- 2–4 layer Transformer text encoder trained from scratch
- Text hidden size: 64–128
- Conditioning injected through cross-attention or other techniques

Below is the verified mathematical parameter count for the exact configuration instantiated in `main.py` (`base_channels=64`, `context_dim=128`, `vocab_size=200`, `max_seq_len=20`):

#### A. U-Net Parameters

| Module Name | Sub-Layers & Operations | Formula / Dimensions | Trainable Parameters |
| :--- | :--- | :--- | :--- |
| **`time_mlp`** | Linear(64, 256) + Linear(256, 256) | $(64 \times 256 + 256) + (256 \times 256 + 256)$ | **82,432** |
| **`inc`** | DoubleConv(3, 64, time=256) | $3\times64\times9 + 128 + (256\times64+64) + 64\times64\times9 + 128 + (3\times64\times1+64)$ | **55,552** |
| **`down1`** | Down(64, 128, time=256) | $64\times128\times9 + 256 + (256\times128+128) + 128\times128\times9 + 256 + (64\times128+128)$ | **262,912** |
| **`down2`** | Down(128, 256, time=256) | $128\times256\times9 + 512 + (256\times256+256) + 256\times256\times9 + 512 + (128\times256+256)$ | **984,576** |
| **`bott1`** | DoubleConv(256, 256, time=256) | $256\times256\times9 + 512 + (256\times256+256) + 256\times256\times9 + 512 + 0$ | **1,246,464** |
| **`bott2`** | DoubleConv(256, 256, time=256) | $256\times256\times9 + 512 + (256\times256+256) + 256\times256\times9 + 512 + 0$ | **1,246,464** |
| **`up1.conv`** | DoubleConv(384, 128, mid=192) | $384\times192\times9 + 384 + (256\times192+192) + 192\times128\times9 + 256 + (384\times128+128)$ | **984,000** |
| **`up2.conv`** | DoubleConv(192, 64, mid=96) | $192\times96\times9 + 192 + (256\times96+96) + 96\times64\times9 + 128 + (192\times64+64)$ | **258,528** |
| **`out`** | OutConv(64, 3) | $64 \times 3 \times 1 + 3$ | **195** |
| **Subtotal (Convolutions & Time MLP)** | | | **5,121,123** |
| **`self_attn_down2`** | SpatialSelfAttention(dim=256, heads=8, dim_head=64) | GroupNorm(256) + $3 \times (256 \times 512) + (512 \times 256 + 256)$ | **525,056** |
| **`self_attn_bott`** | SpatialSelfAttention(dim=256, heads=8, dim_head=64) | GroupNorm(256) + $3 \times (256 \times 512) + (512 \times 256 + 256)$ | **525,056** |
| **`self_attn_up1`** | SpatialSelfAttention(dim=128, heads=8, dim_head=64) | GroupNorm(128) + $3 \times (128 \times 512) + (512 \times 128 + 128)$ | **262,528** |
| **Subtotal (Self-Attention)** | | | **1,312,640** |
| **`attn_inc`** | SpatialCrossAttention(q=64, ctx=128, inner=512) | GN(64) + $64\times512 + 2\times(128\times512) + (512\times64+64)$ | **196,800** |
| **`attn_down1`** | SpatialCrossAttention(q=128, ctx=128, inner=512) | GN(128) + $128\times512 + 2\times(128\times512) + (512\times128+128)$ | **262,528** |
| **`attn_down2`** | SpatialCrossAttention(q=256, ctx=128, inner=512) | GN(256) + $256\times512 + 2\times(128\times512) + (512\times256+256)$ | **393,984** |
| **`attn_bott1`** | SpatialCrossAttention(q=256, ctx=128, inner=512) | GN(256) + $256\times512 + 2\times(128\times512) + (512\times256+256)$ | **393,984** |
| **`attn_up1`** | SpatialCrossAttention(q=128, ctx=128, inner=512) | GN(128) + $128\times512 + 2\times(128\times512) + (512\times128+128)$ | **262,528** |
| **`attn_up2`** | SpatialCrossAttention(q=64, ctx=128, inner=512) | GN(64) + $64\times512 + 2\times(128\times512) + (512\times64+64)$ | **196,800** |
| **Subtotal (Cross-Attention)** | | | **1,706,624** |
| **TOTAL U-NET PARAMETERS** | | | **8,140,387 (~8.14 M)** |

#### B. FullTextEncoder Parameters (`models/transformer.py`)

| Module Name | Sub-Layers & Operations | Formula / Dimensions | Trainable Parameters |
| :--- | :--- | :--- | :--- |
| **`embed`** | InputEmbeddings(vocab_size=200, d_model=128) | $200 \times 128$ | **25,600** |
| **`pos_enc`** | PositionalEncoding(max_len=20, d_model=128) | Fixed sinusoidal buffer (`pe`) | **0** |
| **Each `EncoderBlock` (x3)** | | | |
| - Self-Attention | MultiHeadAttentionBlock(d_model=128, h=4) | $4 \times (128 \times 128 + 128)$ ($W_q, W_k, W_v, W_o$) | **66,048** |
| - Feed-Forward | FeedForwardBlock(d_model=128, d_ff=256) | $(128 \times 256 + 256) + (256 \times 128 + 128)$ | **65,920** |
| - LayerNorms (x2) | 2x ResidualConnection (LayerNormalization) | $2 \times (\alpha \in \mathbb{R}^1 + \beta \in \mathbb{R}^1)$ | **4** |
| - Total per Block | $66,048 + 65,920 + 4$ | | **131,972** |
| **3 Encoder Blocks** | $3 \times 131,972$ | | **395,916** |
| **Final LayerNorm** | LayerNormalization | $\alpha \in \mathbb{R}^1 + \beta \in \mathbb{R}^1$ | **2** |
| **TOTAL TEXT ENCODER** | | | **421,518 (~0.42 M)** |

#### C. Grand Total Parameter Count:
$$\text{Total Parameters} = 8,140,387 + 421,518 = \mathbf{8,561,905} \text{ (~8.56 Million)}$$

- **Compliance with Assignment Envelope**: Fully complies. The model size sits comfortably below 10M parameters, fitting well within the assigned "Tiny" model budget and easily trainable within a single-GPU (NVIDIA T4 16GB) memory profile.

---

## 3. Mandatory From-Scratch Compliance Audit

### 3.1 Prohibited Components Verification

The assignment PDF mandates:
> *"The following components are forbidden in the main solution: Stable Diffusion, Tiny-SD, SDXL, Flux, or any other pretrained diffusion checkpoint; CLIP, T5, BERT, or any pretrained text encoder used for conditioning; a pretrained VAE or latent diffusion pipeline; a ready-made Diffusers training pipeline used as a black box; pretrained image or text embeddings as the main representation."*

**Codebase Verification:**
1. **Model Hubs / Pretrained Weights**:
   - `grep` search across all project files confirms zero imports of `transformers`, `diffusers`, `timm`, `clip`, or Hugging Face hub checkpoints.
   - All modules inherit directly from `torch.nn.Module` and initialize with default random initialization (`torch.nn.init`).
2. **Text Encoder Architecture**:
   - Implemented completely from scratch in `models/transformer.py`.
   - Embeddings are learned lookup matrices (`nn.Embedding(vocab_size, d_model)`).
   - Positional encodings are mathematically computed sine/cosine waveforms.
   - Transformer blocks implement manual query, key, value linear projections and multi-head attention.
3. **Denoiser U-Net**:
   - No pretrained UNet2DConditionModel or external backbone.
   - Convolutions, GroupNorm layers, down/up blocks, and attention blocks are written explicitly.
4. **Diffusion Framework**:
   - No `DDPMScheduler` from Hugging Face `diffusers`.
   - The forward analytical noising formula and reverse generation loop are handwritten in `models/diffusion.py`.
5. **Caveat on Evaluation Metrics (`metrics.py`)**:
   - In `metrics.py`, `torchmetrics` imports `FrechetInceptionDistance`, `KernelInceptionDistance`, and `LearnedPerceptualImagePatchSimilarity`.
   - These modules load pretrained Inception-v3 and VGG networks **strictly for evaluation metrics computation**, which is the universal standard for FID/KID/LPIPS calculation.
   - None of these weights are utilized in the training loop, conditioning pipeline, or generation process. The generative pipeline is 100% compliant with from-scratch requirements.

---

## 4. Diffusion Mathematics & Implementation Correctness (DDPM)

### 4.1 Forward Diffusion Process ($q(x_t \vert x_0)$)

In `models/diffusion.py` (`DiffusionScheduler`, `DiffusionForwardProcess`):

1. **Cosine Noise Schedule Formulation**:
   The implementation uses the cosine variance schedule from Nichol & Dhariwal (2021) (*Improved Denoising Diffusion Probabilistic Models*):
   $$f(t) = \cos\left( \frac{\frac{t}{T} + s}{1 + s} \cdot \frac{\pi}{2} \right)^2, \quad s = 0.008$$
   $$\bar{\alpha}_t = \frac{f(t)}{f(0)}$$
   $$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, \quad \beta_t \gets \min(\beta_t, 0.999)$$
   $$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{i=1}^t \alpha_i$$

   *Code Verification (`models/diffusion.py`, lines 17–38)*:
   ```python
   steps = torch.arange(num_time_steps + 1, dtype=torch.float32, device=self.device)
   f_t = torch.cos(((steps / num_time_steps + s) / (1 + s)) * (math.pi / 2))**2
   ab_full = f_t / f_t[0]
   self.alpha_bars = ab_full[1:].to(self.device)
   ab_prev = ab_full[:-1].to(self.device)
   alphas = self.alpha_bars / ab_prev
   betas = 1 - alphas
   self.betas = torch.clamp(betas, max=0.999).to(self.device)
   self.alphas = 1 - self.betas
   self.alpha_bars = torch.cumprod(self.alphas, dim=0)
   ```
   **Evaluation**: Mathematically exact. Clamping $\beta_t \le 0.999$ prevents numerical singularity near $t \to T$. Re-computing $\bar{\alpha}_t$ from clamped $\alpha_t$ guarantees consistency.

2. **Marginal Noising Formula ($q(x_t \vert x_0)$)**:
   In DDPM, the distribution of $x_t$ conditioned directly on clean image $x_0$ is:
   $$q(x_t \vert x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)\mathbf{I}\right)$$
   Sampling formula:
   $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

   *Code Verification (`models/diffusion.py`, lines 51–64)*:
   ```python
   sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].to(original.device)[:, None, None, None]
   sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].to(original.device)[:, None, None, None]
   return (sqrt_alpha_bar_t * original) + (sqrt_one_minus_alpha_bar_t * noise)
   ```
   **Evaluation**: Fully verified and mathematically exact.

---

### 4.2 Reverse Sampling Process ($p_\theta(x_{t-1} \vert x_t)$)

In `models/diffusion.py` (`DiffusionReverseProcess.sample`) and `inference.py`:

1. **Posterior Mean Estimation**:
   According to Ho et al. (2020), the mean of the reverse transition $p_\theta(x_{t-1} \vert x_t)$ parameterised by noise prediction $\epsilon_\theta(x_t, t, c)$ is:
   $$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, c) \right)$$

   *Code Verification (`models/diffusion.py`, lines 111–119)*:
   ```python
   inv_sqrt_alpha_t = self.inv_sqrt_alphas[t].to(x.device)[:, None, None, None]
   beta_over_sqrt_one_minus_alpha_bar_t = self.beta_over_sqrt_one_minus_alpha_bar[t].to(x.device)[:, None, None, None]
   mean = inv_sqrt_alpha_t * (x - beta_over_sqrt_one_minus_alpha_bar_t * predicted_noise)
   ```
   **Evaluation**: Exactly matches the theoretical formulation.

2. **Posterior Variance Choice**:
   In `models/diffusion.py` line 135:
   $$\sigma_t = \sqrt{\beta_t}$$
   ```python
   sigma_t = torch.sqrt(torch.clamp(beta_t, min=1e-20))
   return mean + sigma_t * z
   ```
   *Theoretical Assessment*: In DDPM, two choices of fixed variance are common:
   - $\sigma_t^2 = \beta_t$ (upper bound)
   - $\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$ (posterior variance lower bound)
   Using $\sigma_t^2 = \beta_t$ is standard and valid.

3. **Boundary Condition at $t = 0$**:
   Lines 122–123 & `inference.py` line 79:
   ```python
   if (t == 0).all() or noise_free:
       return mean
   ```
   At $t = 0$, $z$ is suppressed and the mean $\mu_\theta$ is returned directly. This prevents injecting unnecessary stochastic noise into the final reconstructed image.

---

### 4.3 Loss Function / Training Objective

In `train.py` lines 94–134:
- Timesteps $t$ are sampled uniformly from $\{0, 1, \dots, T-1\}$:
  ```python
  timesteps = torch.randint(0, forward_process.num_time_steps, (batch_size,), device=device).long()
  ```
- Random Gaussian noise $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ is added.
- The U-Net predicts the noise: $\hat{\epsilon} = \epsilon_\theta(x_t, t, c)$.
- Optimization uses standard MSE:
  ```python
  loss = criterion(predicted_noise, noise)
  ```
  This implements the standard DDPM $L_{\text{simple}}(\theta)$ objective:
  $$L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]$$
  This complies with Section 7 of the assignment specification.

---

## 5. Text-Conditioning & Timestep Embedding Injection

### 5.1 Timestep Embedding Injection

1. **Sinusoidal Position Embeddings**:
   `SinusoidalPositionEmbeddings(dim=64)` maps scalar timesteps $t$ into a 64-dimensional vector using geometric progression frequencies:
   $$\omega_k = \exp\left( - \frac{\ln(10000)}{D/2 - 1} \cdot k \right), \quad k \in [0, D/2 - 1]$$
   $$\text{emb}(t) = [\sin(t \cdot \omega), \cos(t \cdot \omega)] \in \mathbb{R}^{B \times 64}$$
2. **MLP Projection**:
   In `models/unet.py`:
   $$\text{Linear}(64 \to 256) \to \text{SiLU}() \to \text{Linear}(256 \to 256)$$
   Yields a 256-dimensional dense embedding $t_{\text{emb}}$.
3. **Residual Block Injection**:
   In `DoubleConv`:
   `time_emb_proj` maps $t_{\text{emb}} \to \text{mid\_channels}$.
   It is broadcast spatially to shape $[B, C, 1, 1]$ and added to features after `conv1` and before `conv2`:
   $$x \gets \text{conv1}(x) + \text{proj}(t_{\text{emb}})$$
   $$x \gets \text{conv2}(x) + \text{residual}(x)$$
   This matches the standard SOTA additive timestep conditioning in ADM and DDPM.

---

### 5.2 Text Conditioning & Spatial Cross-Attention

In `models/unet_parts.py`, `SpatialCrossAttention`:
$$\text{Query: } Q = x_{\text{flat}} W_Q, \quad W_Q \in \mathbb{R}^{C \times 512}$$
$$\text{Key: } K = c_{\text{text}} W_K, \quad W_K \in \mathbb{R}^{D_{\text{text}} \times 512}$$
$$\text{Value: } V = c_{\text{text}} W_V, \quad W_V \in \mathbb{R}^{D_{\text{text}} \times 512}$$
$$\text{Attention: } \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
$$\text{Output: } x + \text{Linear}_{512 \to C}(\text{Attn})$$

#### Key Findings and Architectural Issues:

1. **CRITICAL FLAW: Absence of Attention Mask in `SpatialCrossAttention`**:
   In `SpatialCrossAttention.forward(self, x, context)` (`models/unet_parts.py` line 246):
   The method accepts only `(x, context)` and has **no mask parameter**.
   Inside, it executes:
   ```python
   out = F.scaled_dot_product_attention(
       q, k, v, 
       dropout_p=self.to_out[1].p if self.training else 0.0
   )
   ```
   Without an attention mask, all visual tokens attend equally to padding tokens (`<PAD>`) as they do to genuine semantic attribute tokens. For a sequence length of 20 where only 5 tokens are words and 15 are `<PAD>`, a substantial fraction of cross-attention energy is dissipated onto meaningless padding embeddings!

2. **CRITICAL BUG: Attention Padding Mask Value in `transformer.py`**:
   In `models/transformer.py`, lines 138–139:
   ```python
   if mask is not None:
       attention_scores = attention_scores.masked_fill(mask == 0, -1e-9)
   ```
   *Analysis*:
   The code sets masked positions to $-10^{-9} = -0.000000001$.
   In attention masking, the intended value is $-\infty$ or $-10^9$ (or $-10^4$).
   Setting the logit to $-10^{-9}$ means that in the softmax calculation:
   $$\exp(-10^{-9}) \approx 1.000000000$$
   Padding tokens are **NOT suppressed**! Instead, they receive a non-zero exponent comparable to unmasked tokens. As a consequence, the self-attention blocks in `FullTextEncoder` attend heavily to `<PAD>` tokens.

3. **SUB-OPTIMAL DESIGN: Cross-Attention on Highest Spatial Resolution ($64 \times 64$)**:
   In `models/unet.py`:
   - `self.attn_inc = SpatialCrossAttention(64, context_dim)` (applied at $64 \times 64 = 4096$ tokens)
   - `self.attn_up2 = SpatialCrossAttention(64, context_dim)` (applied at $64 \times 64 = 4096$ tokens)
   With `inner_dim = 512` ($8 \times 64$), each of these blocks projects 64 channels up to 512 channels across 4,096 spatial positions.
   In modern diffusion models, cross-attention is intentionally restricted to compressed latent/spatial levels ($16 \times 16$ or $8 \times 8$). High-resolution feature maps represent fine textures that do not require global cross-attention to text, and operating attention at $64 \times 64$ inflates memory usage and computation unnecessarily.

4. **DEFICIENCY: Custom `LayerNormalization` with Scalar Affine Parameters**:
   In `models/transformer.py` lines 66–67:
   ```python
   self.alpha = nn.Parameter(torch.ones(1))
   self.bias = nn.Parameter(torch.zeros(1))
   ```
   In standard LayerNorm (`nn.LayerNorm(d_model)`), $\alpha$ and $\beta$ are $D$-dimensional vectors ($D = 128$) so each channel learns an independent gain and bias. Defining them as scalar `torch.ones(1)` restricts all 128 features to identical scaling, reducing model expressiveness.

---

### 5.3 Classifier-Free Guidance (CFG) Verification

1. **Training Protocol (`train.py`, lines 109–123)**:
   - With probability `cfg_drop_rate = 0.1`, conditioning is dropped:
     ```python
     drop_mask = torch.rand(batch_size, device=device) < cfg_drop_rate
     if drop_mask.any():
         uncond_context = get_unconditional_context(text_encoder, tokenizer, batch_size, text_tokens.shape[1], device)
         context = torch.where(drop_mask.unsqueeze(1).unsqueeze(2), uncond_context, context)
     ```
   - Enables the U-Net to learn both conditional and unconditional generation distributions simultaneously.
2. **Sampling Protocol (`models/diffusion.py`, lines 89–106)**:
   - Dual-forward pass evaluated efficiently via batch concatenation:
     ```python
     x_input = torch.cat([x, x], dim=0)
     t_input = torch.cat([t, t], dim=0)
     context_input = torch.cat([context, uncond_context], dim=0)
     all_noise = model(x_input, t_input, context=context_input)
     eps_cond, eps_uncond = torch.chunk(all_noise, 2, dim=0)
     predicted_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
     ```
   - Formula matches Ho & Salimans (2022) (*Classifier-Free Diffusion Guidance*).

---

## 6. Preprocessing & Vocabulary Discrepancies

### 6.1 CRITICAL BUG: `AvatarTokenizer.fit` Vocabulary Corruption
In `preprocessing/tokenizer.py`, lines 12–27:
```python
self.vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
...
word_count = 0
for text in training_texts:
    tokens = text.lower().split()
    for token in tokens:
        if token not in self.vocab:
            word_count += 1
            self.vocab[token] = word_count
```
**Mechanism of Failure**:
- `self.vocab` begins with length 4 (indices 0, 1, 2, 3).
- `word_count` is initialized to `0`.
- The first new token sets `word_count = 1`, assigning `self.vocab[first_token] = 1`.
- This creates an index collision with `<UNK>` (index 1).
- The second token is assigned index 2, colliding with `<SOS>` (index 2).
- The third token is assigned index 3, colliding with `<EOS>` (index 3).
- In `self.inverse_vocab = {v: k for k, v in self.vocab.items()}`, `<UNK>`, `<SOS>`, and `<EOS>` are overwritten and lost.

### 6.2 Preprocessing OOD Combinations Disconnect
In `preprocessing/preprocessing_config.json`:
```json
"ood_blocked_combinations": [[["color", "blue"], ["proportion", "exaggerated"]]]
```
In `data/meta/cartoon_image_attributes.csv`:
The attribute columns are `face_color`, `hair_color`, `eye_color`, `glasses`, etc. None of the columns are named `"color"` or `"proportion"`.
In `splitter.py`:
`blocked_set.issubset(current_comb)` never matches any image, resulting in zero images being classified as OOD (`ood_indices = []`).

---

## 7. Concrete Code Remediation Recommendations

Below are the exact code modifications recommended to resolve the identified architectural and numerical defects:

### Fix 1: Correct Mask Value in `models/transformer.py`
```python
# BEFORE (Line 139):
if mask is not None:
    attention_scores = attention_scores.masked_fill(mask == 0, -1e-9)

# AFTER:
if mask is not None:
    attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
```

### Fix 2: Add Attention Masking to `SpatialCrossAttention` in `models/unet_parts.py`
```python
# BEFORE:
def forward(self, x, context):
    ...
    out = F.scaled_dot_product_attention(
        q, k, v, 
        dropout_p=self.to_out[1].p if self.training else 0.0
    )

# AFTER:
def forward(self, x, context, mask=None):
    ...
    # mask shape expected: [B, 1, 1, Seq_Len]
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask,
        dropout_p=self.to_out[1].p if self.training else 0.0
    )
```

### Fix 3: Fix Index Initializer in `preprocessing/tokenizer.py`
```python
# BEFORE (Line 20):
word_count = 0

# AFTER:
word_count = max(self.vocab.values())
```

### Fix 4: Vector Affine Parameters in `models/transformer.py`
```python
# BEFORE:
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

# AFTER:
class LayerNormalization(nn.Module):
    def __init__(self, d_model: int = 128, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
```

---

## 8. Final Conclusion

1. **From-Scratch Mandate**: The core generative model (U-Net, custom Transformer text encoder, DDPM scheduler) is built 100% from scratch without forbidden pretrained checkpoints or Hugging Face diffusers pipelines.
2. **Parameter Budget**: The model architecture totals **~8.56M parameters** (U-Net: 8.14M, Text Encoder: 0.42M), fully satisfying the assignment's "Tiny" envelope.
3. **Diffusion Implementation**: The forward noising process, cosine variance schedule, reverse mean calculation, and MSE epsilon-prediction loss are mathematically correct and conform to DDPM theory.
4. **Deficiencies Requiring Attention**:
   - The `-1e-9` mask bug in `transformer.py` breaks padding suppression.
   - The lack of attention mask propagation in `SpatialCrossAttention` dilutes text conditioning onto `<PAD>` tokens.
   - `AvatarTokenizer.fit` clobbers special tokens `<UNK>, <SOS>, <EOS>` due to `word_count = 0`.
   - MaxPool downsampling and high-resolution cross-attention ($64 \times 64$) represent sub-optimal architectural choices relative to SOTA diffusion engineering.
