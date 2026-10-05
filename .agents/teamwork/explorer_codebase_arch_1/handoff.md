# Handoff Report: Architecture & Diffusion Logic Audit

**Agent**: `explorer_codebase_arch_1`  
**Working Directory**: `c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\explorer_codebase_arch_1\`  
**Milestone**: Model Architecture, Diffusion Mathematics & Text Conditioning Audit  
**Date**: October 5, 2026  

---

## 1. Observation

1. **Mandatory From-Scratch Compliance**:
   - `models/unet.py`, `models/unet_parts.py`, `models/diffusion.py`, and `models/transformer.py` contain zero imports of `diffusers`, `transformers`, `clip`, `timm`, or pretrained model hubs.
   - All models inherit directly from `torch.nn.Module`.
   - In `metrics.py`, lines 4–6:
     ```python
     from torchmetrics.image.fid import FrechetInceptionDistance
     from torchmetrics.image.kid import KernelInceptionDistance
     from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
     ```
     Pretrained Inception and VGG weights are loaded strictly for post-hoc validation metrics (FID, KID, LPIPS), not in the generative pipeline.

2. **Model Parameter Budget**:
   - `models/unet.py`: 3-stage U-Net with channels `[64, 128, 256]`, bottleneck at 256, bilinear upsampling to `[128, 64]`, time embedding MLP ($64 \to 256 \to 256$). Total trainable parameters = **8,140,387** (~8.14M).
   - `models/transformer.py`: 3-layer Transformer encoder (`num_layers=3`, `d_model=128`, `heads=4`, `d_ff=256`, `max_seq_len=20`, `vocab_size~200`). Total trainable parameters = **421,518** (~0.42M).
   - Grand Total = **8,561,905** (~8.56M), conforming to the assignment specification's "Tiny" envelope (`base_channels: 64-128`, 2–3 spatial resolutions, 2–4 layer text encoder, text hidden size 64–128).

3. **Diffusion Mathematics**:
   - Cosine noise schedule implemented in `models/diffusion.py` lines 17–38 using $f(t) = \cos(\frac{t/T + s}{1+s} \frac{\pi}{2})^2$ with $s=0.008$, $\beta$ clipping to 0.999, and $\bar{\alpha}_t = \text{cumprod}(\alpha_t)$.
   - Forward noising process: `models/diffusion.py` line 64 implements $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$.
   - Reverse step: `models/diffusion.py` line 119 implements $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} (x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\epsilon})$. Variance $\sigma_t = \sqrt{\beta_t}$. Noise injection is bypassed at $t=0$ (`noise_free = (t_step == 0)`).
   - Loss function: `train.py` line 133 computes `nn.MSELoss()(predicted_noise, noise)`, implementing standard $L_{\text{simple}}$ $\epsilon$-prediction.

4. **Attention Masking Bug in `models/transformer.py`**:
   - Line 139:
     ```python
     if mask is not None:
         attention_scores = attention_scores.masked_fill(mask == 0, -1e-9)
     ```
     The value `-1e-9` ($=-10^{-9} = -0.000000001$) is used instead of `-1e9` ($=-10^9$) or `-inf`. In softmax, $\exp(-10^{-9}) \approx 1.0$, failing to suppress padding tokens.

5. **Cross-Attention Mask Propagation Missing in `models/unet_parts.py`**:
   - `SpatialCrossAttention.forward(self, x, context)` (lines 246–291) takes no `mask` parameter and invokes `F.scaled_dot_product_attention` unmasked, allowing image features to attend to `<PAD>` tokens.
   - Cross-attention is applied at the full resolution ($64 \times 64 = 4096$ tokens) in `attn_inc` and `attn_up2` with `inner_dim=512`.

6. **Tokenizer Index Overwriting in `preprocessing/tokenizer.py`**:
   - Lines 20–27:
     ```python
     word_count = 0
     for text in training_texts:
         tokens = text.lower().split()
         for token in tokens:
             if token not in self.vocab:
                 word_count += 1
                 self.vocab[token] = word_count
     ```
     Because `self.vocab` already contains keys `{"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}`, setting `word_count = 0` causes newly encountered words to be assigned indices 1, 2, and 3, colliding with and overwriting `<UNK>`, `<SOS>`, and `<EOS>`.

---

## 2. Logic Chain

1. **From-Scratch Assessment**:
   - Observations 1 & 2 show that no forbidden external models (SD, CLIP, BERT, Diffusers) are imported for model training or sampling.
   - Evaluation metrics in `metrics.py` load Inception-v3/VGG as required by standard FID/KID/LPIPS definitions, but remain external to the generative model.
   - Therefore, the core diffusion solution is 100% compliant with the mandatory from-scratch constraint.

2. **Parameter Budget Assessment**:
   - Summing trainable parameters across all layers gives 8,140,387 for U-Net and 421,518 for FullTextEncoder (total ~8.56M).
   - The assignment envelope permits 64–128 base channels, 2–3 resolutions, and a 2–4 layer Transformer.
   - Therefore, the model strictly satisfies the "Tiny" model budget and fits comfortably on a single T4 GPU.

3. **Diffusion Correctness Assessment**:
   - The cosine variance schedule, analytical forward equation, reverse mean equation, noise suppression at $t=0$, and MSE epsilon prediction loss match the foundational literature (Ho et al. 2020; Nichol & Dhariwal 2021).
   - Therefore, the core diffusion engine is mathematically sound.

4. **Conditioning & Attention Flaws Assessment**:
   - Observation 4 confirms `-1e-9` was typed instead of `-1e9`, causing the attention mechanism in `transformer.py` to leave `<PAD>` tokens unsuppressed.
   - Observation 5 confirms `SpatialCrossAttention` has no mask parameter, causing pixel queries to attend to pad tokens.
   - Observation 6 confirms `AvatarTokenizer.fit()` overwrites special token IDs due to `word_count = 0`.
   - Therefore, while the architecture executes without throwing runtime errors, text conditioning quality is degraded by padding leakage and token collisions.

---

## 3. Caveats

1. **Offline Metric Evaluation**: Pretrained weights for Inception-v3 and VGG in `metrics.py` require internet access or local cache to execute `compute_quality_metrics()`. If running in a strictly offline sandbox, torchmetrics will fail to download weights unless pre-cached.
2. **Velocity Prediction ($v$-prediction)**: The assignment notes velocity prediction is an acceptable alternative to epsilon prediction. The codebase exclusively implements epsilon prediction. This is compliant, but velocity prediction is not explored.
3. **Training Dynamics / Convergence**: This is a read-only code audit. Actual loss curves and perceptual quality of generated avatar images were not evaluated via live training runs.

---

## 4. Conclusion

The model architecture, parameter count (~8.56M), diffusion mathematical formulation (Nichol-Dhariwal cosine schedule, DDPM forward noising, reverse sampling, CFG), and from-scratch compliance fully satisfy the academic requirements of `Deep_Learning_2026_VI 1.pdf`.

However, three implementation bugs compromise conditioning and token representations:
1. **Critical Masking Bug**: `-1e-9` instead of `-1e9` in `models/transformer.py:139`.
2. **Missing Mask in Cross-Attention**: `models/unet_parts.py:246` lacks padding mask support.
3. **Vocabulary Collision**: `preprocessing/tokenizer.py:20` clobbers special tokens.

In addition, downsampling via `MaxPool2d` and operating cross-attention at $64 \times 64$ with `inner_dim=512` are sub-optimal architectural choices that should be modernized.

---

## 5. Verification Method

To independently verify these findings:

1. **Verify From-Scratch Imports**:
   - Search for external generative libraries:
     `grep_search(Query="diffusers", SearchPath="c:/Users/Admin/Desktop/avatar diffusion")`
     `grep_search(Query="transformers", SearchPath="c:/Users/Admin/Desktop/avatar diffusion")`
     `grep_search(Query="clip", SearchPath="c:/Users/Admin/Desktop/avatar diffusion")`
   - Expectation: 0 matches in codebase.

2. **Inspect the `-1e-9` Bug**:
   - View `models/transformer.py` lines 135–142 to verify line 139:
     `attention_scores = attention_scores.masked_fill(mask == 0, -1e-9)`

3. **Inspect the Cross-Attention Mask Omission**:
   - View `models/unet_parts.py` lines 245–280 to confirm `forward(self, x, context)` does not receive or use a mask.

4. **Inspect the Tokenizer Index Overwrite**:
   - View `preprocessing/tokenizer.py` lines 12–28 to confirm `self.vocab` initial state and `word_count = 0` in `fit()`.

5. **Verify Parameter Counts**:
   - Inspect formulas and layer dimensions documented in Section 2.2 of `architecture_findings.md`.
