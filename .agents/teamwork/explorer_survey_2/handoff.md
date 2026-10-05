# Handoff Report: Neural Network Architecture & Diffusion Math Survey

**Agent**: Survey Explorer 2 (Architecture & Math)  
**Working Directory**: `c:\Users\Admin\Desktop\avatar diffusion\.agents\teamwork\explorer_survey_2`  
**Date**: October 5, 2026  
**Type**: Hard Handoff (Investigation & Survey Complete)

---

## 1. Observation

Direct forensic observations from the target codebase files:

### 1.1 `models/transformer.py`
- **Lines 138–148 (`MultiHeadAttentionBlock.attention`)**:
  ```python
  if mask is not None:
      # float("-inf") è lo standard PyTorch: softmax(float("-inf")) == 0.0 senza overflow
      attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

  # applicata lungo l'ultima dimensione che corrisponde al key_len
  attention_scores = attention_scores.softmax(dim=-1)

  # Gestione di casi limite con intere righe mascherate (tutti -inf producono NaN in softmax)
  if torch.isnan(attention_scores).any():
      attention_scores = torch.nan_to_num(attention_scores, nan=0.0)
  ```
  The attention masking uses `float("-inf")` (not `-1e-9`) and includes `torch.nan_to_num(..., nan=0.0)`. Matches Blueprint 1.3 line-by-line.
- **Lines 66–67 (`LayerNormalization`)**:
  ```python
  self.alpha = nn.Parameter(torch.ones(1))
  self.bias = nn.Parameter(torch.zeros(1))
  ```
  Both affine parameters are 1D scalar tensors of shape `torch.Size([1])` across all channels (`DEF-15`). Matches Section 3.2 parameter breakdown table ($2 \times 2 = 4$ params per block).

### 1.2 `models/unet_parts.py`
- **Lines 237–279 (`SpatialCrossAttention.forward`)**:
  ```python
  def forward(self, x, context, mask=None):
      ...
      attn_mask = None
      if mask is not None:
          if mask.ndim == 2:
              attn_mask = mask.unsqueeze(1).unsqueeze(2)
          elif mask.ndim == 3:
              attn_mask = mask.unsqueeze(1)
          elif mask.ndim == 4:
              attn_mask = mask
          else:
              attn_mask = mask.view(b, 1, 1, -1)

          if attn_mask.dtype != torch.bool:
              attn_mask = (attn_mask != 0)

      out = F.scaled_dot_product_attention(
          q, k, v,
          attn_mask=attn_mask,
          dropout_p=self.to_out[1].p if self.training else 0.0
      )
  ```
  Accepts `mask=None`, adapts masks across ranks 2, 3, 4, and arbitrary dimensions, casts to boolean, and passes to `F.scaled_dot_product_attention`. Matches Blueprint 1.4 line-by-line.

### 1.3 `models/unet.py`
- **Lines 50–77 (`Unet.forward`)**:
  ```python
  def forward(self, x, time, context, mask=None):
      t = self.time_mlp(time)
      skip1 = self.attn_inc(self.inc(x, t), context, mask=mask)
      skip2 = self.attn_down1(self.down1(skip1, t), context, mask=mask)
      ...
      skip3 = self.attn_down2(x_down2, context, mask=mask)
      ...
      bott = self.attn_bott1(bott, context, mask=mask)
      ...
      x = self.attn_up1(x, context, mask=mask)
      x = self.attn_up2(x, context, mask=mask)
      out = self.out(x)
      return out
  ```
  Accepts `mask=None` and routes `mask=mask` to all 6 cross-attention blocks. Matches Blueprint 1.4 line-by-line.

### 1.4 `models/diffusion.py`
- **Lines 49–51 (`DiffusionScheduler.__init__`)**:
  ```python
  self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alpha_bars[:-1]])
  self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alpha_bars)).to(self.device)
  self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bars)).to(self.device)
  ```
  Posterior mean coefficients and cumulative product previous values are precalculated. Matches Blueprint 2.1 lines 1026–1028.
- **Line 75 (`DiffusionReverseProcess.sample`)**:
  ```python
  def sample(self, model, x, t, context=None, uncond_context=None, guidance_scale=3.0, noise_free=False, clip_denoised=True):
  ```
  Does NOT accept `mask=None` or `uncond_mask=None`.
- **Line 103 & Line 113 (`DiffusionReverseProcess.sample`)**:
  ```python
  all_noise = model(x_input, t_input, context=context_input)
  predicted_noise = model(x, t, context=context)
  ```
  Does NOT pass `mask` to `model`.
- **Lines 119–128 (`DiffusionReverseProcess.sample`)**:
  ```python
  pred_x0 = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
  if clip_denoised:
      pred_x0.clamp_(-1.0, 1.0)
  coef1 = self.posterior_mean_coef1[t].to(x.device)[:, None, None, None]
  coef2 = self.posterior_mean_coef2[t].to(x.device)[:, None, None, None]
  mean = coef1 * pred_x0 + coef2 * x
  ```
  `pred_x0` calculation, clamping, and posterior mean are present.
- **Line 143 (`DiffusionReverseProcess.sample`)**:
  ```python
  sigma_t = torch.sqrt(torch.clamp(beta_t, min=1e-20))
  ```
  `beta_t` is NOT DEFINED anywhere in `sample()`. In Blueprint 2.1, line 1060 defines:
  `beta_t = self.betas[t].to(x.device)[:, None, None, None]`.

### 1.5 External Call Sites
- **`evaluate.py:100-105`**:
  ```python
  x = reverse_process.sample(
      model=unet, x=x, t=t,
      context=cond_ctx, uncond_context=uncond_ctx,
      mask=mask, uncond_mask=uncond_mask,
      guidance_scale=3.5, clip_denoised=True
  )
  ```
  Passes `mask` and `uncond_mask` which triggers `TypeError` with current `models/diffusion.py`.
- **`inference.py:81-89`**:
  ```python
  x = self.reverse_process.sample(
      model=self.unet, x=x, t=t,
      context=context, uncond_context=uncond_context,
      guidance_scale=guidance_scale, noise_free=noise_free
  )
  ```
  Executes reverse loop for `t_step in reversed(range(1000))`; on step $t=999$, triggers fatal `NameError: name 'beta_t' is not defined`.

---

## 2. Logic Chain

1. **Transformer Attention (`DEF-03`)**:
   - Observation 1.1 shows `models/transformer.py:140` utilizes `float("-inf")` with NaN protection.
   - Therefore, DEF-03 has already been remediated and matches Blueprint 1.3 completely.

2. **Transformer Normalization (`DEF-15`)**:
   - Observation 1.1 shows `models/transformer.py:66-67` uses scalar parameters `torch.ones(1)`.
   - The defect audit report categorized DEF-15 as `LOW` severity and did not provide a replacement blueprint in Section 10 to protect the strict ~8.56M parameter budget.
   - Therefore, scalar parameterization is compliant with project instructions.

3. **U-Net Cross-Attention & Propagation (`DEF-04`)**:
   - Observation 1.2 shows `SpatialCrossAttention.forward` handles rank-adaptive masks (2D to 4D) and boolean casting for FlashAttention.
   - Observation 1.3 shows `Unet.forward` exposes `mask=None` and routes it to all 6 cross-attention blocks.
   - Therefore, DEF-04 is completely remediated across both U-Net files and matches Blueprint 1.4 line-by-line.

4. **Diffusion Reverse Sampling (`DEF-17`)**:
   - Observation 1.4 verifies that `DiffusionScheduler` precomputes `posterior_mean_coef1/2` and `alphas_cumprod_prev`, and `sample()` applies `pred_x0.clamp_(-1.0, 1.0)`.
   - However, Observation 1.4 reveals that `beta_t` was omitted prior to line 143 (`sigma_t = torch.sqrt(torch.clamp(beta_t, min=1e-20))`).
   - Any execution with $t > 0$ and `noise_free == False` evaluates line 143 and immediately raises `NameError: name 'beta_t' is not defined`.
   - Furthermore, Observation 1.4 shows `sample()` does not accept `mask` or `uncond_mask`, causing Observation 1.5 (`evaluate.py`) to raise `TypeError: sample() got an unexpected keyword argument 'mask'`.
   - Observation 1.4 also shows that `model` is called without `mask` during sampling, silently disabling cross-attention masking during generation.
   - Therefore, DEF-17 is NOT fully resolved and contains fatal breaking defects in `models/diffusion.py`.

---

## 3. Caveats

1. **Read-Only Explorer Constraints**: As an Explorer, no edits were made to source files in `models/`. The fix must be applied by the remediation agent.
2. **GPU Execution**: No active CUDA GPU execution was performed during this survey. Code paths and error traces were derived via static AST and control flow analysis.
3. **No other files in `models/`**: The `models/` directory contains only `transformer.py`, `unet_parts.py`, `unet.py`, and `diffusion.py`.

---

## 4. Conclusion

- **`models/transformer.py`**: **PASS**. 100% compliant with Blueprint 1.3. No action required.
- **`models/unet_parts.py`**: **PASS**. 100% compliant with Blueprint 1.4. No action required.
- **`models/unet.py`**: **PASS**. 100% compliant with Blueprint 1.4. No action required.
- **`models/diffusion.py`**: **ACTION REQUIRED**. Blueprint 2.1 is partially implemented and currently non-functional:
  - Must define `beta_t = self.betas[t].to(x.device)[:, None, None, None]` before computing `sigma_t`.
  - Must add `mask=None, uncond_mask=None` to `sample()` signature.
  - Must concatenate `mask_input` under CFG and pass `mask` to `model(...)`.
  - Must use `self.sqrt_alpha_bars` and `self.sqrt_one_minus_alpha_bars` buffer indexing.

---

## 5. Verification Method

To independently verify these findings, perform the following code inspections and execution checks:

### 5.1 Static Verification
1. Inspect `models/diffusion.py` line 143: verify `beta_t` is referenced without preceding definition in `sample()`.
2. Inspect `models/diffusion.py` line 75: verify `mask` is absent from `sample()` parameter list.
3. Inspect `evaluate.py` line 103: verify `mask=mask, uncond_mask=uncond_mask` is passed to `sample()`.

### 5.2 Python Execution Test
Execute the following verification script in Python (e.g. via an automated test or interactive python shell):

```python
import torch
from models.unet import Unet
from models.diffusion import DiffusionReverseProcess

device = "cpu"
unet = Unet(in_channels=3, out_channels=3, base_channels=64, context_dim=128).to(device)
rev = DiffusionReverseProcess(num_time_steps=1000, device=device)

x = torch.randn(1, 3, 64, 64)
t = torch.tensor([500], dtype=torch.long)
ctx = torch.randn(1, 20, 128)
mask = torch.ones(1, 20, dtype=torch.bool)

# Test 1: Verify TypeError with mask argument
try:
    rev.sample(unet, x, t, context=ctx, mask=mask)
except TypeError as e:
    print(f"VERIFIED TypeError: {e}")

# Test 2: Verify NameError with beta_t
try:
    rev.sample(unet, x, t, context=ctx)
except NameError as e:
    print(f"VERIFIED NameError: {e}")
```

### Invalidation Conditions
If `models/diffusion.py` is updated with Blueprint 2.1, both Test 1 and Test 2 will pass cleanly without raising exceptions.
