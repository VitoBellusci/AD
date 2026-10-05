# Survey Explorer 2: Architecture & Diffusion Math Investigation Report

**Audit Scope**: Neural network architecture and diffusion math in `models/` (`transformer.py`, `unet_parts.py`, `unet.py`, `diffusion.py`)  
**Target Specifications**: Defects DEF-03, DEF-04, DEF-15, DEF-16, DEF-17; Blueprints 1.3, 1.4, 2.1 in `audit_report.md`  
**Date**: October 5, 2026  
**Status**: Investigation Complete (Read-Only)

---

## 1. Executive Summary

A comprehensive forensic inspection of the neural network architecture and diffusion mathematics in `models/` reveals that while the Transformer text encoder and U-Net have already been successfully updated to match Blueprints 1.3 and 1.4, **the diffusion reverse process in `models/diffusion.py` suffers from an incomplete implementation of Blueprint 2.1 containing a fatal `NameError` crash (`beta_t` is undefined) and missing mask arguments that trigger `TypeError` when called by `evaluate.py`.**

### Summary Matrix

| Component / Defect | Location | Blueprint Target | Current Implementation Status | Verdict |
|---|---|---|---|---|
| **DEF-03: Attention Mask Underflow** | `models/transformer.py:138-148` | Blueprint 1.3 | Replaced `-1e-9` with `float("-inf")`; added `torch.nan_to_num(..., nan=0.0)` for all-masked row handling. | **100% COMPLIANT** (Resolved) |
| **DEF-15: Scalar LayerNorm Parameters** | `models/transformer.py:66-67` | N/A (Low Severity) | Uses `self.alpha = nn.Parameter(torch.ones(1))` and `self.bias = nn.Parameter(torch.zeros(1))`. | **AS DESIGNED** (Preserves 8.56M parameter budget; no blueprint assigned) |
| **DEF-04: Cross-Attention Mask Omission** | `models/unet_parts.py:237-279` | Blueprint 1.4 | `SpatialCrossAttention.forward` accepts `mask`, applies rank-adaptive reshaping (2D/3D/4D/ND), casts to boolean, and feeds `attn_mask` into `F.scaled_dot_product_attention`. | **100% COMPLIANT** (Resolved) |
| **DEF-04: U-Net Mask Wiring** | `models/unet.py:50-77` | Blueprint 1.4 | `Unet.forward` signature includes `mask=None` and routes `mask=mask` to all 6 cross-attention blocks (`attn_inc`, `attn_down1`, `attn_down2`, `attn_bott1`, `attn_up1`, `attn_up2`). | **100% COMPLIANT** (Resolved) |
| **DEF-16: Architectural Critique** | `models/unet_parts.py:103, 237` | N/A (Low Severity) | Uses `MaxPool2d(2)` and $64\times 64$ cross-attention. | **AS DESIGNED** (Preserved to maintain parameter constraints; no blueprint assigned) |
| **DEF-17: Reverse Process Range Clamping & DDPM Math** | `models/diffusion.py:49-51, 75-145` | Blueprint 2.1 | Precomputes `alphas_cumprod_prev`, `posterior_mean_coef1`, `posterior_mean_coef2`; implements `pred_x0` calculation, `clamp_(-1.0, 1.0)`, and posterior mean. **HOWEVER: `beta_t` is never defined before line 143 (`NameError`), `mask`/`uncond_mask` arguments are missing (`TypeError`), and mask propagation is omitted in `sample()`.** | **CRITICAL REGRESSION / BROKEN** (Needs Remediation) |

---

## 2. Transformer Architecture (`models/transformer.py`)

### 2.1 MultiHeadAttentionBlock Mask Handling (`DEF-03` / Blueprint 1.3)

In `models/transformer.py`, lines 123–155 implement the attention calculation:

```python
# models/transformer.py:123-155
@staticmethod
def attention(query, key, value, mask, dropout):
    d_k = query.shape[-1]
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # float("-inf") è lo standard PyTorch: softmax(float("-inf")) == 0.0 senza overflow
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

    # applicata lungo l'ultima dimensione che corrisponde al key_len
    attention_scores = attention_scores.softmax(dim=-1)

    # Gestione di casi limite con intere righe mascherate (tutti -inf producono NaN in softmax)
    if torch.isnan(attention_scores).any():
        attention_scores = torch.nan_to_num(attention_scores, nan=0.0)

    if dropout is not None:
        attention_scores = dropout(attention_scores)

    return attention_scores @ value, attention_scores
```

#### Forensic Assessment
1. **Mask Value**: The original underflow bug (where `mask == 0` was filled with `-1e-9`, yielding $\exp(-10^{-9}) \approx 0.999999999 \approx 1.0$ and failing to suppress `<PAD>` tokens) has been completely eradicated. The code now utilizes `float("-inf")`.
2. **AMP & Numerical Stability**: `float("-inf")` is fully compatible with FP32, FP16 (AMP), and BF16 in PyTorch, mapping directly to softmax probability $0.0$.
3. **All-Masked Edge Cases**: Lines 145–147 implement `torch.nan_to_num(attention_scores, nan=0.0)`, which handles edge cases where an entire sequence row is masked (e.g. all-pad queries in unconditional contexts), preventing `NaN` gradient propagation.
4. **Blueprint 1.3 Match**: Matches lines 887–907 of `audit_report.md` character-for-character.

### 2.2 LayerNormalization Parameterization (`DEF-15`)

In `models/transformer.py`, lines 55–76 define custom `LayerNormalization`:

```python
# models/transformer.py:55-76
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
```

#### Forensic Assessment
1. **Parameter Shape**: `self.alpha` and `self.bias` are 1D scalar tensors of shape `torch.Size([1])`, rather than feature-channel vectors of shape `torch.Size([d_model])` (`[128]`).
2. **Expressive Impact**: All 128 embedding dimensions share a single scalar gain $\alpha$ and shift $\beta$. This is less expressive than standard PyTorch `nn.LayerNorm(d_model)`.
3. **Assignment Compliance & Parameter Count**:
   - In Section 3.2 of `audit_report.md` (lines 201–205), the parameter count accounting explicitly calculates:
     `Custom LayerNormalization (\alpha \in \mathbb{R}^1, \beta \in \mathbb{R}^1) = 2 \times 2 scalar parameters = 4 per block`.
     Total text encoder parameters: exactly 421,518.
     Total model parameters: exactly 8,561,905 (~8.56M).
   - Upgrading `self.alpha` to `torch.ones(d_model)` would add $2 \times 128 \times 7 = 1,792$ parameters, change the parameter count, and require refactoring `LayerNormalization.__init__` to accept `d_model` across `ResidualConnection:198` and `Encoder:253`.
   - `audit_report.md` Section 9 ranked DEF-15 as `LOW` severity and did NOT include a blueprint in Section 10. `ORIGINAL_REQUEST.md` mandates: *"Do not rewrite components that are not explicitly targeted by a defect blueprint. Maintain the 'from-scratch' constraints and the 'Tiny' parameter budget."*
   - **Conclusion**: Retaining scalar parameters in `models/transformer.py` is intentional, safe, and fully compliant.

---

## 3. U-Net Architecture (`models/unet_parts.py` & `models/unet.py`)

### 3.1 `SpatialCrossAttention` in `models/unet_parts.py` (`DEF-04` / Blueprint 1.4)

In `models/unet_parts.py`, lines 218–279 define `SpatialCrossAttention`:

```python
# models/unet_parts.py:218-279
class SpatialCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dropout=0.0):
        super().__init__()
        self.heads = heads
        inner_dim = query_dim * heads

        self.norm = nn.GroupNorm(num_groups=32, num_channels=query_dim, eps=1e-6, affine=True)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context, mask=None):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.view(b, c, -1).permute(0, 2, 1) # [B, H*W, C]

        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)

        # Reshape per Multi-Head Attention: [B, Heads, Seq_Len, Dim_Head]
        q = q.view(b, -1, self.heads, q.shape[-1] // self.heads).transpose(1, 2)
        k = k.view(b, -1, self.heads, k.shape[-1] // self.heads).transpose(1, 2)
        v = v.view(b, -1, self.heads, v.shape[-1] // self.heads).transpose(1, 2)

        # Gestione dinamica del rango della maschera per prevenire esplosione dimensionale
        attn_mask = None
        if mask is not None:
            if mask.ndim == 2:
                # [B, Seq_Len] -> [B, 1, 1, Seq_Len]
                attn_mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.ndim == 3:
                # [B, 1, Seq_Len] -> [B, 1, 1, Seq_Len]
                attn_mask = mask.unsqueeze(1)
            elif mask.ndim == 4:
                # Già [B, 1, 1, Seq_Len] o compatibile broadcast
                attn_mask = mask
            else:
                attn_mask = mask.view(b, 1, 1, -1)

            # scaled_dot_product_attention supporta maschere booleane (True = attend, False = ignore)
            if attn_mask.dtype != torch.bool:
                attn_mask = (attn_mask != 0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.to_out[1].p if self.training else 0.0
        )

        out = out.transpose(1, 2).reshape(b, -1, out.shape[-1] * self.heads)
        out = self.to_out(out)
        out = out.permute(0, 2, 1).view(b, c, h, w)
        return x + out
```

#### Forensic Assessment
1. **Signature**: Accepts `mask=None` as required.
2. **Rank-Adaptive Mask Handling**:
   - 2D mask `[B, Seq_Len]` (from `train.py` or simple tokenizer outputs) is expanded via `.unsqueeze(1).unsqueeze(2)` to `[B, 1, 1, Seq_Len]`.
   - 3D mask `[B, 1, Seq_Len]` is expanded via `.unsqueeze(1)` to `[B, 1, 1, Seq_Len]`.
   - 4D mask `[B, 1, 1, Seq_Len]` (constructed by `train.py:106`: `mask = (text_tokens != pad_token_id).unsqueeze(1).unsqueeze(2)`) is passed through without invalid 6D expansion.
   - Any other rank is safely collapsed via `.view(b, 1, 1, -1)`.
3. **Boolean Casting**: Lines 267–268 ensure `attn_mask` has `torch.bool` dtype (`attn_mask != 0`), adhering to PyTorch's `F.scaled_dot_product_attention` convention where `True` indicates keys that should be attended to and `False` represents masked tokens.
4. **Blueprint 1.4 Match**: Matches lines 916–973 of `audit_report.md` exactly.

### 3.2 Mask Propagation in `models/unet.py` (`DEF-04` / Blueprint 1.4)

In `models/unet.py`, lines 50–77 define `Unet.forward`:

```python
# models/unet.py:50-77
def forward(self, x, time, context, mask=None):
    t = self.time_mlp(time)

    # ENCODER con propagazione esplicita della maschera
    skip1 = self.attn_inc(self.inc(x, t), context, mask=mask)
    skip2 = self.attn_down1(self.down1(skip1, t), context, mask=mask)
    
    x_down2 = self.down2(skip2, t)
    x_down2 = self.self_attn_down2(x_down2)
    skip3 = self.attn_down2(x_down2, context, mask=mask)

    # BOTTLENECK
    bott = self.bott1(skip3, t)
    bott = self.self_attn_bott(bott)
    bott = self.attn_bott1(bott, context, mask=mask)
    bott = self.bott2(bott, t)

    # DECODER
    x = self.up1(bott, skip2, t)
    x = self.self_attn_up1(x)
    x = self.attn_up1(x, context, mask=mask)
    
    x = self.up2(x, skip1, t)
    x = self.attn_up2(x, context, mask=mask)

    # OUTPUT
    out = self.out(x)
    return out
```

#### Forensic Assessment
1. **Mask Wiring**: `mask=mask` is explicitly passed to all six spatial cross-attention layers:
   - `self.attn_inc` ($64\times 64$)
   - `self.attn_down1` ($32\times 32$)
   - `self.attn_down2` ($16\times 16$)
   - `self.attn_bott1` ($16\times 16$, bottleneck)
   - `self.attn_up1` ($32\times 32$)
   - `self.attn_up2` ($64\times 64$)
2. **Self-Attention Isolation**: `self_attn_down2`, `self_attn_bott`, and `self_attn_up1` receive only visual feature representations `x`, correctly isolated from text conditioning.
3. **Blueprint 1.4 Match**: Matches lines 980–1008 of `audit_report.md` exactly.

---

## 4. Diffusion Mathematics & Scheduling (`models/diffusion.py`)

### 4.1 `DiffusionScheduler` Verification

In `models/diffusion.py`, lines 11–52 initialize `DiffusionScheduler`:

```python
# models/diffusion.py:11-52
def __init__(self, num_time_steps=1000, s=0.008, device="cpu"):
    self.num_time_steps = num_time_steps
    self.device = torch.device(device)

    # Nichol-Dhariwal cosine schedule
    steps = torch.arange(num_time_steps + 1, dtype=torch.float32, device=self.device)
    f_t = torch.cos(((steps / num_time_steps + s) / (1 + s)) * (math.pi / 2))**2
    ab_full = f_t / f_t[0]    

    self.alpha_bars = ab_full[1:].to(self.device)
    ab_prev = ab_full[: -1].to(self.device)
    alphas = self.alpha_bars / ab_prev
    betas = 1 - alphas
    self.betas = torch.clamp(betas, max=0.999).to(self.device)
    self.alphas = 1 - self.betas
    self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    # Forward process buffers
    self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
    self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

    # Reverse process buffers
    self.inv_sqrt_alphas = 1 / torch.sqrt(self.alphas)
    self.beta_over_sqrt_one_minus_alpha_bar = self.betas / torch.sqrt(1 - self.alpha_bars)

    # Pre-calcolo per la formulazione corretta di reverse sampling con clipping di x_0 (Ho et al. 2020 Eq. 12)
    self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alpha_bars[:-1]])
    self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alpha_bars)).to(self.device)
    self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bars)).to(self.device)
```

#### Forensic Assessment
1. **Mathematical Schedule**: Strictly complies with Nichol & Dhariwal (2021) Cosine Schedule with offset $s=0.008$.
2. **Singularity Prevention**: Clamping $\beta_t \le 0.999$ and recomputing cumulative product $\bar{\alpha}_t$ ensures exact numerical consistency.
3. **Posterior Mean Coefficients**: `alphas_cumprod_prev`, `posterior_mean_coef1`, and `posterior_mean_coef2` are precomputed in `__init__`, exactly satisfying Blueprint 2.1 lines 1026–1028.

---

### 4.2 Defect Analysis in `DiffusionReverseProcess.sample()`

In `models/diffusion.py`, lines 75–145 implement `DiffusionReverseProcess.sample`:

```python
# models/diffusion.py:75-145
class DiffusionReverseProcess(DiffusionScheduler):
    def sample(self, model, x, t, context=None, uncond_context=None, guidance_scale=3.0, noise_free=False, clip_denoised=True):
        if guidance_scale > 1.0 and context is not None and uncond_context is not None:
            x_input = torch.cat([x, x], dim=0)
            t_input = torch.cat([t, t], dim=0)
            context_input = torch.cat([context, uncond_context], dim=0)

            all_noise = model(x_input, t_input, context=context_input)
            eps_cond, eps_uncond = torch.chunk(all_noise, 2, dim=0)
            predicted_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            predicted_noise = model(x, t, context=context)

        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - self.alpha_bars[t]).to(x.device)[:, None, None, None]
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bars[t]).to(x.device)[:, None, None, None]
        
        pred_x0 = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
        
        if clip_denoised:
            pred_x0.clamp_(-1.0, 1.0)
            
        coef1 = self.posterior_mean_coef1[t].to(x.device)[:, None, None, None]
        coef2 = self.posterior_mean_coef2[t].to(x.device)[:, None, None, None]
        mean = coef1 * pred_x0 + coef2 * x

        if (t == 0).all() or noise_free:
            return mean

        z = torch.randn_like(x)
        # CRITICAL BUG: beta_t is NEVER defined!
        sigma_t = torch.sqrt(torch.clamp(beta_t, min=1e-20))

        return mean + sigma_t * z
```

#### Detailed Forensic Analysis of Flaws in Current `sample()`:

#### Bug 1: Fatal `NameError: name 'beta_t' is not defined` (CRITICAL CRASH)
- **Mechanism**: On line 143:
  ```python
  sigma_t = torch.sqrt(torch.clamp(beta_t, min=1e-20))
  ```
  Variable `beta_t` is referenced, but **it was never initialized or assigned anywhere in `sample()`**.
- **Blueprint 2.1 Definition**: Blueprint 2.1 (line 1060) defines:
  ```python
  beta_t = self.betas[t].to(x.device)[:, None, None, None]
  ```
- **Consequence**: Whenever `sample()` is executed with $t > 0$ and `noise_free == False` (which applies to steps 999 down to 1 during 1000-step DDPM sampling), execution terminates with an unrecoverable `NameError: name 'beta_t' is not defined`.
- **Why this was not caught yet**: `inference.py` currently crashes prior to sampling due to hardcoded checkpoint paths (`DEF-08`), and `evaluate.py` was previously orphaned (`DEF-06`).

#### Bug 2: Missing `mask` and `uncond_mask` Parameters in Signature (`TypeError`)
- **Mechanism**: The signature is:
  ```python
  def sample(self, model, x, t, context=None, uncond_context=None, guidance_scale=3.0, noise_free=False, clip_denoised=True):
  ```
  Parameters `mask=None, uncond_mask=None` are completely missing.
- **Consequence**: In `evaluate.py` line 100–105:
  ```python
  x = reverse_process.sample(
      model=unet, x=x, t=t,
      context=cond_ctx, uncond_context=uncond_ctx,
      mask=mask, uncond_mask=uncond_mask,
      guidance_scale=3.5, clip_denoised=True
  )
  ```
  Calling `sample()` with `mask` immediately crashes with `TypeError: sample() got an unexpected keyword argument 'mask'`.

#### Bug 3: Mask Propagation Broken During Sampling (Silent Quality Degradation)
- **Mechanism**: In `sample()`, the model is invoked on line 103 without `mask`:
  ```python
  all_noise = model(x_input, t_input, context=context_input)
  ```
  And on line 113:
  ```python
  predicted_noise = model(x, t, context=context)
  ```
- **Consequence**: Even though Blueprint 1.4 modified `Unet.forward` to accept `mask`, `sample()` fails to construct `mask_input` and pass it to `model`. Consequently, during reverse sampling, the U-Net spatial cross-attention layers attend to `<PAD>` tokens unmasked, degrading conditioning fidelity.

#### Bug 4: Suboptimal Recomputation of Square Roots
- **Mechanism**: Lines 116–117 calculate:
  ```python
  sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - self.alpha_bars[t]).to(x.device)[:, None, None, None]
  sqrt_alpha_bar_t = torch.sqrt(self.alpha_bars[t]).to(x.device)[:, None, None, None]
  ```
- **Blueprint 2.1**: Directly slices the precomputed buffers:
  ```python
  sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].to(x.device)[:, None, None, None]
  sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].to(x.device)[:, None, None, None]
  ```

---

## 5. Line-by-Line Blueprint Comparison

### 5.1 Blueprint 1.3 vs `models/transformer.py`

| Blueprint 1.3 (audit_report.md:887-907) | Current Code (models/transformer.py:125-155) | Status |
|---|---|---|
| `@staticmethod` | `@staticmethod` | Identical |
| `def attention(query, key, value, mask, dropout):` | `def attention(query, key, value, mask, dropout):` | Identical |
| `d_k = query.shape[-1]` | `d_k = query.shape[-1]` | Identical |
| `attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)` | `attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)` | Identical |
| `if mask is not None:` | `if mask is not None:` | Identical |
| `attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))` | `attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))` | Identical |
| `attention_scores = attention_scores.softmax(dim=-1)` | `attention_scores = attention_scores.softmax(dim=-1)` | Identical |
| `if torch.isnan(attention_scores).any():` | `if torch.isnan(attention_scores).any():` | Identical |
| `attention_scores = torch.nan_to_num(attention_scores, nan=0.0)` | `attention_scores = torch.nan_to_num(attention_scores, nan=0.0)` | Identical |
| `if dropout is not None:` | `if dropout is not None:` | Identical |
| `attention_scores = dropout(attention_scores)` | `attention_scores = dropout(attention_scores)` | Identical |
| `return attention_scores @ value, attention_scores` | `return attention_scores @ value, attention_scores` | Identical |

**Verdict**: 100% Match.

---

### 5.2 Blueprint 1.4 vs `models/unet_parts.py` & `models/unet.py`

#### Part A: `SpatialCrossAttention` in `models/unet_parts.py`

| Blueprint 1.4 (audit_report.md:930-973) | Current Code (models/unet_parts.py:237-279) | Status |
|---|---|---|
| `def forward(self, x, context, mask=None):` | `def forward(self, x, context, mask=None):` | Identical |
| `b, c, h, w = x.shape` | `b, c, h, w = x.shape` | Identical |
| `x_norm = self.norm(x)` | `x_norm = self.norm(x)` | Identical |
| `x_flat = x_norm.view(b, c, -1).permute(0, 2, 1)` | `x_flat = x_norm.view(b, c, -1).permute(0, 2, 1)` | Identical |
| `q = self.to_q(x_flat)` | `q = self.to_q(x_flat)` | Identical |
| `k = self.to_k(context)` | `k = self.to_k(context)` | Identical |
| `v = self.to_v(context)` | `v = self.to_v(context)` | Identical |
| `q = q.view(b, -1, self.heads, q.shape[-1] // self.heads).transpose(1, 2)` | `q = q.view(b, -1, self.heads, q.shape[-1] // self.heads).transpose(1, 2)` | Identical |
| `k = k.view(b, -1, self.heads, k.shape[-1] // self.heads).transpose(1, 2)` | `k = k.view(b, -1, self.heads, k.shape[-1] // self.heads).transpose(1, 2)` | Identical |
| `v = v.view(b, -1, self.heads, v.shape[-1] // self.heads).transpose(1, 2)` | `v = v.view(b, -1, self.heads, v.shape[-1] // self.heads).transpose(1, 2)` | Identical |
| `attn_mask = None` | `attn_mask = None` | Identical |
| `if mask is not None:` | `if mask is not None:` | Identical |
| `if mask.ndim == 2: attn_mask = mask.unsqueeze(1).unsqueeze(2)` | `if mask.ndim == 2: attn_mask = mask.unsqueeze(1).unsqueeze(2)` | Identical |
| `elif mask.ndim == 3: attn_mask = mask.unsqueeze(1)` | `elif mask.ndim == 3: attn_mask = mask.unsqueeze(1)` | Identical |
| `elif mask.ndim == 4: attn_mask = mask` | `elif mask.ndim == 4: attn_mask = mask` | Identical |
| `else: attn_mask = mask.view(b, 1, 1, -1)` | `else: attn_mask = mask.view(b, 1, 1, -1)` | Identical |
| `if attn_mask.dtype != torch.bool: attn_mask = (attn_mask != 0)` | `if attn_mask.dtype != torch.bool: attn_mask = (attn_mask != 0)` | Identical |
| `out = F.scaled_dot_product_attention(...)` | `out = F.scaled_dot_product_attention(...)` | Identical |
| `out = out.transpose(1, 2).reshape(...)` | `out = out.transpose(1, 2).reshape(...)` | Identical |
| `out = self.to_out(out)` | `out = self.to_out(out)` | Identical |
| `out = out.permute(0, 2, 1).view(b, c, h, w)` | `out = out.permute(0, 2, 1).view(b, c, h, w)` | Identical |
| `return x + out` | `return x + out` | Identical |

**Verdict**: 100% Match.

#### Part B: `Unet.forward` in `models/unet.py`

| Blueprint 1.4 (audit_report.md:980-1007) | Current Code (models/unet.py:50-77) | Status |
|---|---|---|
| `def forward(self, x, time, context, mask=None):` | `def forward(self, x, time, context, mask=None):` | Identical |
| `skip1 = self.attn_inc(self.inc(x, t), context, mask=mask)` | `skip1 = self.attn_inc(self.inc(x, t), context, mask=mask)` | Identical |
| `skip2 = self.attn_down1(self.down1(skip1, t), context, mask=mask)` | `skip2 = self.attn_down1(self.down1(skip1, t), context, mask=mask)` | Identical |
| `skip3 = self.attn_down2(x_down2, context, mask=mask)` | `skip3 = self.attn_down2(x_down2, context, mask=mask)` | Identical |
| `bott = self.attn_bott1(bott, context, mask=mask)` | `bott = self.attn_bott1(bott, context, mask=mask)` | Identical |
| `x = self.attn_up1(x, context, mask=mask)` | `x = self.attn_up1(x, context, mask=mask)` | Identical |
| `x = self.attn_up2(x, context, mask=mask)` | `x = self.attn_up2(x, context, mask=mask)` | Identical |

**Verdict**: 100% Match.

---

### 5.3 Blueprint 2.1 vs `models/diffusion.py`

| Blueprint 2.1 (audit_report.md:1026-1081) | Current Code (models/diffusion.py:49-51, 75-145) | Status | Critical Deviation |
|---|---|---|---|
| `self.alphas_cumprod_prev = ...` | `self.alphas_cumprod_prev = ...` (line 49) | **Match** | None |
| `self.posterior_mean_coef1 = ...` | `self.posterior_mean_coef1 = ...` (line 50) | **Match** | None |
| `self.posterior_mean_coef2 = ...` | `self.posterior_mean_coef2 = ...` (line 51) | **Match** | None |
| `def sample(self, model, x, t, context=None, uncond_context=None, mask=None, uncond_mask=None, guidance_scale=3.0, noise_free=False, clip_denoised=True):` | `def sample(self, model, x, t, context=None, uncond_context=None, guidance_scale=3.0, noise_free=False, clip_denoised=True):` (line 75) | **MISMATCH** | **`mask=None, uncond_mask=None` missing from signature** |
| `if mask is not None and uncond_mask is not None: mask_input = torch.cat([mask, uncond_mask], dim=0)` | *(Not present)* | **MISMATCH** | **CFG mask concatenation omitted** |
| `all_noise = model(x_input, t_input, context=context_input, mask=mask_input)` | `all_noise = model(x_input, t_input, context=context_input)` (line 103) | **MISMATCH** | **`mask` not passed to model under CFG** |
| `predicted_noise = model(x, t, context=context, mask=mask)` | `predicted_noise = model(x, t, context=context)` (line 113) | **MISMATCH** | **`mask` not passed to model without CFG** |
| `sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].to(x.device)[:, None, None, None]` | `sqrt_alpha_bar_t = torch.sqrt(self.alpha_bars[t]).to(x.device)[:, None, None, None]` (line 117) | Suboptimal | Redundant square root recalculation |
| `beta_t = self.betas[t].to(x.device)[:, None, None, None]` | *(Not present)* | **FATAL MISMATCH** | **`beta_t` NEVER defined; causes `NameError`** |
| `pred_x0 = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t` | `pred_x0 = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t` (line 119) | **Match** | None |
| `pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)` | `pred_x0.clamp_(-1.0, 1.0)` (line 123) | **Match** | In-place vs out-of-place |
| `mean = coef1 * pred_x0 + coef2 * x` | `mean = coef1 * pred_x0 + coef2 * x` (line 128) | **Match** | None |
| `sigma_t = torch.sqrt(torch.clamp(beta_t, min=1e-20))` | `sigma_t = torch.sqrt(torch.clamp(beta_t, min=1e-20))` (line 143) | **CRASH** | Crashes due to undefined `beta_t` above |

---

## 6. Actionable Blueprint Remediation Proposal

To achieve full compliance with Blueprint 2.1 and eliminate both the `NameError` and the `TypeError`, the remediation implementer must apply the exact Blueprint 2.1 implementation to `DiffusionReverseProcess.sample()` in `models/diffusion.py`:

```python
# models/diffusion.py (Proposed drop-in replacement for DiffusionReverseProcess.sample)
def sample(self, model, x, t, context=None, uncond_context=None, 
           mask=None, uncond_mask=None, guidance_scale=3.0, noise_free=False, clip_denoised=True):
    """
    Esegue un singolo step di campionamento inverso con Classifier-Free Guidance e Dynamic Range Clipping.
    Risolve DEF-17 ed elimina il NameError su beta_t e il TypeError sul passaggio di mask.
    """
    # 1. Classifier-Free Guidance (CFG) con concatenazione sicura delle maschere
    if guidance_scale > 1.0 and context is not None and uncond_context is not None:
        x_input = torch.cat([x, x], dim=0)
        t_input = torch.cat([t, t], dim=0)
        context_input = torch.cat([context, uncond_context], dim=0)
        
        mask_input = None
        if mask is not None and uncond_mask is not None:
            mask_input = torch.cat([mask, uncond_mask], dim=0)
        elif mask is not None:
            uncond_m = torch.ones_like(mask)
            mask_input = torch.cat([mask, uncond_m], dim=0)
            
        all_noise = model(x_input, t_input, context=context_input, mask=mask_input)
        eps_cond, eps_uncond = torch.chunk(all_noise, 2, dim=0)
        predicted_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    else:
        predicted_noise = model(x, t, context=context, mask=mask)

    # 2. Coefficienti per lo step t corrente
    sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].to(x.device)[:, None, None, None]
    sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].to(x.device)[:, None, None, None]
    beta_t = self.betas[t].to(x.device)[:, None, None, None]

    # 3. Stima di x_0 pulita (Ho et al. 2020 Eq. 12 / Nichol & Dhariwal 2021)
    pred_x0 = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

    # 4. Dynamic Range Clipping: ancora la traiettoria al dominio dell'immagine [-1.0, 1.0]
    if clip_denoised:
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

    # 5. Calcolo della media a posteriori mu_theta da pred_x0 clippato
    coef1 = self.posterior_mean_coef1[t].to(x.device)[:, None, None, None]
    coef2 = self.posterior_mean_coef2[t].to(x.device)[:, None, None, None]
    mean = coef1 * pred_x0 + coef2 * x

    # 6. Condizione terminale per t=0
    if (t == 0).all() or noise_free:
        return mean

    # 7. Iniezione di rumore Langevin (stabilità con clamp min=1e-20)
    z = torch.randn_like(x)
    sigma_t = torch.sqrt(torch.clamp(beta_t, min=1e-20))
    return mean + sigma_t * z
```

Additionally, in `inference.py` lines 81–89, the caller should pass `mask=mask, uncond_mask=uncond_mask` to `self.reverse_process.sample` to take advantage of the wired cross-attention masking during inference.
