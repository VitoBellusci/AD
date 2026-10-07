# Project: Avatar Diffusion — Comprehensive Code Review, Verification & Fix-up

## Architecture
- **Data Pipeline**: `preprocessing/dataset.py`, `preprocessing/caption_generator.py`, `preprocessing/splitter.py`, `preprocessing/tokenizer.py`.
  - Maps 18 discrete avatar attributes to English natural language captions.
  - Partitions dataset into 4 disjoint splits: `train` (79,634 samples, 80% in-distribution), `val` (9,954 samples, 10% in-distribution), `test_ind` (9,954 samples, 10% in-distribution), and `test_ood` (458 samples, 100% held-out `(hair=98, glasses=11)` combinations).
  - Vocabulary constructed strictly from the `train` split (149 tokens, zero synthetic OOD words).
- **Model Architecture**:
  - `FullTextEncoder` (`models/transformer.py`): 4-layer custom Transformer encoder with learned token embedding, fixed sinusoidal positional encodings, bidirectional multi-head self-attention with `-inf` padding masks, Pre-LN residual connections. Zero external pretrained weights. (2.14M parameters).
  - `Unet` (`models/unet.py`, `models/unet_parts.py`): 3-level pixel-space U-Net operating on $64\times 64$ RGB images with sinusoidal/MLP timestep conditioning, GroupNorm(8), and spatial cross-attention injecting text embeddings. (24.52M parameters).
  - Total system parameter budget: 26.66M parameters (conforms to ~10M–25M "Tiny" budget).
- **Diffusion Mechanics**:
  - `DiffusionScheduler` (`models/diffusion.py`): Supports Nichol-Dhariwal cosine variance schedule ($s=0.008, T=1000$) and Ho et al. linear beta schedule ($10^{-4} \to 0.02$).
  - `DiffusionForwardProcess`: Closed-form analytical perturbation $q(x_t|x_0)$.
  - `DiffusionReverseProcess`: Ho et al. Eq. 12 posterior mean with dynamic range clipping $\hat{x}_0 \in [-1.0, 1.0]$ and Classifier-Free Guidance ($w=3.5$).
- **Inference & Evaluation**:
  - `inference.py`: 1000-step DDPM and 50-step DDIM fast sampling conditioned on natural language prompts across seeds.
  - `evaluate.py` & `metrics.py`: Computes FID (feature=64), KID (subset_size=50), pairwise LPIPS diversity across seeds, parameter count, sampling latency, and peak VRAM across both IID and OOD test splits.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Image Rescaling & Normalization | Resizing to 64x64, normalizing to [-1, 1] | M1 | Survey |
| 2 | Deterministic Natural Language Captions | Natural text captions from metadata attributes, zero numerical IDs | M1 | Survey |
| 3 | Train-Only Vocabulary Extraction | Vocabulary fitted strictly on training partition texts | M1 | Survey |
| 4 | Special Token Preservation | Preserve <PAD>:0, <UNK>:1, <SOS>:2, <EOS>:3 | M1 | Survey |
| 5 | Combinatorial Hold-Out Partitioning | Isolates (hair=98, glasses=11) into test_ood | M1 | Survey |
| 6 | Persistent Split Serialization | Serializes train, val, test_ind, test_ood to splits.json | M1 | Survey |
| 7 | Learned Token Embedding | Handcrafted nn.Embedding from scratch | M2 | Survey |
| 8 | Positional Encoding Buffer | Sinusoidal position encoding buffer up to seq_len 20 | M2 | Survey |
| 9 | Masked Transformer Self-Attention | -inf softmax masking for pad tokens with nan_to_num guard | M2 | Survey |
| 10 | Pixel-Space Denoiser U-Net | 3-stage custom U-Net on RGB pixels | M2 | Survey |
| 11 | Timestep Conditioning MLP | Sinusoidal time embedding + Linear-SiLU-Linear MLP | M2 | Survey |
| 12 | Spatial Cross-Attention | Conditioning visual features on text tokens with boolean mask | M2 | Survey |
| 13 | Cosine Noise Schedule | Nichol-Dhariwal cosine variance schedule | M2 | Survey |
| 14 | Analytical Forward Diffusion | q(x_t \| x_0) closed-form Gaussian noise sampling | M2 | Survey |
| 15 | Reverse Sampling & x0 Clamping | Ho et al. Eq 12 with pred_x0 clamped to [-1, 1] | M2 | Survey |
| 16 | Classifier-Free Guidance | 10% null token dropout, extrapolated dual-pass sampling | M2 | Survey |
| 17 | Decoupled Weight Decay Optimization | AdamW with 1D bias/norm parameters exempt from weight decay | M2 | Survey |
| 18 | Text Encoder LR Warmup Chaining | 5-epoch linear warmup to cosine decay | M2 | Survey |
| 19 | Decoupled Gradient Clipping | Separate clip_grad_norm_ for UNet and Text Encoder | M2 | Survey |
| 20 | Checkpoint State Preservation | Checkpoint saving and deterministic epoch regex restoration | M2 | Survey |
| 21 | Fréchet Inception Distance (FID) | Distributional metric on [0, 1] normalized batches | M3 | Survey |
| 22 | Kernel Inception Distance (KID) | Unbiased MMD with guarded subset size | M3 | Survey |
| 23 | Seed Diversity Metric (LPIPS) | Pairwise perceptual diversity across seeds for identical prompts | M3 | Survey |
| 24 | Efficiency Profiling | Parameters, sampling latency, and peak VRAM | M3 | Survey |
| 25 | Dual Partition Benchmarking | Separate evaluation on Ordinary Test (IID) and Compositional OOD | M3 | Survey |
| 26 | Fast DDIM Evaluation | Fast reverse sampling support in evaluate.py | M3 | Survey |
| 27 | CLI Runner for train.py | if __name__ == "__main__": entrypoint delegating to main() | M1 | Survey |
| 28 | Default Inference Prompt Alignment | Update inference.py default prompt to match held-out OOD composition | M1 | Survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Data Pipeline & CLI Alignment | Fix vocab leakage in main.py, regenerate splits.json (all 4 splits), train.py CLI entrypoint, inference.py default prompt | none | DONE |
| M2 | Models & Diffusion Math Hardening | Guard multi-GPU RNG load, optional linear schedule, verify from-scratch constraints | M1 | DONE |
| M3 | Evaluation Suite Integration | Integrate efficiency & diversity metrics in evaluate.py, add --num_steps/DDIM, guard KID subset_size | M2 | DONE |
| M4 | Functional Verification Runs | Execute dummy training (1 epoch), reverse sampling generation, evaluation metric run | M3 | DONE |
| M5 | Review, Challenge & Forensic Audit | 2 Reviewers, 2 Challengers, 1 Forensic Auditor (zero tolerance) | M4 | DONE |

## Interface Contracts
### Preprocessing ↔ Models / Training
- `AvatarTokenizer`: expects `vocab.json` fitted exclusively on training captions.
  - Vocabulary IDs: 0=`<PAD>`, 1=`<UNK>`, 2=`<SOS>`, 3=`<EOS>`, 4..148=words.
  - Returns `tokens`: `[B, max_seq_len]`, `attention_mask`: `[B, max_seq_len]`.
- `splits.json`:
  - Contains keys `"train"`, `"val"`, `"test_ind"`, `"test_ood"`.
  - All keys are non-empty and 100% mutually disjoint.

### Models ↔ Evaluation / Inference
- `DiffusionEvaluator`:
  - `compute_efficiency_metrics(unet, text_encoder, device)` -> dict of params, latency, VRAM.
  - `compute_diversity_across_seeds(images)` -> float mean pairwise LPIPS.
  - `update_quality_metrics(real, fake)` -> updates FID/KID stats.
  - `compute_quality_metrics()` -> dict of FID, KID mean/std.

## Code Layout
- `preprocessing/`: Dataset, transforms, caption generator, tokenizer, compositional splitter.
- `models/`: Transformer text encoder, U-Net denoiser, attention modules, diffusion scheduler and processes.
- `train.py`: Training engine, loss computation, gradient step, checkpoint saving.
- `main.py`: CLI orchestration, dataset loading, model initialization, training launch.
- `inference.py`: Image generation CLI, DDPM/DDIM reverse sampling, prompt rendering.
- `evaluate.py`: Benchmark evaluation suite running quality, diversity, and efficiency metrics on IID and OOD splits.
- `metrics.py`: Torchmetrics wrappers for FID, KID, LPIPS diversity, and resource profiling.
