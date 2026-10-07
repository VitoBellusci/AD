# Implementer 1 Briefing: UNet Exponential Moving Average (EMA) Integration

## 1. Mission Overview
Integrate Exponential Moving Average (EMA) for the UNet model in the PyTorch diffusion training loop (`train.py` and `main.py`) to prevent mode collapse during training.

## 2. Requirements Traceability
- **R1 (EMA Implementation via Pre-built Library)**:
  - Integrate PyTorch's official EMA library (`torch.optim.swa_utils.AveragedModel` with `get_ema_multi_avg_fn`), which is pre-built, production-grade, hardware-accelerated (`_foreach_lerp_`), and specifically recommended by PyTorch and the author of `torch-ema`.
  - Maintain an EMA copy of the UNet weights with configurable decay (default: 0.9999).
- **R2 (Training Loop & Checkpoint Serialization)**:
  - Update EMA weights after every optimization step in `train.py`.
  - Save both the active UNet weights (`unet_state_dict`) and the EMA UNet weights (`ema_unet_state_dict` and `ema_state_dict`) in every checkpoint.
  - Implement full checkpoint resumption in `main.py` and `train.py`, ensuring training resumes seamlessly with restored EMA parameters and RNG states.

## 3. Acceptance Criteria
- [ ] Fast dummy training run completes successfully without crashing.
- [ ] Checkpoint files contain EMA state dictionary alongside active model weights.
- [ ] Training resumes from generated checkpoints without errors.
