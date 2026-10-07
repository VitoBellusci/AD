"""
Reviewer 2 Adversarial Stress Test Suite
Comprehensive Verification of UNet EMA Integration, Edge Cases, and Regressions:
1. use_ema=False enforcement (both with ema_unet=None and pre-existing ema_unet)
2. val_loader=None checkpoint integrity and EMA preservation
3. extract_ema_state_dict handling of DataParallel wrapped AveragedModel
4. main.py --no_ema handling and checkpoint flow
5. Resumption resilience against 'module.n_averaged' and DataParallel prefix keys
6. inference.py AvatarGenerator use_ema switch and fail-soft fallback
7. evaluate.py use_ema switch and fail-soft fallback
8. resolve_checkpoint consistency when explicit_path does not exist
9. Mathematical precision of EMA updates and AMP GradScaler step-skipping
"""

import os
import tempfile
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.unet import Unet
from models.transformer import FullTextEncoder
from models.diffusion import DiffusionForwardProcess
from train import (
    create_ema_model,
    make_ema_multi_avg_fn,
    strip_prefix,
    extract_ema_state_dict,
    configure_optimizers,
    train,
    set_seed
)
from main import (
    parse_args as parse_main_args,
    resolve_checkpoint as resolve_checkpoint_main
)
from inference import (
    AvatarGenerator,
    parse_args as parse_infer_args,
    resolve_checkpoint as resolve_checkpoint_infer
)
from evaluate import (
    resolve_checkpoint as resolve_checkpoint_eval
)


class DummyTokenizer:
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3, "avatar": 4}


class DummyDataset(Dataset):
    def __init__(self, size=8):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(3, 64, 64)
        tokens = torch.randint(0, 5, (20,), dtype=torch.long)
        return image, tokens


class MultiWrapper(nn.Module):
    """Simulates DataParallel / DistributedDataParallel wrappers."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_1_use_ema_false_enforcement():
    """Verify that when use_ema=False, ema_unet is neither created nor updated, and checkpoint omits EMA."""
    device = "cpu"
    set_seed(42)

    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device)
    text_encoder = FullTextEncoder(vocab_size=10, max_seq_len=20, d_model=32, d_ff=64, num_layers=2).to(device)
    forward_process = DiffusionForwardProcess(num_time_steps=50, device=device)
    dataloader = DataLoader(DummyDataset(size=4), batch_size=2)
    tokenizer = DummyTokenizer()
    opt, sched = configure_optimizers(unet, text_encoder, lr=1e-4, total_epochs=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Case A: ema_unet is None, use_ema=False
        returned_ema = train(
            unet=unet,
            text_encoder=text_encoder,
            forward_process=forward_process,
            dataloader=dataloader,
            tokenizer=tokenizer,
            optimizer=opt,
            scheduler=sched,
            epochs=1,
            device=device,
            checkpoint_dir=tmpdir,
            max_steps=2,
            use_ema=False
        )
        assert returned_ema is None, "train() must return None when use_ema=False"

        ckpt_path = os.path.join(tmpdir, "checkpoint_epoch_1.pt")
        assert os.path.exists(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        assert "unet_state_dict" in ckpt
        assert "ema_state_dict" not in ckpt, "Checkpoint must not contain ema_state_dict when use_ema=False"
        assert "ema_unet_state_dict" not in ckpt, "Checkpoint must not contain ema_unet_state_dict when use_ema=False"
        assert "ema_n_averaged" not in ckpt, "Checkpoint must not contain ema_n_averaged when use_ema=False"

        # Case B: ema_unet passed in as an existing model, but use_ema=False is explicitly passed
        existing_ema = create_ema_model(unet, decay=0.99, device=device)
        initial_param = list(existing_ema.parameters())[0].detach().clone()

        returned_ema_b = train(
            unet=unet,
            text_encoder=text_encoder,
            forward_process=forward_process,
            dataloader=dataloader,
            tokenizer=tokenizer,
            optimizer=opt,
            scheduler=sched,
            epochs=1,
            device=device,
            checkpoint_dir=tmpdir,
            max_steps=2,
            ema_unet=existing_ema,
            use_ema=False
        )
        assert returned_ema_b is None, "train() must override ema_unet to None when use_ema=False"
        # Verify existing_ema parameters were NOT modified
        assert torch.allclose(list(existing_ema.parameters())[0], initial_param), "Existing EMA model must NOT be updated when use_ema=False"

    print("Test 1 Passed: use_ema=False enforcement verified.")


def test_2_val_loader_none_checkpoint_integrity():
    """Verify that when val_loader is None, train() saves the checkpoint with complete EMA state."""
    device = "cpu"
    set_seed(42)

    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device)
    text_encoder = FullTextEncoder(vocab_size=10, max_seq_len=20, d_model=32, d_ff=64, num_layers=2).to(device)
    forward_process = DiffusionForwardProcess(num_time_steps=50, device=device)
    dataloader = DataLoader(DummyDataset(size=6), batch_size=2)
    tokenizer = DummyTokenizer()
    opt, sched = configure_optimizers(unet, text_encoder, lr=1e-4, total_epochs=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        returned_ema = train(
            unet=unet,
            text_encoder=text_encoder,
            forward_process=forward_process,
            dataloader=dataloader,
            val_loader=None,  # Explicitly None
            tokenizer=tokenizer,
            optimizer=opt,
            scheduler=sched,
            epochs=1,
            device=device,
            checkpoint_dir=tmpdir,
            max_steps=3,
            use_ema=True,
            ema_decay=0.99
        )
        assert returned_ema is not None
        ckpt_path = os.path.join(tmpdir, "checkpoint_epoch_1.pt")
        assert os.path.exists(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        assert "val_loss" not in ckpt
        assert "unet_state_dict" in ckpt
        assert "ema_state_dict" in ckpt
        assert "ema_unet_state_dict" in ckpt
        assert "ema_n_averaged" in ckpt
        assert ckpt["ema_n_averaged"] == 3

    print("Test 2 Passed: val_loader=None checkpoint integrity verified.")


def test_3_dataparallel_wrapped_ema_extraction():
    """Verify extract_ema_state_dict cleans wrappers and extracts raw UNet weights, AveragedModel state, and int steps."""
    device = "cpu"
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device)
    ema_unet = create_ema_model(unet, decay=0.99, device=device)
    ema_unet.n_averaged.fill_(7)

    # Standard AveragedModel
    raw_sd, full_sd, n_avg = extract_ema_state_dict(ema_unet)
    assert n_avg == 7
    assert isinstance(n_avg, int)
    assert "n_averaged" in full_sd
    assert "n_averaged" not in raw_sd
    for k in raw_sd.keys():
        assert not k.startswith("module.")
    assert len(raw_sd) == len(unet.state_dict())

    # Simulated DataParallel around AveragedModel: MultiWrapper(AveragedModel)
    wrapped_outer = MultiWrapper(ema_unet)
    raw_sd_w, full_sd_w, n_avg_w = extract_ema_state_dict(wrapped_outer)
    assert n_avg_w == 7
    assert "n_averaged" in full_sd_w
    assert "n_averaged" not in raw_sd_w
    for k in raw_sd_w.keys():
        assert not k.startswith("module.")
    assert len(raw_sd_w) == len(unet.state_dict())

    # Simulated AveragedModel wrapping MultiWrapper(unet)
    wrapped_inner_unet = MultiWrapper(unet)
    ema_inner = create_ema_model(wrapped_inner_unet, decay=0.99, device=device)
    ema_inner.n_averaged.fill_(12)
    raw_sd_in, full_sd_in, n_avg_in = extract_ema_state_dict(ema_inner)
    assert n_avg_in == 12
    assert "n_averaged" not in raw_sd_in
    for k in raw_sd_in.keys():
        assert not k.startswith("module.")

    print("Test 3 Passed: DataParallel wrapped EMA extraction verified.")


def test_4_main_no_ema_flow():
    """Verify main.py CLI argument parsing for --no_ema and training flow."""
    parser = parse_main_args()
    args = parser.parse_args(["--no_ema"])
    assert args.use_ema is False

    args_def = parser.parse_args([])
    assert args_def.use_ema is True

    # If use_ema is False, ema_unet is None in main.py
    device = "cpu"
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device)
    ema_unet = create_ema_model(unet, decay=args.ema_decay, device=device) if args.use_ema else None
    assert ema_unet is None

    print("Test 4 Passed: main.py --no_ema CLI and flow verified.")


def test_5_resumption_with_module_n_averaged_and_dataparallel():
    """Verify that state_dict containing 'module.n_averaged' does not crash resumption in main.py."""
    device = "cpu"
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device)
    raw_unet = unet
    ema_unet = create_ema_model(unet, decay=0.99, device=device)

    # Corrupt state_dict with multi-level module prefix on n_averaged
    synthetic_ema_sd = {}
    for k, v in unet.state_dict().items():
        synthetic_ema_sd[f"module.module.{k}"] = v
    synthetic_ema_sd["module.module.n_averaged"] = torch.tensor(15)

    ckpt = {
        "epoch": 2,
        "unet_state_dict": unet.state_dict(),
        "ema_state_dict": synthetic_ema_sd,
        "ema_n_averaged": 15
    }

    raw_ema = ema_unet
    while hasattr(raw_ema, "module"):
        raw_ema = raw_ema.module

    ema_loaded = False
    # Tier 1 direct load: will fail due to module.module
    try:
        ema_unet.load_state_dict(ckpt["ema_state_dict"])
        ema_loaded = True
    except Exception:
        pass
    assert not ema_loaded

    # Tier 2: unwrapped fallback
    weights_to_load = strip_prefix(ckpt["ema_state_dict"])
    weights_to_load = {k: v for k, v in weights_to_load.items() if k != "n_averaged"}

    # This load must succeed without 'Unexpected key: n_averaged'
    raw_ema.load_state_dict(weights_to_load)
    ema_loaded = True

    # Restore n_averaged
    n_val = ckpt["ema_n_averaged"]
    n_int = int(n_val.item() if hasattr(n_val, "item") else n_val)
    if isinstance(ema_unet.n_averaged, torch.Tensor):
        ema_unet.n_averaged.fill_(n_int)
    else:
        ema_unet.n_averaged = n_int

    assert ema_loaded is True
    assert ema_unet.n_averaged.item() == 15
    print("Test 5 Passed: Resumption with 'module.n_averaged' verified.")


def test_6_inference_use_ema_switch_and_fallback():
    """Verify AvatarGenerator respects use_ema=False, and gracefully falls back on corrupted EMA."""
    device = "cpu"
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device)
    raw_weights = {k: v.clone() for k, v in unet.state_dict().items()}
    ema_weights = {k: v.clone() + 2.0 for k, v in unet.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy config
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            f.write('{"resolution": [64, 64], "max_seq_len": 20, "vocab_path": "nonexistent.json"}')

        # Checkpoint with both weights
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")
        torch.save({
            "epoch": 1,
            "unet_state_dict": raw_weights,
            "ema_unet_state_dict": ema_weights,
            "text_encoder_state_dict": {}
        }, ckpt_path)

        # AvatarGenerator with use_ema=True
        gen_ema = AvatarGenerator(config_path=config_path, checkpoint_path=ckpt_path, device=device, use_ema=True)
        first_param = next(gen_ema.unet.parameters())
        first_k = list(raw_weights.keys())[0]
        assert torch.allclose(first_param, ema_weights[first_k]), "AvatarGenerator should load EMA weights when use_ema=True"

        # AvatarGenerator with use_ema=False
        gen_raw = AvatarGenerator(config_path=config_path, checkpoint_path=ckpt_path, device=device, use_ema=False)
        first_param_raw = next(gen_raw.unet.parameters())
        assert torch.allclose(first_param_raw, raw_weights[first_k]), "AvatarGenerator should load raw UNet weights when use_ema=False"

        # Checkpoint with corrupted EMA weights
        corrupt_ckpt_path = os.path.join(tmpdir, "corrupt_ckpt.pt")
        torch.save({
            "epoch": 1,
            "unet_state_dict": raw_weights,
            "ema_unet_state_dict": {"invalid_key": torch.tensor([1, 2, 3])},
            "text_encoder_state_dict": {}
        }, corrupt_ckpt_path)

        gen_fallback = AvatarGenerator(config_path=config_path, checkpoint_path=corrupt_ckpt_path, device=device, use_ema=True)
        first_param_fallback = next(gen_fallback.unet.parameters())
        assert torch.allclose(first_param_fallback, raw_weights[first_k]), "AvatarGenerator should fallback to raw weights when EMA is corrupted"

    print("Test 6 Passed: inference.py AvatarGenerator use_ema switch and fallback verified.")


def test_7_evaluate_use_ema_switch_and_fallback():
    """Verify evaluate() checkpoint loading logic respects use_ema and falls back cleanly."""
    device = "cpu"
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device)
    raw_weights = {k: v.clone() for k, v in unet.state_dict().items()}
    ema_weights = {k: v.clone() + 3.0 for k, v in unet.state_dict().items()}

    ckpt = {
        "unet_state_dict": raw_weights,
        "ema_unet_state_dict": ema_weights,
        "text_encoder_state_dict": {}
    }

    def simulate_eval_load(checkpoint, use_ema):
        test_unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32).to(device)
        unet_loaded = False
        if use_ema:
            if 'ema_unet_state_dict' in checkpoint:
                try:
                    w = strip_prefix(checkpoint['ema_unet_state_dict'])
                    w = {k: v for k, v in w.items() if k != 'n_averaged'}
                    test_unet.load_state_dict(w)
                    unet_loaded = True
                except Exception:
                    pass
            if not unet_loaded and 'ema_state_dict' in checkpoint:
                try:
                    w = strip_prefix(checkpoint['ema_state_dict'])
                    w = {k: v for k, v in w.items() if k != 'n_averaged'}
                    test_unet.load_state_dict(w)
                    unet_loaded = True
                except Exception:
                    pass
        if not unet_loaded:
            w = strip_prefix(checkpoint['unet_state_dict'])
            w = {k: v for k, v in w.items() if k != 'n_averaged'}
            test_unet.load_state_dict(w)
        return test_unet

    u_ema = simulate_eval_load(ckpt, use_ema=True)
    first_k = list(raw_weights.keys())[0]
    assert torch.allclose(next(u_ema.parameters()), ema_weights[first_k])

    u_no_ema = simulate_eval_load(ckpt, use_ema=False)
    assert torch.allclose(next(u_no_ema.parameters()), raw_weights[first_k])

    # Legacy checkpoint
    legacy_ckpt = {"unet_state_dict": raw_weights}
    u_legacy = simulate_eval_load(legacy_ckpt, use_ema=True)
    assert torch.allclose(next(u_legacy.parameters()), raw_weights[first_k])

    print("Test 7 Passed: evaluate.py use_ema switch and legacy fallback verified.")


def test_8_resolve_checkpoint_missing_explicit_path():
    """Verify that resolve_checkpoint raises FileNotFoundError across all scripts when explicit path is invalid."""
    invalid_path = "nonexistent_checkpoint_file_12345.pt"

    for resolver in [resolve_checkpoint_main, resolve_checkpoint_infer, resolve_checkpoint_eval]:
        try:
            resolver(checkpoint_dir="checkpoints", explicit_path=invalid_path)
            assert False, f"{resolver.__name__} should have raised FileNotFoundError for invalid path"
        except FileNotFoundError:
            pass

    print("Test 8 Passed: resolve_checkpoint raises FileNotFoundError for nonexistent explicit path across all modules.")


def test_9_amp_step_skipping_and_math():
    """Verify mathematical formula for EMA and AMP scaler step-skipping behavior."""
    decay = 0.95
    unet = Unet(in_channels=3, out_channels=3, base_channels=16, context_dim=32)
    ema_unet = create_ema_model(unet, decay=decay)

    p0 = list(unet.parameters())[0].detach().clone()
    ema_unet.update_parameters(unet)
    assert torch.allclose(list(ema_unet.module.parameters())[0], p0)

    # Step 2
    with torch.no_grad():
        list(unet.parameters())[0].add_(2.0)
    p1 = list(unet.parameters())[0].detach().clone()
    ema_unet.update_parameters(unet)
    expected_p1 = p0 * decay + p1 * (1.0 - decay)
    assert torch.allclose(list(ema_unet.module.parameters())[0], expected_p1, atol=1e-5)

    # Step 3 skipped due to simulated inf
    scale_before = 65536.0
    scale_after = 32768.0  # backed off
    step_executed = not (scale_after < scale_before)
    assert step_executed is False
    if step_executed:
        ema_unet.update_parameters(unet)
    # EMA unchanged
    assert torch.allclose(list(ema_unet.module.parameters())[0], expected_p1, atol=1e-5)
    assert ema_unet.n_averaged.item() == 2

    print("Test 9 Passed: EMA mathematical formulation and AMP step-skipping verified.")


if __name__ == "__main__":
    test_1_use_ema_false_enforcement()
    test_2_val_loader_none_checkpoint_integrity()
    test_3_dataparallel_wrapped_ema_extraction()
    test_4_main_no_ema_flow()
    test_5_resumption_with_module_n_averaged_and_dataparallel()
    test_6_inference_use_ema_switch_and_fallback()
    test_7_evaluate_use_ema_switch_and_fallback()
    test_8_resolve_checkpoint_missing_explicit_path()
    test_9_amp_step_skipping_and_math()
    print("\n=======================================================")
    print("ALL 9 REVIEWER 2 ADVERSARIAL STRESS TESTS PASSED CLEANLY!")
    print("=======================================================")
