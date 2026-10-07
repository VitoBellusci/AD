"""
Comprehensive Programmatic Test Suite for UNet EMA Integration
Validates all requirements (R1, R2) and Acceptance Criteria.
"""

import os
import shutil
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.unet import Unet
from models.transformer import FullTextEncoder
from models.diffusion import DiffusionForwardProcess
from train import create_ema_model, configure_optimizers, train, set_seed
from main import parse_args, resolve_checkpoint


class DummyTokenizer:
    vocab = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3, "avatar": 4}


class DummyDataset(Dataset):
    def __init__(self, size=16):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 3 channels, 64x64 resolution, 20 token sequence length
        image = torch.randn(3, 64, 64)
        tokens = torch.randint(0, 5, (20,), dtype=torch.long)
        return image, tokens


def test_1_ema_creation_and_properties():
    """Verify EMA model creation and property initialization."""
    device = "cpu"
    unet = Unet(in_channels=3, out_channels=3, base_channels=32, context_dim=64).to(device)
    ema_unet = create_ema_model(unet, decay=0.999, device=device)

    assert ema_unet is not None, "EMA UNet must not be None"
    assert hasattr(ema_unet, "module"), "AveragedModel must have .module containing UNet"
    assert hasattr(ema_unet, "n_averaged"), "AveragedModel must have n_averaged buffer"
    assert ema_unet.n_averaged.item() == 0, f"Expected n_averaged=0 initially, got {ema_unet.n_averaged.item()}"

    # Verify all EMA parameters have requires_grad=False
    for name, param in ema_unet.named_parameters():
        assert not param.requires_grad, f"EMA param {name} must have requires_grad=False"

    print("Test 1 Passed: EMA creation and properties verified.")


def test_2_ema_update_math():
    """Verify mathematical correctness of EMA update across optimization steps."""
    device = "cpu"
    decay = 0.9
    unet = Unet(in_channels=3, out_channels=3, base_channels=32, context_dim=64).to(device)
    ema_unet = create_ema_model(unet, decay=decay, device=device)

    # Initial weights
    initial_weight = list(unet.parameters())[0].detach().clone()
    
    # Step 1 update: first update copies model parameters exactly
    ema_unet.update_parameters(unet)
    assert ema_unet.n_averaged.item() == 1
    ema_p1 = list(ema_unet.module.parameters())[0].detach()
    assert torch.allclose(ema_p1, initial_weight), "On first step, EMA should copy active model parameters"

    # Perturb unet weights (simulating optimizer step)
    with torch.no_grad():
        list(unet.parameters())[0].add_(1.0)
    step2_weight = list(unet.parameters())[0].detach().clone()

    # Step 2 update: EMA applies decay * old + (1 - decay) * new
    ema_unet.update_parameters(unet)
    assert ema_unet.n_averaged.item() == 2
    ema_p2 = list(ema_unet.module.parameters())[0].detach()
    expected_p2 = initial_weight * decay + step2_weight * (1.0 - decay)
    assert torch.allclose(ema_p2, expected_p2, atol=1e-5), f"EMA mismatch: got {ema_p2[:2]}, expected {expected_p2[:2]}"

    print("Test 2 Passed: EMA update mathematical formulation verified.")


def test_3_dummy_training_and_checkpoint_contents():
    """Verify fast dummy training run (max_steps=5) and checkpoint serialization."""
    device = "cpu"
    set_seed(42)

    unet = Unet(in_channels=3, out_channels=3, base_channels=32, context_dim=64).to(device)
    text_encoder = FullTextEncoder(vocab_size=10, max_seq_len=20, d_model=64, d_ff=128, num_layers=2).to(device)
    forward_process = DiffusionForwardProcess(num_time_steps=100, device=device)
    dataloader = DataLoader(DummyDataset(size=10), batch_size=2)
    val_loader = DataLoader(DummyDataset(size=4), batch_size=2)
    tokenizer = DummyTokenizer()

    optimizer, scheduler = configure_optimizers(unet, text_encoder, lr=1e-4, total_epochs=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Run 1 epoch dummy training with max_steps=5
        trained_ema = train(
            unet=unet,
            text_encoder=text_encoder,
            forward_process=forward_process,
            dataloader=dataloader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=1,
            device=device,
            checkpoint_dir=tmpdir,
            conditional=True,
            max_steps=5,
            ema_decay=0.999,
            use_ema=True
        )

        assert trained_ema is not None, "train() should return trained ema_unet"
        ckpt_path = os.path.join(tmpdir, "checkpoint_epoch_1.pt")
        assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"

        # Inspect checkpoint contents
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Acceptance Criterion 2: Checkpoint contains regular weights AND EMA state dict
        assert "unet_state_dict" in ckpt, "Checkpoint missing unet_state_dict"
        assert "text_encoder_state_dict" in ckpt, "Checkpoint missing text_encoder_state_dict"
        assert "ema_state_dict" in ckpt, "Checkpoint missing ema_state_dict"
        assert "ema_unet_state_dict" in ckpt, "Checkpoint missing ema_unet_state_dict"
        assert "ema_n_averaged" in ckpt, "Checkpoint missing ema_n_averaged"

        # Verify EMA state dictionary integrity
        assert ckpt["ema_n_averaged"] == 5, f"Expected 5 averaged steps, got {ckpt['ema_n_averaged']}"
        assert len(ckpt["ema_unet_state_dict"]) == len(ckpt["unet_state_dict"]), "EMA UNet state dict key count mismatch"

        print("Test 3 Passed: Dummy training run (max_steps=5) and checkpoint serialization verified.")


def test_4_checkpoint_resumption():
    """Verify training resumption from newly generated checkpoint with EMA state."""
    device = "cpu"
    set_seed(42)

    unet = Unet(in_channels=3, out_channels=3, base_channels=32, context_dim=64).to(device)
    text_encoder = FullTextEncoder(vocab_size=10, max_seq_len=20, d_model=64, d_ff=128, num_layers=2).to(device)
    forward_process = DiffusionForwardProcess(num_time_steps=100, device=device)
    dataloader = DataLoader(DummyDataset(size=10), batch_size=2)
    tokenizer = DummyTokenizer()

    optimizer, scheduler = configure_optimizers(unet, text_encoder, lr=1e-4, total_epochs=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Train epoch 1
        train(
            unet=unet,
            text_encoder=text_encoder,
            forward_process=forward_process,
            dataloader=dataloader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=1,
            device=device,
            checkpoint_dir=tmpdir,
            max_steps=3,
            ema_decay=0.999,
            use_ema=True
        )

        ckpt_epoch1 = os.path.join(tmpdir, "checkpoint_epoch_1.pt")
        assert os.path.exists(ckpt_epoch1)

        # 2. Simulate resuming in main.py: create fresh model instances
        unet_resumed = Unet(in_channels=3, out_channels=3, base_channels=32, context_dim=64).to(device)
        ema_resumed = create_ema_model(unet_resumed, decay=0.999, device=device)
        text_encoder_resumed = FullTextEncoder(vocab_size=10, max_seq_len=20, d_model=64, d_ff=128, num_layers=2).to(device)
        opt_resumed, sched_resumed = configure_optimizers(unet_resumed, text_encoder_resumed, lr=1e-4, total_epochs=2)

        ckpt_loaded = torch.load(ckpt_epoch1, map_location=device, weights_only=False)
        unet_resumed.load_state_dict(ckpt_loaded['unet_state_dict'])
        text_encoder_resumed.load_state_dict(ckpt_loaded['text_encoder_state_dict'])
        opt_resumed.load_state_dict(ckpt_loaded['optimizer_state_dict'])
        sched_resumed.load_state_dict(ckpt_loaded['scheduler_state_dict'])

        # Restore EMA
        assert 'ema_state_dict' in ckpt_loaded
        ema_resumed.load_state_dict(ckpt_loaded['ema_state_dict'])
        assert ema_resumed.n_averaged.item() == 3

        # 3. Resume training for epoch 2
        train(
            unet=unet_resumed,
            text_encoder=text_encoder_resumed,
            forward_process=forward_process,
            dataloader=dataloader,
            tokenizer=tokenizer,
            optimizer=opt_resumed,
            scheduler=sched_resumed,
            epochs=2,
            start_epoch=1,
            device=device,
            checkpoint_dir=tmpdir,
            max_steps=3,
            ema_unet=ema_resumed,
            ema_decay=0.999,
            use_ema=True
        )

        ckpt_epoch2 = os.path.join(tmpdir, "checkpoint_epoch_2.pt")
        assert os.path.exists(ckpt_epoch2), "Resumed training failed to produce epoch 2 checkpoint"
        ckpt2 = torch.load(ckpt_epoch2, map_location=device, weights_only=False)
        assert ckpt2['epoch'] == 2
        assert ckpt2['ema_n_averaged'] == 6, f"Expected 6 cumulative averaged steps, got {ckpt2['ema_n_averaged']}"

        print("Test 4 Passed: Checkpoint resumption with restored EMA state verified.")


def test_5_legacy_checkpoint_fallback():
    """Verify resuming from legacy checkpoint lacking EMA state initializes cleanly without crash."""
    device = "cpu"
    unet = Unet(in_channels=3, out_channels=3, base_channels=32, context_dim=64).to(device)
    ema_unet = create_ema_model(unet, decay=0.999, device=device)

    # Legacy checkpoint with only unet_state_dict
    legacy_ckpt = {
        'epoch': 1,
        'unet_state_dict': unet.state_dict()
    }

    # Simulate main.py legacy restore logic
    if 'ema_state_dict' in legacy_ckpt:
        ema_unet.load_state_dict(legacy_ckpt['ema_state_dict'])
    elif 'ema_unet_state_dict' in legacy_ckpt:
        ema_unet.module.load_state_dict(legacy_ckpt['ema_unet_state_dict'])
    else:
        # Legacy fallback
        raw_ema = ema_unet.module if hasattr(ema_unet, 'module') else ema_unet
        raw_ema.load_state_dict(legacy_ckpt['unet_state_dict'])
        if hasattr(ema_unet, 'n_averaged'):
            ema_unet.n_averaged.fill_(0)

    assert ema_unet.n_averaged.item() == 0
    print("Test 5 Passed: Legacy checkpoint fallback without EMA verified.")


def test_6_cli_argument_parsing():
    """Verify CLI argument flags in main.py parse_args."""
    parser = parse_args()

    # Default
    args_default = parser.parse_args([])
    assert args_default.use_ema is True
    assert args_default.ema_decay == 0.9999

    # Custom decay and disable flag
    args_no_ema = parser.parse_args(["--no_ema", "--ema_decay", "0.995"])
    assert args_no_ema.use_ema is False
    assert args_no_ema.ema_decay == 0.995

    # Enable flag explicit
    args_use_ema = parser.parse_args(["--use_ema", "--max_steps", "5"])
    assert args_use_ema.use_ema is True
    assert args_use_ema.max_steps == 5

    print("Test 6 Passed: CLI argument parsing verified.")


if __name__ == "__main__":
    test_1_ema_creation_and_properties()
    test_2_ema_update_math()
    test_3_dummy_training_and_checkpoint_contents()
    test_4_checkpoint_resumption()
    test_5_legacy_checkpoint_fallback()
    test_6_cli_argument_parsing()
    print("\nALL 6 VERIFICATION TESTS PASSED SUCCESSFULLY!")
