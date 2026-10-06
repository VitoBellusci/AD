import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import numpy as np


from typing import Optional


def set_seed(seed: int = 42):
    """
    Imposta un seed fisso per garantire la totale riproducibilità degli esperimenti.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_unconditional_context(text_encoder, tokenizer, batch_size, max_seq_len, device):
    """
    Genera l'embedding per il contesto incondizionato (testo vuoto o solo PAD).
    Utile sia per la baseline incondizionata che per il Classifier-Free Guidance.
    """
    # Creiamo un tensore pieno di token <PAD> (o un token <UNK> vuoto)
    pad_token_id = tokenizer.vocab.get("<PAD>", 0)
    uncond_tokens = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long, device=device)
    
    # La maschera per il PAD: consentiamo l'attenzione uniforme sui token non condizionati
    mask = torch.ones((batch_size, 1, 1, max_seq_len), device=device, dtype=torch.bool)
    
    # Estrazione dell'embedding
    uncond_context = text_encoder(uncond_tokens, mask)
    return uncond_context

def configure_optimizers(
    unet: nn.Module,
    text_encoder: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 5,
    total_epochs: int = 50
):
    """
    Configura l'ottimizzatore AdamW con weight decay disaccoppiato (DEF-19):
    - Parametri 2D/4D (pesi) ricevono weight_decay.
    - Parametri 1D (bias, LayerNorm, GroupNorm) ricevono weight_decay = 0.0.
    Scheduler: Warmup lineare per 5 epoche seguito da Cosine Annealing via SequentialLR.
    """
    decay_params = []
    no_decay_params = []
    
    for model in [unet, text_encoder]:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "norm" in name.lower() or "bias" in name.lower():
                no_decay_params.append(param)
            else:
                decay_params.append(param)
                
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    optimizer = optim.AdamW(optim_groups, lr=lr)

    # Scheduler con Warmup Lineare e Cosine Annealing
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs))
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )
    return optimizer, scheduler

def train(
    unet: nn.Module,
    text_encoder: nn.Module,
    forward_process,
    dataloader: DataLoader,
    tokenizer,
    optimizer,
    scheduler,
    epochs: int,
    device: str,
    val_loader: Optional[DataLoader] = None,
    checkpoint_dir: str = "checkpoints",
    start_epoch: int = 0,
    conditional: bool = True,
    cfg_drop_rate: float = 0.1,
    lr: float = 1e-4
):
    """
    Ciclo di addestramento unificato per:
    - unconditional neural baseline (conditional = False)
    - conditional model (conditional = True)

    Supporta:
    - Mascheramento esplicito unet(..., mask=mask) e gestione sicura mask=None (Blueprint 1.4, DEF-04)
    - Decoupled gradient clipping per unet e text_encoder (Blueprint 3.1, DEF-19)
    - Tracking e logging della validation loss con torch.no_grad() (DEF-10)
    - Mixed precision AMP (autocast + GradScaler) quando device.type == 'cuda' (DEF-14)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device(device) if isinstance(device, str) else device
    unet.to(device)
    text_encoder.to(device)

    # AMP setup (DEF-14)
    use_amp = (device.type == "cuda" and torch.cuda.is_available())
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None

    criterion = nn.MSELoss()

    unet.train()
    text_encoder.train()

    print(f"Inizio Addestramento - Modalità: {'Condizionata' if conditional else 'Incondizionata (Baseline)'} | AMP: {use_amp}")

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, text_tokens in progress_bar:
            images = images.to(device)
            text_tokens = text_tokens.to(device)
            batch_size = images.shape[0]

            optimizer.zero_grad()

            """            
            Si campiona un timestep t uniformemente per ogni immagine nel batch, da 0 al numero di timestep.
            (batch_size,) definisce il livello di rumore, diverso per ogni immagine processata presente nel batch_size.
            """         
            timesteps = torch.randint(0, forward_process.num_time_steps, (batch_size,), device=device).long()

            # Generazione del rumore target (epsilon)
            noise = torch.randn_like(images).to(device)

            # Processo Forward: aggiunta del rumore all'immagine pulita
            noisy_images = forward_process.add_noise(images, noise, timesteps)

            # Estrazione del contesto testuale e maschera (DEF-04)
            # Inizializzazione esplicita a None per evitare UnboundLocalError quando conditional=False
            mask = None
            with torch.amp.autocast('cuda', enabled=use_amp):
                if conditional:
                    pad_token_id = tokenizer.vocab.get("<PAD>", 0)
                    mask = (text_tokens != pad_token_id).unsqueeze(1).unsqueeze(2).to(device)
                    context = text_encoder(text_tokens, mask)

                    # Classifier-Free Guidance (CFG) Dropout
                    # Con probabilità `cfg_drop_rate`, scartiamo il testo e passiamo un contesto vuoto.
                    if cfg_drop_rate > 0.0:
                        drop_mask = torch.rand(batch_size, device=device) < cfg_drop_rate
                        if drop_mask.any():
                            uncond_context = get_unconditional_context(
                                text_encoder, tokenizer, batch_size, text_tokens.shape[1], device
                            )
                            context = torch.where(drop_mask.unsqueeze(1).unsqueeze(2), uncond_context, context)
                            uncond_mask = torch.ones_like(mask)
                            mask = torch.where(drop_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3), uncond_mask, mask)
                else:
                    # Per la baseline incondizionata, passiamo sempre un contesto vuoto e mask = None
                    mask = None
                    context = get_unconditional_context(
                        text_encoder, tokenizer, batch_size, text_tokens.shape[1], device
                    )

                # 5. Predizione del rumore target con supporto AMP (DEF-14) e maschera esplicita (DEF-04)
                predicted_noise = unet(noisy_images, timesteps, context, mask=mask)
                loss = criterion(predicted_noise, noise)

            # 6. Backward pass con GradScaler (se AMP attivo) e decoupled gradient clipping (DEF-19)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"MSE Loss": f"{loss.item():.4f}"})

        # Riduce il learning rate
        scheduler.step()
        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch+1} completata | Loss Media: {avg_loss:.4f}")

        # Validation loss tracking (DEF-10)
        avg_val_loss = None
        if val_loader is not None and len(val_loader) > 0:
            unet.eval()
            text_encoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_images, val_tokens in val_loader:
                    val_images = val_images.to(device)
                    val_tokens = val_tokens.to(device)
                    val_b = val_images.shape[0]

                    val_timesteps = torch.randint(0, forward_process.num_time_steps, (val_b,), device=device).long()
                    val_noise = torch.randn_like(val_images).to(device)
                    val_noisy_images = forward_process.add_noise(val_images, val_noise, val_timesteps)

                    val_mask = None
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        if conditional:
                            pad_token_id = tokenizer.vocab.get("<PAD>", 0)
                            val_mask = (val_tokens != pad_token_id).unsqueeze(1).unsqueeze(2).to(device)
                            val_context = text_encoder(val_tokens, val_mask)
                        else:
                            val_mask = None
                            val_context = get_unconditional_context(
                                text_encoder, tokenizer, val_b, val_tokens.shape[1], device
                            )

                        val_pred_noise = unet(val_noisy_images, val_timesteps, val_context, mask=val_mask)
                        val_step_loss = criterion(val_pred_noise, val_noise)

                    val_loss += val_step_loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1} | Validation Loss Media: {avg_val_loss:.4f}")
            unet.train()
            text_encoder.train()

        # 7. Salvataggio del Checkpoint (permettendo di riprendere l'addestramento)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        checkpoint_dict = {
            'epoch': epoch + 1,
            'unet_state_dict': unet.state_dict(),
            'text_encoder_state_dict': text_encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'random_rng_state': random.getstate(),
            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'torch_cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        if avg_val_loss is not None:
            checkpoint_dict['val_loss'] = avg_val_loss
        if scaler is not None:
            checkpoint_dict['scaler_state_dict'] = scaler.state_dict()

        torch.save(checkpoint_dict, checkpoint_path)
        print(f"Checkpoint salvato: {checkpoint_path}")