import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import numpy as np


from typing import Optional, Union, Any
try:
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
except ImportError:
    from torch.optim.swa_utils import AveragedModel
    get_ema_multi_avg_fn = None


def make_ema_multi_avg_fn(decay: float = 0.9999):
    """
    Funzione di multi-averaging per EMA.
    Aggiorna i pesi storici theta_ema con i pesi correnti theta_cur secondo un fattore di decadimento
    Utilizza torch._foreach_lerp_ se disponibile, altrimenti esegue lerp_ in-place per ogni parametro.
    """
    weight = 1.0 - decay

    @torch.no_grad()
    def ema_update(averaged_param_list, current_param_list, num_averaged):
        try:
            # aggiornamento simultaneo dei tensori
            torch._foreach_lerp_(averaged_param_list, current_param_list, weight)
        except (AttributeError, RuntimeError):
            for p_avg, p_cur in zip(averaged_param_list, current_param_list):
                # interpolazione lineare in-place
                p_avg.lerp_(p_cur, weight)

    return ema_update


def strip_prefix(state_dict):
    """
    Rimuove ricorsivamente eventuali prefissi 'module.' o '_orig_mod.' generati da DataParallel, DDP o torch.compile dal dizionario dei pesi
    """
    cleaned = {}
    for k, v in state_dict.items():
        changed = True
        while changed:
            changed = False
            if k.startswith('module.'):
                k = k[7:]
                changed = True
            elif k.startswith('_orig_mod.'):
                k = k[10:]
                changed = True
        cleaned[k] = v
    return cleaned


def create_ema_model(
    model: nn.Module,
    decay: float = 0.9999,
    device: Optional[Union[str, torch.device]] = None
) -> AveragedModel:
    """
    Crea un modello Exponential Moving Average (EMA) per il denoiser UNet.
    Utilizza torch.optim.swa_utils.AveragedModel con get_ema_multi_avg_fn per mantenere la media esponenziale mobile dei pesi
    e prevenire il model collapse durante la diffusione.
    """
    if not (0.0 <= decay <= 1.0):
        raise ValueError(f"ema_decay deve essere compreso tra 0.0 e 1.0, ricevuto: {decay}")

    raw_model = model
    while hasattr(raw_model, "module"):
        raw_model = raw_model.module

    if device is None:
        try:
            # Estrazione dispositivo: next estrae il primo tensore disponibile
            first_param = next(raw_model.parameters())
            dev = first_param.device
        except StopIteration:
            dev = None
    else:
        dev = torch.device(device) if isinstance(device, str) else device

    multi_fn = None
    if get_ema_multi_avg_fn is not None:
        try:
            multi_fn = get_ema_multi_avg_fn(decay=decay)
        except Exception:
            multi_fn = make_ema_multi_avg_fn(decay=decay)
    else:
        multi_fn = make_ema_multi_avg_fn(decay=decay)

    ema_model = AveragedModel(
        raw_model,
        device=dev,
        multi_avg_fn=multi_fn
    )

    # Disabilitazione di Autograd
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


def extract_ema_state_dict(ema_unet):
    """
    Estrae in modo robusto lo state_dict del modello denoiser base (raw_unet),
    lo state_dict completo di AveragedModel e il contatore intero n_averaged,
    gestendo qualsiasi combinazione di wrapping DataParallel/DistributedDataParallel
    sia all'esterno che all'interno di AveragedModel.
    """
    if ema_unet is None:
        return None, None, None

    # 1. Unwrapping di eventuali wrapper esterni (es. DataParallel(AveragedModel))
    # Itera sull'attributo finché non trova l'oggetto EMA vero e proprio
    avg_model = ema_unet
    while hasattr(avg_model, "module") and not hasattr(avg_model, "n_averaged"):
        avg_model = avg_model.module

    # 2. Estrazione contatore n_averaged
    n_avg = None
    if hasattr(avg_model, "n_averaged"):
        raw_n = avg_model.n_averaged
        n_avg = int(raw_n.item() if hasattr(raw_n, "item") else raw_n)

    # 3. Estrazione dello state_dict pulito per il modulo denoiser base (UNet)
    inner_model = avg_model.module if hasattr(avg_model, "module") else avg_model
    while hasattr(inner_model, "module"):
        inner_model = inner_model.module

    raw_unet_sd = strip_prefix(inner_model.state_dict())
    raw_unet_sd = {k: v for k, v in raw_unet_sd.items() if k != "n_averaged"}

    # 4. Estrazione dello state_dict canonico per AveragedModel per permettere la ripresa del training da un checkpoint
    full_ema_sd = {}
    # Tentativo di richiamo dello state dict. Tuttavia questo dizionario contiene tutti i pesi della rete ma le sue
    # chiavi potrebbero presentare prefissi multipli
    source_sd = avg_model.state_dict() if hasattr(avg_model, "state_dict") else ema_unet.state_dict()
    """
    Finché la chiave corrente (clean_k) inizia con module. o _orig_mod., il ciclo rimuove questi prefissi (troncando 
    rispettivamente i primi 7 o 10 caratteri) e ripete il controllo. Questo garantisce l'estrazione del nome "puro" del parametro, 
    indipendentemente da quanti livelli di incapsulamento (wrapper) si siano sovrapposti durante il run.
    """
    for k, v in source_sd.items():
        clean_k = k
        changed = True
        while changed:
            changed = False
            if clean_k.startswith("module."):
                clean_k = clean_k[7:]
                changed = True
            elif clean_k.startswith("_orig_mod."):
                clean_k = clean_k[10:]
                changed = True
        if clean_k != "n_averaged":
            full_ema_sd[f"module.{clean_k}"] = v
        else:
            full_ema_sd["n_averaged"] = v

    # Fallback
    if "n_averaged" not in full_ema_sd and n_avg is not None:
        full_ema_sd["n_averaged"] = torch.tensor(n_avg, dtype=torch.long)

    return raw_unet_sd, full_ema_sd, n_avg




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
    # Creiamo un tensore pieno di token <PAD>
    pad_token_id = tokenizer.vocab.get("<PAD>", 0)
    uncond_tokens = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long, device=device)
    
    # La maschera per il PAD: consentiamo l'attenzione uniforme sui token non condizionati
    mask = torch.ones((batch_size, 1, 1, max_seq_len), device=device, dtype=torch.bool)
    
    # Estrazione dell'embedding
    uncond_context = text_encoder(uncond_tokens, mask)
    return uncond_context


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
    lr: float = 1e-4,
    max_steps: Optional[int] = None,
    ema_unet: Optional[Union[nn.Module, Any]] = None,
    ema_decay: float = 0.999,
    use_ema: bool = True
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
    - Exponential Moving Average (EMA) per UNet per prevenire il mode collapse (R1, R2)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device(device) if isinstance(device, str) else device
    unet.to(device)
    text_encoder.to(device)

    # Gestione esplicita use_ema=False: se disabilitato, ema_unet non viene istanziato né aggiornato
    if not use_ema:
        ema_unet = None

    # Inizializzazione EMA UNet (R1)
    if ema_unet is None and use_ema:
        ema_unet = create_ema_model(unet, decay=ema_decay, device=device)
        print(f"EMA UNet inizializzato (decay={ema_decay}) su dispositivo: {device}")
    elif ema_unet is not None and isinstance(ema_unet, nn.Module):
        ema_unet.to(device)

    """
    AMP (Automatic Mixed Precision) setup
    Questa tecnica accekera l'addestramento convertendo alcune operazioni in formati a precisione ridotta, tuttavia, restringendo
    il range numerico disponibile, aumentand il rischio che grandi gradienti causino un overflow o che, viceversa, piccoli gradienti 
    causino underflow
    """
    use_amp = (device.type == "cuda" and torch.cuda.is_available())
    # Utilizzato per contrastare l'underflow, moltiplica la loss per un fattore di scala prima della backpropagation
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None

    criterion = nn.MSELoss()

    unet.train()
    text_encoder.train()

    print(f"Inizio Addestramento - Modalità: {'Condizionata' if conditional else 'Incondizionata (Baseline)'} | AMP: {use_amp} | EMA: {ema_unet is not None}")

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        step = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for images, text_tokens in progress_bar:
            images = images.to(device)
            text_tokens = text_tokens.to(device)
            batch_size = images.shape[0]

            optimizer.zero_grad()

            """            
            Si campiona un timestep t uniformemente per ogni immagine nel batch, da 0 al numero di timestep.
            (batch_size,) definisce il livello di rumore, diverso per ogni immagine processata presente nel batch_size, per diversificare l'apprendimento
            """         
            timesteps = torch.randint(0, forward_process.num_time_steps, (batch_size,), device=device).long()

            # Generazione del rumore target (epsilon)
            noise = torch.randn_like(images).to(device)

            # Processo Forward: aggiunta del rumore all'immagine pulita
            noisy_images = forward_process.add_noise(images, noise, timesteps)

            # Estrazione del contesto testuale e maschera
            # Inizializzazione esplicita a None per evitare UnboundLocalError quando conditional=False
            mask = None
            with torch.amp.autocast('cuda', enabled=use_amp):
                if conditional:
                    pad_token_id = tokenizer.vocab.get("<PAD>", 0)
                    # Maschera d'attenzione contenente true in corrispondenza dei token testuali effettivi e false negli indici occupati dai <PAD>
                    mask = (text_tokens != pad_token_id).unsqueeze(1).unsqueeze(2).to(device)
                    # Embedding denso e semantico
                    context = text_encoder(text_tokens, mask)

                    # Classifier-Free Guidance (CFG) Dropout
                    # Con probabilità `cfg_drop_rate`, scartiamo il testo e passiamo un contesto vuoto.
                    if cfg_drop_rate > 0.0:
                        # Genera per ciascuna immagine nel batch un valore stocastico compreso tra 0 e 1. Se tale valore risulta inferiore al rateo 
                        # di dropout impostato, l'elemento corrispondente nel tensore booleano 1D diventerà True.
                        drop_mask = torch.rand(batch_size, device=device) < cfg_drop_rate
                        if drop_mask.any():
                            # Embedding incondizionato
                            uncond_context = get_unconditional_context(
                                text_encoder, tokenizer, batch_size, text_tokens.shape[1], device
                            )
                            # Espande dimensionalmente la drop_mask e, per gli indici del batch in cui questa risulta true, sostituisce le feature
                            # semantiche testuali precalcolate con l'embedding incondizionato
                            context = torch.where(drop_mask.unsqueeze(1).unsqueeze(2), uncond_context, context)
                            uncond_mask = torch.ones_like(mask)
                            """
                            Per gli elementi privi di prompt a causa del dropout, sovrascrive la maschera rigorosa calcolata al passaggio 3 con la 
                            nuova maschera riempita di 1. Questa operazione inganna la cross-attention, forzando la rete a spalmare i pesi dell'attenzione 
                            uniformemente sull'intero contesto vuoto.
                            """
                            mask = torch.where(drop_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3), uncond_mask, mask)
                else:
                    # Per la baseline incondizionata, passiamo sempre un contesto vuoto e mask = None
                    mask = None
                    context = get_unconditional_context(
                        text_encoder, tokenizer, batch_size, text_tokens.shape[1], device
                    )

                # Calcolo del target di velocita (v_target) per v-prediction
                # Formula: v_target = sqrt_alpha_bar_t * noise - sqrt_one_minus_alpha_bar_t * original
                v_target = forward_process.get_velocity(images, noise, timesteps)

                # Predizione del target di velocita con supporto AMP e maschera esplicita
                predicted_v = unet(noisy_images, timesteps, context, mask=mask)
                loss = criterion(predicted_v, v_target)

            # Backward pass con GradScaler (se AMP attivo) e decoupled gradient clipping
            step_executed = True
            if scaler is not None:
                # Per impedire l'underflow causato dalle operazioni in formati a precisione ridotta, questo metodo moltiplica
                # preventivamente la loss per un fattore di scala elevato
                scaler.scale(loss).backward()
                # Prima di poter applicare qualsiasi regolarizzazione sui gradienti, è necessario riportarli alla loro grandezza reale
                scaler.unscale_(optimizer)
                # Rimedio per l'overflow
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)
                # salvataggio della scala corrente
                scale_before = scaler.get_scale()
                # Se i gradienti contenevano inf/NaN, GradScaler salta optimizer.step() e riduce la scala
                scaler.step(optimizer)
                scaler.update()

                scale_after = scaler.get_scale()

                # Se il gradScaler ha ridotto il fattore, significa che lo step di ottimizzazione non è avvenuto
                if scale_after < scale_before:
                    step_executed = False
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), max_norm=1.0)
                optimizer.step()

            # Aggiornamento pesi EMA UNet dopo ogni step di ottimizzazione riuscito 
            if ema_unet is not None and step_executed:
                raw_unet = unet
                while hasattr(raw_unet, 'module'):
                    raw_unet = raw_unet.module
                target_ema = ema_unet
                while hasattr(target_ema, 'module') and not hasattr(target_ema, 'update_parameters'):
                    target_ema = target_ema.module
                if hasattr(target_ema, 'update_parameters'):
                    target_ema.update_parameters(raw_unet)
                elif hasattr(target_ema, 'update'):
                    target_ema.update()


            epoch_loss += loss.item()
            step += 1
            progress_bar.set_postfix({"MSE Loss": f"{loss.item():.4f}"})
            if max_steps is not None and step >= max_steps:
                print(f"Raggiunto limite di {max_steps} step per epoca.")
                break

        # Riduce il learning rate
        scheduler.step()
        avg_loss = epoch_loss / max(1, step)
        print(f"Epoch {epoch+1} completata | Loss Media: {avg_loss:.4f}")

        # Validation loss tracking 
        avg_val_loss = None
        if val_loader is not None and len(val_loader) > 0:
            # Disattiva meccanismi come il dropout e congelando le statistiche dei layer di normalizzazione
            unet.eval()
            text_encoder.eval()
            val_loss = 0.0
            val_step = 0
            # Disattivazione del tracciamento delle operazioni per prevenire l'accumulo in memoria dei tensori intermedi
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

                        val_v_target = forward_process.get_velocity(val_images, val_noise, val_timesteps)

                        val_pred_v = unet(val_noisy_images, val_timesteps, val_context, mask=val_mask)
                        val_step_loss = criterion(val_pred_v, val_v_target)

                    val_loss += val_step_loss.item()
                    val_step += 1
                    if max_steps is not None and val_step >= max_steps:
                        break

            avg_val_loss = val_loss / max(1, val_step)
            print(f"Epoch {epoch+1} | Validation Loss Media: {avg_val_loss:.4f}")
            unet.train()
            text_encoder.train()

        # Salvataggio del Checkpoint
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

        # Salvataggio pesi EMA UNet nel checkpoint
        if ema_unet is not None:
            raw_ema_sd, full_ema_sd, n_avg = extract_ema_state_dict(ema_unet)
            if raw_ema_sd is not None:
                # Dizionario dei pesi senza prefissi
                checkpoint_dict['ema_unet_state_dict'] = raw_ema_sd
            if full_ema_sd is not None:
                # Dizionario strutturato con i prefissi .module
                checkpoint_dict['ema_state_dict'] = full_ema_sd
            if n_avg is not None:
                # Valore intero del contatore
                checkpoint_dict['ema_n_averaged'] = n_avg

        torch.save(checkpoint_dict, checkpoint_path)
        print(f"Checkpoint salvato: {checkpoint_path}")

    if ema_unet is not None and hasattr(ema_unet, 'eval'):
        ema_unet.eval()

    return ema_unet