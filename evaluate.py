import os
import glob
import re
import csv
import argparse
import json
import torch
from torch.utils.data import DataLoader, Subset

from models.unet import Unet
from models.transformer import FullTextEncoder
from models.diffusion import DiffusionReverseProcess
from preprocessing.dataset import AvatarDataset
from preprocessing.tokenizer import AvatarTokenizer
from preprocessing.config import PreprocessingConfig
from metrics import DiffusionEvaluator

def resolve_checkpoint(checkpoint_dir: str = "checkpoints", explicit_path: str = None) -> str:
    """
    Risolve il checkpoint in modo deterministico estraendo l'indice numerico di epoca via regex.
    Supporta un percorso esplicito fornito dall'utente.
    """
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"Checkpoint esplicito non trovato: '{explicit_path}'")

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Cartella dei checkpoint '{checkpoint_dir}' non esistente.")

    def extract_epoch(p: str) -> int:
        match = re.search(r'checkpoint_epoch_(\d+)\.pt', os.path.basename(p))
        return int(match.group(1)) if match else -1

    candidates = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))
    if not candidates:
        fallback = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        if not fallback:
            raise FileNotFoundError(f"Nessun checkpoint valido trovato in '{checkpoint_dir}'")
        return fallback[0]
    return max(candidates, key=extract_epoch)

def load_eval_dataset(data_dir: str, config: PreprocessingConfig, tokenizer: AvatarTokenizer) -> AvatarDataset:
    """
    Carica il dataset in modo flessibile supportando sia l'inizializzazione diretta con data_dir
    che quella basata su image_paths e metadata.
    """
    try:
        return AvatarDataset(data_dir=data_dir, config=config, tokenizer=tokenizer)
    except TypeError:
        pass

    possible_csvs = [
        os.path.join(data_dir, "meta", "cartoon_image_attributes.csv"),
        os.path.join(data_dir, "cartoon_image_attributes.csv"),
        "data/meta/cartoon_image_attributes.csv",
    ]
    possible_img_dirs = [
        os.path.join(data_dir, "cartoonset100k_jpg"),
        data_dir,
        "data/cartoonset100k_jpg",
    ]

    csv_path = None
    for p in possible_csvs:
        if os.path.exists(p):
            csv_path = p
            break

    img_dir = None
    for d in possible_img_dirs:
        if os.path.isdir(d):
            img_dir = d
            break

    raw_metadata = []
    image_paths = []
    if csv_path and os.path.exists(csv_path):
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_filename = row.pop('filename', '')
                full_path = os.path.join(img_dir or "", raw_filename)
                image_paths.append(full_path)
                raw_metadata.append(row)

    return AvatarDataset(
        image_paths=image_paths,
        metadata=raw_metadata,
        tokenizer=tokenizer,
        config=config,
        use_ram_cache=False
    )

def sample_batch(
    unet,
    reverse_process,
    cond_ctx,
    uncond_ctx,
    mask=None,
    uncond_mask=None,
    shape=None,
    device="cpu",
    num_steps=50,
    guidance_scale=3.5
):
    """
    Reverse sampling con supporto sia per DDPM standard (1000 step)
    che per campionamento deterministico accelerato DDIM (< 1000 step).
    """
    batch_size = shape[0]
    total_timesteps = reverse_process.num_time_steps
    x = torch.randn(shape, device=device)

    if num_steps >= total_timesteps:
        for t_idx in reversed(range(total_timesteps)):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            noise_free = (t_idx == 0)
            x = reverse_process.sample(
                model=unet, x=x, t=t,
                context=cond_ctx, uncond_context=uncond_ctx,
                mask=mask, uncond_mask=uncond_mask,
                guidance_scale=guidance_scale, noise_free=noise_free,
                clip_denoised=True
            )
    else:
        timesteps = torch.linspace(0, total_timesteps - 1, steps=num_steps).long().to(device)
        reversed_timesteps = timesteps.flip(0)

        for i in range(num_steps):
            t_val = reversed_timesteps[i].item()
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            t_prev_val = reversed_timesteps[i + 1].item() if (i + 1 < num_steps) else None

            if guidance_scale > 1.0 and cond_ctx is not None and uncond_ctx is not None:
                x_input = torch.cat([x, x], dim=0)
                t_input = torch.cat([t, t], dim=0)
                context_input = torch.cat([cond_ctx, uncond_ctx], dim=0)
                mask_input = None
                if mask is not None and uncond_mask is not None:
                    mask_input = torch.cat([mask, uncond_mask], dim=0)
                elif mask is not None:
                    mask_input = torch.cat([mask, torch.ones_like(mask)], dim=0)
                model_out = unet(x_input, t_input, context=context_input, mask=mask_input)
                v_cond, v_uncond = torch.chunk(model_out, 2, dim=0)
                predicted_v = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                predicted_v = unet(x, t, context=cond_ctx, mask=mask)

            # Ricostruzione x_0 e rumore dalla velocity predetta (v-parameterization, R3)
            pred_x0 = reverse_process.predict_x0_from_v(x, predicted_v, t)
            predicted_noise = reverse_process.predict_noise_from_v(x, predicted_v, t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            if t_prev_val is None:
                x = pred_x0
            else:
                t_prev = torch.full((batch_size,), t_prev_val, device=device, dtype=torch.long)
                alpha_bar_prev = reverse_process._extract(reverse_process.alpha_bars, t_prev, x)
                sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
                sqrt_one_minus_alpha_bar_prev = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev, min=0.0))
                x = sqrt_alpha_bar_prev * pred_x0 + sqrt_one_minus_alpha_bar_prev * predicted_noise

    return x

def evaluate(checkpoint_path: str, data_dir: str = "data", batch_size: int = 50, num_samples: int = 200, num_steps: int = 50, use_ema: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Avvio Valutazione Rigorosa su Device: {device} ---")
    print(f"Caricamento checkpoint: {checkpoint_path}")

    # 1. Inizializzazione Configurazione e Tokenizer
    config_path = "preprocessing/preprocessing_config.json"
    if not os.path.exists(config_path):
        config_path = "./preprocessing/preprocessing_config.json"
    config = PreprocessingConfig(config_path)

    tokenizer = AvatarTokenizer(config=config)
    vocab_path = getattr(config, "vocab_path", "preprocessing/vocab.json")
    if os.path.exists(vocab_path):
        tokenizer.load_vocab(vocab_path)
    else:
        print(f"Avviso: Vocabolario non trovato in '{vocab_path}', utilizzo token speciali di default.")

    # 2. Inizializzazione e Caricamento Modelli
    unet = Unet(in_channels=3, out_channels=3, base_channels=96, context_dim=256).to(device)
    text_encoder = FullTextEncoder(
        vocab_size=len(tokenizer.vocab), 
        max_seq_len=config.max_seq_len, 
        d_model=256,
        d_ff=512,
        num_layers=4
    ).to(device)
    reverse_process = DiffusionReverseProcess(num_time_steps=1000, device=device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    def strip_prefix(state_dict):
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

    unet_loaded = False
    if use_ema:
        if 'ema_unet_state_dict' in checkpoint:
            try:
                unet_weights = strip_prefix(checkpoint['ema_unet_state_dict'])
                unet_weights = {k: v for k, v in unet_weights.items() if k != 'n_averaged'}
                unet.load_state_dict(unet_weights)
                unet_loaded = True
                print("Caricamento pesi EMA UNet (ema_unet_state_dict) per valutazione.")
            except Exception as e:
                print(f"Avviso: Caricamento ema_unet_state_dict fallito ({e}), tento fallback.")

        if not unet_loaded and 'ema_state_dict' in checkpoint:
            try:
                unet_weights = strip_prefix(checkpoint['ema_state_dict'])
                unet_weights = {k: v for k, v in unet_weights.items() if k != 'n_averaged'}
                unet.load_state_dict(unet_weights)
                unet_loaded = True
                print("Caricamento pesi EMA UNet (ema_state_dict) per valutazione.")
            except Exception as e:
                print(f"Avviso: Caricamento ema_state_dict fallito ({e}), tento fallback.")

    if not unet_loaded:
        unet_weights = strip_prefix(checkpoint['unet_state_dict'])
        unet_weights = {k: v for k, v in unet_weights.items() if k != 'n_averaged'}
        unet.load_state_dict(unet_weights)
        if use_ema:
            print("Nessun peso EMA valido trovato; caricamento pesi regolari UNet per valutazione.")
        else:
            print("Caricamento pesi regolari UNet per valutazione (--no_ema attivo).")

    # Gestione retrocompatibilita dimensioni vocabolario tra checkpoint e tokenizer attuale
    if 'text_encoder_state_dict' in checkpoint and checkpoint['text_encoder_state_dict']:
        text_encoder_weights = strip_prefix(checkpoint['text_encoder_state_dict'])
        embed_key = None
        if 'embed.embedding.weight' in text_encoder_weights:
            embed_key = 'embed.embedding.weight'
        elif 'embedding.weight' in text_encoder_weights:
            embed_key = 'embedding.weight'

        if embed_key is not None:
            ckpt_vocab_size = text_encoder_weights[embed_key].shape[0]
            embed_dim = text_encoder_weights[embed_key].shape[1]
            if ckpt_vocab_size != text_encoder.embedding.num_embeddings:
                text_encoder.embed.embedding = torch.nn.Embedding(
                    ckpt_vocab_size, embed_dim
                ).to(device)

            if embed_key == 'embedding.weight' and 'embed.embedding.weight' not in text_encoder_weights:
                text_encoder_weights['embed.embedding.weight'] = text_encoder_weights.pop('embedding.weight')

        text_encoder.load_state_dict(text_encoder_weights)
    unet.eval()
    text_encoder.eval()

    # 3. Caricamento Dataset e Partizioni (In-Distribution vs OOD)
    dataset = load_eval_dataset(data_dir=data_dir, config=config, tokenizer=tokenizer)
    splits_path = getattr(config, "splits_path", "preprocessing/splits.json")
    
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"File delle partizioni non trovato in '{splits_path}'. Assicurarsi di aver eseguito il preprocessing.")

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    splits_to_eval = {
        "Ordinary Test (In-Distribution)": splits.get("test_ind", []),
        "OOD Test (Compositional Held-Out)": splits.get("test_ood", [])
    }

    evaluator = DiffusionEvaluator(device=device)

    # 4. Metriche di Efficienza Computazionale
    print("\n==========================================")
    print("Metriche di Efficienza Computazionale:")
    print("==========================================")
    eff_metrics = evaluator.compute_efficiency_metrics(unet, text_encoder, device)
    print(f"  Parametri Totali: {eff_metrics['Total Parameters']:,}")
    print(f"  Parametri U-Net: {eff_metrics['U-Net Parameters']:,}")
    print(f"  Parametri Text Encoder: {eff_metrics['Text Encoder Parameters']:,}")
    print(f"  Latenza di Campionamento: {eff_metrics['Sampling Latency (s)']:.4f} s")
    print(f"  Picco VRAM: {eff_metrics['Peak VRAM Usage (MB)']:.2f} MB")

    # 5. Valutazione Diversità tra Seed (Pairwise LPIPS)
    print("\n==========================================")
    print("Valutazione Diversità tra Seed (Pairwise LPIPS):")
    print("==========================================")
    diversity_prompts = [
        "a cartoon avatar with porcelain skin, short hair, blue eyes, round glasses, and no facial hair",
        "a cartoon avatar with brown skin, wavy hair, dark eyes, no glasses, and full beard"
    ]
    diversity_scores = []
    with torch.no_grad():
        for p_idx, prompt_text in enumerate(diversity_prompts):
            tokens = tokenizer.encode(prompt_text)
            max_valid_id = text_encoder.embedding.num_embeddings - 1
            pad_id = min(tokenizer.vocab.get("<PAD>", 0), max_valid_id)
            unk_token_id = min(tokenizer.vocab.get("<UNK>", 1), max_valid_id)
            tokens = [t if 0 <= t <= max_valid_id else unk_token_id for t in tokens]

            p_tokens = torch.tensor([tokens], dtype=torch.long, device=device)
            p_mask = (p_tokens != pad_id).unsqueeze(1).unsqueeze(2).to(device)
            p_cond_ctx = text_encoder(p_tokens, p_mask)

            p_uncond_tok = torch.full_like(p_tokens, pad_id)
            p_uncond_mask = torch.ones_like(p_mask)
            p_uncond_ctx = text_encoder(p_uncond_tok, p_uncond_mask)

            seed_imgs = []
            for s in [1000, 1001, 1002, 1003]:
                torch.manual_seed(s)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(s)
                h, w = config.resolution
                gen_img = sample_batch(
                    unet=unet,
                    reverse_process=reverse_process,
                    cond_ctx=p_cond_ctx,
                    uncond_ctx=p_uncond_ctx,
                    mask=p_mask,
                    uncond_mask=p_uncond_mask,
                    shape=(1, 3, h, w),
                    device=device,
                    num_steps=num_steps
                )
                seed_imgs.append(gen_img)

            stacked_seeds = torch.cat(seed_imgs, dim=0)
            div = evaluator.compute_diversity_across_seeds(stacked_seeds)
            diversity_scores.append(div)
            print(f"  Prompt [{p_idx + 1}]: \"{prompt_text[:45]}...\" -> Pairwise LPIPS: {div:.4f}")

    if diversity_scores:
        print(f"  Diversità Media Pairwise LPIPS: {sum(diversity_scores)/len(diversity_scores):.4f}")

    # 6. Loop di Valutazione Qualitativa per ciascuna partizione (IID vs OOD)
    for split_name, indices in splits_to_eval.items():
        if len(indices) == 0:
            print(f"ATTENZIONE: Partizione '{split_name}' vuota! Salto.")
            continue

        print(f"\n==========================================")
        print(f"Valutazione su Partizione: {split_name} (Campioni totali: {len(indices)})")
        print(f"==========================================")

        eval_indices = indices[:min(num_samples, len(indices))]
        subset_loader = DataLoader(Subset(dataset, eval_indices), batch_size=batch_size, shuffle=False)

        samples_accumulated = 0

        with torch.no_grad():
            for real_images, text_tokens in subset_loader:
                real_images = real_images.to(device)
                text_tokens = text_tokens.to(device)
                curr_b = real_images.shape[0]

                # Contesto condizionato e maschera
                max_valid_id = text_encoder.embedding.num_embeddings - 1
                pad_id = min(tokenizer.vocab.get("<PAD>", 0), max_valid_id)
                unk_token_id = min(tokenizer.vocab.get("<UNK>", 1), max_valid_id)
                text_tokens = torch.where(
                    (text_tokens >= 0) & (text_tokens <= max_valid_id),
                    text_tokens,
                    unk_token_id
                )
                mask = (text_tokens != pad_id).unsqueeze(1).unsqueeze(2).to(device)
                cond_ctx = text_encoder(text_tokens, mask)

                # Contesto incondizionato per Classifier-Free Guidance
                uncond_tokens = torch.full_like(text_tokens, pad_id)
                uncond_mask = torch.ones_like(mask)
                uncond_ctx = text_encoder(uncond_tokens, uncond_mask)

                # Reverse sampling loop con DDPM / DDIM accelerato
                h, w = config.resolution
                fake_images = sample_batch(
                    unet=unet,
                    reverse_process=reverse_process,
                    cond_ctx=cond_ctx,
                    uncond_ctx=uncond_ctx,
                    mask=mask,
                    uncond_mask=uncond_mask,
                    shape=(curr_b, 3, h, w),
                    device=device,
                    num_steps=num_steps,
                    guidance_scale=3.5
                )

                # Accumulo batch obbligatorio per FID e KID prima di .compute()
                evaluator.update_quality_metrics(real_images, fake_images)
                samples_accumulated += curr_b

        # Calcolo metriche qualitative
        metrics_res = evaluator.compute_quality_metrics()
        print(f"Risultati Qualitativi ({split_name}) [Campioni: {samples_accumulated}]:")
        for k, v in metrics_res.items():
            print(f"  {k}: {v:.4f}")

    print("\n--- Valutazione Completata con Successo! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valutazione Accademica Avatar Diffusion")
    parser.add_argument("--checkpoint", type=str, default=None, help="Percorso del file checkpoint (.pt)")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size per la generazione e valutazione")
    parser.add_argument("--num_samples", type=int, default=100, help="Numero massimo di campioni da valutare per partizione")
    parser.add_argument("--num_steps", type=int, default=50, help="Numero di step di campionamento (DDIM se < 1000, DDPM se >= 1000)")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory dei dati del dataset")
    parser.add_argument(
        "--use_ema",
        dest="use_ema",
        action="store_true",
        default=True,
        help="Utilizza i pesi EMA UNet se disponibili nel checkpoint (default: True)"
    )
    parser.add_argument(
        "--no_ema",
        dest="use_ema",
        action="store_false",
        help="Disabilita l'uso dei pesi EMA e valuta i pesi regolari di UNet"
    )
    args = parser.parse_args()

    ckpt = resolve_checkpoint(explicit_path=args.checkpoint)
    evaluate(
        ckpt,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        use_ema=args.use_ema
    )
