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
        config=config
    )

def evaluate(checkpoint_path: str, data_dir: str = "data", batch_size: int = 50, num_samples: int = 200):
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
        return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    unet_weights = strip_prefix(checkpoint['unet_state_dict'])
    text_encoder_weights = strip_prefix(checkpoint['text_encoder_state_dict'])

    unet.load_state_dict(unet_weights)

    # Gestione retrocompatibilita dimensioni vocabolario tra checkpoint e tokenizer attuale
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

    # 4. Loop di Valutazione per ciascuna partizione
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

                # Reverse sampling loop con intermediate clipping
                h, w = config.resolution
                x = torch.randn((curr_b, 3, h, w), device=device)
                for t_idx in reversed(range(1000)):
                    t = torch.full((curr_b,), t_idx, device=device, dtype=torch.long)
                    x = reverse_process.sample(
                        model=unet, x=x, t=t,
                        context=cond_ctx, uncond_context=uncond_ctx,
                        mask=mask, uncond_mask=uncond_mask,
                        guidance_scale=3.5, clip_denoised=True
                    )

                fake_images = x  # [-1.0, 1.0]

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
    parser.add_argument("--data_dir", type=str, default="data", help="Directory dei dati del dataset")
    args = parser.parse_args()

    ckpt = resolve_checkpoint(explicit_path=args.checkpoint)
    evaluate(ckpt, data_dir=args.data_dir, batch_size=args.batch_size, num_samples=args.num_samples)
