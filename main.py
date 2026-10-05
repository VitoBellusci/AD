import torch
import csv
import os
import re
import json
import glob
import argparse
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

# Import dei moduli della pipeline
from preprocessing.config import PreprocessingConfig
from preprocessing.tokenizer import AvatarTokenizer
from preprocessing.dataset import AvatarDataset
from preprocessing.splitter import CompositionalSplitter
from preprocessing.caption_generator import CaptionGenerator
from models.transformer import FullTextEncoder
from models.unet import Unet
from models.diffusion import DiffusionForwardProcess
from train import train, set_seed

def resolve_checkpoint(checkpoint_dir: str = "checkpoints", explicit_path: str = None) -> str:
    """
    Risolve il checkpoint in modo deterministico ordinando per indice numerico di epoca,
    evitando la fragilità non portabile di os.path.getctime (Blueprint 3.2, DEF-20).
    """
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")
    available = glob.glob(pattern)
    
    if not available:
        fallback = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        if not fallback:
            raise FileNotFoundError(f"Nessun file di checkpoint trovato nella cartella '{checkpoint_dir}'.")
        return fallback[0]

    def extract_epoch(path: str) -> int:
        match = re.search(r'checkpoint_epoch_(\d+)\.pt', os.path.basename(path))
        return int(match.group(1)) if match else -1

    return max(available, key=extract_epoch)

def configure_optimizers(
    unet: torch.nn.Module,
    text_encoder: torch.nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 5,
    total_epochs: int = 50
):
    """
    Configura l'ottimizzatore AdamW con weight decay disaccoppiato (Blueprint 3.1, DEF-19):
    - Parametri con ndim >= 2 ricevono weight_decay.
    - Parametri con ndim <= 1, bias e norm ricevono weight_decay = 0.0.
    Scheduler: LinearLR warmup concatenato a CosineAnnealingLR via SequentialLR.
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

    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs))
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )
    return optimizer, scheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Avatar Diffusion Training Pipeline")
    parser.add_argument("--conditional", dest="conditional", action="store_true", default=True,
                        help="Train conditional diffusion model (default: True)")
    parser.add_argument("--unconditional", dest="conditional", action="store_false",
                        help="Train unconditional baseline (DEF-11)")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--cfg_drop_rate", type=float, default=0.1, help="CFG dropout rate (default: 0.1)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Explicit checkpoint path to resume from")
    return parser

def main(args=None):
    # 1. Parse argomenti CLI e configurazione (DEF-11)
    parser = parse_args()
    if args is None:
        parsed_args, _ = parser.parse_known_args()
    elif isinstance(args, list):
        parsed_args = parser.parse_args(args)
    elif isinstance(args, argparse.Namespace):
        parsed_args = args
    else:
        parsed_args = parser.parse_args([])

    # Setup iniziale per la riproducibilità
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo in uso: {device}")

    # 2. Inizializzazione Configurazione
    config = PreprocessingConfig("./preprocessing/preprocessing_config.json")

    # CARICAMENTO METADATI GOOGLE CARTOON SET
    image_dir = "./data/cartoonset100k_jpg" 
    csv_path = "./data/meta/cartoon_image_attributes.csv"

    raw_metadata = []
    image_paths = []

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Estrae la stringa originale dal CSV (es. "0/cs12345.jpg")
            raw_filename = row.pop('filename')
            
            # Unisce il percorso base con la sottocartella e il nome file
            # Risultato: "cartoonset100k_jpg/0/cs12345.jpg"
            full_path = os.path.join(image_dir, raw_filename)
            image_paths.append(full_path)
            
            # Salva i restanti attributi come dizionario
            raw_metadata.append(row)

    # 3. Split Composizionale a 4 vie (DEF-01, DEF-09, DEF-13)
    splitter = CompositionalSplitter(config)
    train_indices, val_indices, test_ind_indices, ood_indices = splitter.split(raw_metadata)
    print(f"Partizioni create: {len(train_indices)} train, {len(val_indices)} val, "
          f"{len(test_ind_indices)} test in-distribution, {len(ood_indices)} test OOD")
    
    train_metadata = [raw_metadata[i] for i in train_indices]
    
    # 4. Generazione Didascalie e Fit del Tokenizer SOLO sul Training Set
    caption_gen = CaptionGenerator()
    train_texts = [caption_gen.generate(m) for m in train_metadata]
    
    tokenizer = AvatarTokenizer(config)
    tokenizer.fit(train_texts)
    tokenizer.save_vocab() # Salviamo il vocabolario per l'inferenza
    
    # 5. Creazione del Dataset e DataLoader con Subset e val_loader (DEF-09, DEF-10)
    dataset = AvatarDataset(
        image_paths=image_paths,
        metadata=raw_metadata,
        tokenizer=tokenizer,
        config=config
    )
    val_dataset = Subset(dataset, val_indices)
    train_dataset = Subset(dataset, train_indices)
    
    batch_size = parsed_args.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(train_dataset) >= batch_size)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 6. Inizializzazione Modelli Architetturali
    vocab_size = len(tokenizer.vocab)
    text_encoder = FullTextEncoder(
        vocab_size=vocab_size, 
        max_seq_len=config.max_seq_len,
        d_model=128
    ).to(device)
    
    unet = Unet(
        in_channels=3, 
        out_channels=3, 
        base_channels=64, 
        context_dim=128
    ).to(device)
    
    # 7. Inizializzazione Processo di Diffusione (Forward)
    forward_process = DiffusionForwardProcess(num_time_steps=1000, device=device)
    start_epoch = 0

    # 8. Inizializza l'ottimizzatore e lo scheduler con configure_optimizers (Blueprint 3.1, DEF-19)
    optimizer, scheduler = configure_optimizers(
        unet=unet,
        text_encoder=text_encoder,
        lr=parsed_args.lr,
        weight_decay=1e-4,
        warmup_epochs=5,
        total_epochs=parsed_args.epochs
    )

    # 9. Logica di ricerca e caricamento del Checkpoint con regex (Blueprint 3.2, DEF-20)
    checkpoint_dir = parsed_args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        latest_checkpoint = resolve_checkpoint(checkpoint_dir=checkpoint_dir, explicit_path=parsed_args.resume)
        print(f"Trovato checkpoint! Ripristino da: {latest_checkpoint}")
        
        # Carica il checkpoint mappandolo sul device corretto
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        
        # Ripristina i pesi della rete
        unet.load_state_dict(checkpoint['unet_state_dict'])
        text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        
        # Ripristina lo stato dell'ottimizzatore e dello scheduler
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Ripristina i seed per la perfetta riproducibilità del rumore
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
        if 'torch_cuda_rng_state' in checkpoint and checkpoint['torch_cuda_rng_state'] is not None:
            if torch.cuda.is_available():
                cuda_rng_states = [state.cpu() for state in checkpoint['torch_cuda_rng_state']]
                torch.cuda.set_rng_state_all(cuda_rng_states)           
        # Imposta l'epoca da cui ripartire
        start_epoch = checkpoint['epoch']
        print(f"Ripresa del training dall'epoca {start_epoch + 1}")
    except (FileNotFoundError, IndexError):
        print("Nessun checkpoint trovato. Inizio addestramento da zero.")

    # 10. Avvio dell'Addestramento (DEF-10, DEF-11)
    conditional_mode = parsed_args.conditional
    mode_str = "Condizionata" if conditional_mode else "Incondizionata (Baseline)"
    print(f"Avvio del loop di training - Modalità: {mode_str}...")

    train(
        unet=unet,
        text_encoder=text_encoder,
        forward_process=forward_process,
        dataloader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=parsed_args.epochs,
        device=device,
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
        conditional=conditional_mode,
        cfg_drop_rate=parsed_args.cfg_drop_rate,
        lr=parsed_args.lr
    )

if __name__ == "__main__":
    main()