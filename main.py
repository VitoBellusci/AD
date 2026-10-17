import sys
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
from train import train, set_seed, create_ema_model, strip_prefix

def resolve_checkpoint(checkpoint_dir: str = "checkpoints", explicit_path: str = None) -> str:
    """
    Risoluzione del checkpoint in modo deterministico ordinando per indice numerico di epoca
    """
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"Checkpoint specificato non trovato: '{explicit_path}'")

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

    # Confronta i valori restituiti da extract_epoch (i numeri dell'epoca), che sono associati ad ogni percorso
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
    Configura l'ottimizzatore AdamW con weight decay disaccoppiato:
    - Parametri con ndim >= 2 ricevono weight_decay.
    - Parametri con ndim <= 1, bias e norm ricevono weight_decay = 0.0.
    Scheduler: LinearLR warmup concatenato a CosineAnnealingLR.
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

    """
    Lo scheduling del lr avviene in due fasi sequenziali:
    - Crescita lineare nei primi (grandi) passi
    - Cosine che applica una discesa fluida per raggiungere il minimo
    """
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
    parser.add_argument("--epochs", type=int, default=70, help="Total training epochs (default: 70)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--cfg_drop_rate", type=float, default=0.1, help="CFG dropout rate (default: 0.1)")
    default_ckpt_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "checkpoints"
    parser.add_argument("--checkpoint_dir", type=str, default=default_ckpt_dir, help="Directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Explicit checkpoint path to resume from")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps per epoch for fast dummy/verification run")
    parser.add_argument("--use_ema", dest="use_ema", action="store_true", default=True,
                        help="Enable Exponential Moving Average (EMA) for UNet (default: True)")
    parser.add_argument("--no_ema", dest="use_ema", action="store_false",
                        help="Disable Exponential Moving Average (EMA) for UNet")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="Exponential Moving Average decay factor (default: 0.999)")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset")
    parser.add_argument("--image_dir", type=str, default=None, help="Explicit directory for cartoonset images")
    parser.add_argument("--csv_path", type=str, default=None, help="Explicit path to cartoon_image_attributes.csv")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader num_workers (default: 4)")
    return parser

def main(args=None):
    # 1. Parse argomenti CLI e configurazione (DEF-11)
    parser = parse_args()
    if args is None:
        if "--epochs" not in sys.argv and "--max_steps" in sys.argv:
            parser.set_defaults(epochs=1)
        parsed_args, _ = parser.parse_known_args()
    elif isinstance(args, list):
        if "--epochs" not in args and "--max_steps" in args:
            parser.set_defaults(epochs=1)
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

    # CARICAMENTO METADATI GOOGLE CARTOON SET (supporto locale e Kaggle)
    csv_path = parsed_args.csv_path
    if not csv_path or not os.path.exists(csv_path):
        candidate_csvs = [
            "/kaggle/input/datasets/vitobellu/cartoon-set/meta/meta/cartoon_image_attributes.csv",
            os.path.join(parsed_args.data_dir, "meta", "cartoon_image_attributes.csv"),
            os.path.join(parsed_args.data_dir, "cartoon_image_attributes.csv"),
            "data/meta/cartoon_image_attributes.csv",
        ]
        for c in candidate_csvs:
            if os.path.exists(c):
                csv_path = c
                break

    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError("File attributi CSV non trovato. Specificare --csv_path o --data_dir.")

    image_dir = parsed_args.image_dir
    if not image_dir or not os.path.exists(image_dir):
        candidate_img_dirs = [
            "/kaggle/input/datasets/vitobellu/cartoon-set/archive/cartoonset100k_jpg",
            os.path.join(parsed_args.data_dir, "cartoonset100k_jpg"),
            parsed_args.data_dir,
            "data/cartoonset100k_jpg",
        ]
        for d in candidate_img_dirs:
            if os.path.isdir(d):
                image_dir = d
                break

    if not image_dir or not os.path.exists(image_dir):
        raise FileNotFoundError("Directory immagini non trovata. Specificare --image_dir o --data_dir.")

    raw_metadata = []
    image_paths = []

    with open(csv_path, mode='r', encoding='utf-8') as f:
        # L'utilizzo di questa classe, invece di un lettore standard, consente di interpretare automaticamente la prima riga del file CSV come
        # riga di intestazione. Di conseguenza, ogni riga successiva viene restituita come un dizionario Python, dove le chiavi sono i nomi delle colonne
        # e i valori sono i dati delle righe 
        reader = csv.DictReader(f)
        for row in reader:
            # Estrae la stringa originale dal CSV (es. "0/cs12345.jpg") e contemporaneamente rimuove la coppia dal dizionario row
            raw_filename = row.pop('filename')
            
            # Unisce il percorso base con la sottocartella e il nome file
            # Risultato: "cartoonset100k_jpg/0/cs12345.jpg"
            full_path = os.path.join(image_dir, raw_filename)
            image_paths.append(full_path)
            
            # Salva i restanti attributi come dizionario come dizionari. Ci saranno soltanto gli attributi visivi grezzi
            raw_metadata.append(row)

    # Split Composizionale a 4 vie
    splitter = CompositionalSplitter(config)
    train_indices, val_indices, test_ind_indices, ood_indices = splitter.split(raw_metadata)
    print(f"Partizioni create: {len(train_indices)} train, {len(val_indices)} val, "
          f"{len(test_ind_indices)} test in-distribution, {len(ood_indices)} test OOD")
    
    train_metadata = [raw_metadata[i] for i in train_indices]
    
    # Generazione didascalie e fit del Tokenizer solo sul training set
    caption_gen = CaptionGenerator()
    train_texts = [caption_gen.generate(m) for m in train_metadata]
    
    tokenizer = AvatarTokenizer(config)
    tokenizer.fit(train_texts)
    tokenizer.save_vocab() # Salviamo il vocabolario per l'inferenza
    
    # Creazione del dataset e dataLoader con subset e val_loader
    dataset = AvatarDataset(
        image_paths=image_paths,
        metadata=raw_metadata,
        tokenizer=tokenizer,
        config=config,
        use_ram_cache=False
    )

    # Invece di duplicare i dati in memoria istanziando dataset separati, con la classe Subset, si generare dei dataset a partire dagli indici
    # precedentemente estratti
    val_dataset = Subset(dataset, val_indices)
    train_dataset = Subset(dataset, train_indices)
    
    batch_size = parsed_args.batch_size

    use_workers = parsed_args.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(train_dataset) >= batch_size),
        num_workers=parsed_args.num_workers,
        pin_memory=(device == "cuda"),
        prefetch_factor=2 if use_workers else None, # il prefetching ordina a ciascun processo parallelo di preparare e caricare in RAM i batch successivi prima che la rete li richieda esplicitamente
        persistent_workers=use_workers  # si forza la permanenza dei processi attivi
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=parsed_args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=use_workers
    )
    
    # Inizializzazione Modelli Architetturali
    vocab_size = len(tokenizer.vocab)
    text_encoder = FullTextEncoder(
        vocab_size=vocab_size, 
        max_seq_len=config.max_seq_len,
        d_model=256,
        d_ff=512,
        num_layers=4
    ).to(device)
    
    unet = Unet(
        in_channels=3, 
        out_channels=3, 
        base_channels=96, 
        context_dim=256
    ).to(device)

    """
    Inizializzazione EMA UNet
    L'aggiornamento dei pesi può risultare caotico, motivo per cui è stato implementanta questa tecnica che mantiene una copia della reta i cui pesi
    non vengono aggiornati dai gradienti ma calcolati come media mobile dei pesi storici (si tratta di un'interpolazione lineare)
    """
    ema_unet = None
    if parsed_args.use_ema:
        # Al passo t, i pesi sono calcolati mantenendo il "decay" (es. 0.999) del valore precedente ed integrando solo la restante parte
        # dei nuovi pesi della UNet
        ema_unet = create_ema_model(unet, decay=parsed_args.ema_decay, device=device)
        print(f"EMA UNet inizializzato con decay={parsed_args.ema_decay}")

    # unet = torch.nn.DataParallel(unet)
    # text_encoder = torch.nn.DataParallel(text_encoder)
    
    # Inizializzazione Processo di Diffusione (Forward)
    forward_process = DiffusionForwardProcess(num_time_steps=1000, device=device)
    start_epoch = 0

    # Inizializza l'ottimizzatore e lo scheduler con configure_optimizers
    optimizer, scheduler = configure_optimizers(
        unet=unet,
        text_encoder=text_encoder,
        lr=parsed_args.lr,
        weight_decay=1e-4,
        warmup_epochs=5,
        total_epochs=parsed_args.epochs
    )

    # Logica di ricerca e caricamento del Checkpoint con regex
    checkpoint_dir = parsed_args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        latest_checkpoint = resolve_checkpoint(checkpoint_dir=checkpoint_dir, explicit_path=parsed_args.resume)
        print(f"Trovato checkpoint! Ripristino da: {latest_checkpoint}")
        
        # Carica il checkpoint mappandolo sul device corretto
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        
        # Ripristina i pesi della rete (rimuovendo ricorsivamente eventuale prefisso module.)
        unet_weights = strip_prefix(checkpoint['unet_state_dict'])
        text_encoder_weights = strip_prefix(checkpoint['text_encoder_state_dict'])

        raw_unet = unet
        while hasattr(raw_unet, 'module'):
            raw_unet = raw_unet.module
        raw_text_encoder = text_encoder
        while hasattr(raw_text_encoder, 'module'):
            raw_text_encoder = raw_text_encoder.module

        raw_unet.load_state_dict(unet_weights)

        # Gestione retrocompatibilità dimensioni vocabolario tra checkpoint e tokenizer attuale
        embed_key = None
        if 'embed.embedding.weight' in text_encoder_weights:
            embed_key = 'embed.embedding.weight'
        elif 'embedding.weight' in text_encoder_weights:
            embed_key = 'embedding.weight'

        if embed_key is not None:
            # Estrazione della vecchia grandezza del vocabolario
            ckpt_vocab_size = text_encoder_weights[embed_key].shape[0]
            embed_dim = text_encoder_weights[embed_key].shape[1]
            # Sfrutto la proprietà embedding per confrontare il vocabolario del vecchio file con la dimensione che il modello attualmente
            # istanziato si aspetta di ricevere
            if ckpt_vocab_size != raw_text_encoder.embedding.num_embeddings:
                raw_text_encoder.embed.embedding = torch.nn.Embedding(
                    ckpt_vocab_size, embed_dim
                ).to(device)

            # Se nei checkpoint la chiave aveva il nome obsoleto, tramite il metodo .pop() si rimuovono i pesi dalla vecchia chiave per riassegnarli alla nuova
            if embed_key == 'embedding.weight' and 'embed.embedding.weight' not in text_encoder_weights:
                text_encoder_weights['embed.embedding.weight'] = text_encoder_weights.pop('embedding.weight')

        raw_text_encoder.load_state_dict(text_encoder_weights)
        
        # Ripristina lo stato dell'ottimizzatore e dello scheduler in modo sicuro
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as opt_err:
            print(f"Avviso: Impossibile ripristinare stato ottimizzatore ({opt_err}), proseguo con nuovo ottimizzatore.")

        # Ripristino pesi EMA UNet
        if ema_unet is not None:
            # L'architettura UNet di base vera e propria
            raw_ema = ema_unet
            # Rimozione dai wrapper di parallelizzazione
            while hasattr(raw_ema, 'module'):
                raw_ema = raw_ema.module

            # Oggetto che gestirà la media mobile e che contiene il contatore degli step
            avg_model = ema_unet
            # se un checkpoint è stato salvato senza DataParallel, caricarlo in un modello avvolto da module. 
            # genererebbe un errore di incompatibilità delle chiavi.
            while hasattr(avg_model, 'module') and not hasattr(avg_model, 'n_averaged'):
                avg_model = avg_model.module

            ema_loaded = False
            # Tentativo di ripristino diretto dello stato completo AveragedModel
            if 'ema_state_dict' in checkpoint:
                try:
                    avg_model.load_state_dict(checkpoint['ema_state_dict'])
                    ema_loaded = True
                    print("Ripristinato stato EMA UNet da ema_state_dict.")
                except Exception as ema_err:
                    print(f"Avviso durante ripristino diretto ema_state_dict ({ema_err}), fallback su pesi unwrapped.")

            # Se il ripristino diretto fallisce o se è presente solo ema_unet_state_dict
            if not ema_loaded:
                weights_to_load = None
                if 'ema_unet_state_dict' in checkpoint:
                    weights_to_load = strip_prefix(checkpoint['ema_unet_state_dict'])
                    # Creazione di un nuovo dizionario senza la chiave "n_averaged". Questi pesi saranno passati alle altre archietture eccetto avg_model
                    weights_to_load = {k: v for k, v in weights_to_load.items() if k != 'n_averaged'}
                    print("Ripristino pesi EMA UNet da ema_unet_state_dict.")
                elif 'ema_state_dict' in checkpoint:
                    weights_to_load = strip_prefix(checkpoint['ema_state_dict'])
                    weights_to_load = {k: v for k, v in weights_to_load.items() if k != 'n_averaged'}
                    print("Ripristino pesi EMA UNet da ema_state_dict su modulo unwrapped.")

                if weights_to_load is not None:
                    try:
                        # Iniezione dei pesi nella rete
                        raw_ema.load_state_dict(weights_to_load)
                        ema_loaded = True
                    except Exception as load_err:
                        print(f"Avviso durante caricamento pesi EMA su modulo base ({load_err}).")

                # Ripristina il contatore n_averaged per sapere quanti aggiornamenti sono stati fatti
                if ema_loaded and hasattr(avg_model, 'n_averaged'):
                    if 'ema_n_averaged' in checkpoint:
                        n_val = checkpoint['ema_n_averaged']
                        n_int = int(n_val.item() if hasattr(n_val, 'item') else n_val)
                    elif 'ema_state_dict' in checkpoint and 'n_averaged' in checkpoint['ema_state_dict']:
                        n_val = checkpoint['ema_state_dict']['n_averaged']
                        n_int = int(n_val.item() if hasattr(n_val, 'item') else n_val)
                    else:
                        # Se non si trova da nessuna parte, viene settato a 1
                        n_int = 1

                    if isinstance(avg_model.n_averaged, torch.Tensor):
                        avg_model.n_averaged.fill_(n_int)
                    else:
                        avg_model.n_averaged = n_int

            # Fallback per checkpoint legacy senza stato EMA
            if not ema_loaded:
                # caricamento dei pesi della UNet principale. EMA non deve avere i pesi casuali dell'inizializzazione
                raw_ema.load_state_dict(raw_unet.state_dict())
                if hasattr(avg_model, 'n_averaged'):
                    if isinstance(avg_model.n_averaged, torch.Tensor):
                        avg_model.n_averaged.fill_(0)
                    else:
                        avg_model.n_averaged = 0
                print("Nessun peso EMA valido trovato nel checkpoint. Inizializzato EMA dai pesi attivi di UNet (n_averaged=0).")
        
        # Ripristina i seed per la perfetta riproducibilità del rumore
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
        # Ogni GPU mantiene un proprio stato isolato
        if 'torch_cuda_rng_state' in checkpoint and checkpoint['torch_cuda_rng_state'] is not None:
            if torch.cuda.is_available():
                # Estrazione degli stati di tutte le schede video originarie e trasferimento nella memoria RAM
                cuda_rng_states = [state.cpu() for state in checkpoint['torch_cuda_rng_state']]
                try:
                    # Check del numero di stati CUDA con il numero di GPU attualmente installate
                    if len(cuda_rng_states) == torch.cuda.device_count():
                        torch.cuda.set_rng_state_all(cuda_rng_states)
                    # Prende lo stato della prima GPU salvata
                    elif len(cuda_rng_states) > 0 and torch.cuda.device_count() > 0:
                        torch.cuda.set_rng_state(cuda_rng_states[0])
                except (RuntimeError, ValueError) as rng_err:
                    print(f"Avviso: Impossibile ripristinare CUDA RNG state ({rng_err}), proseguo con RNG predefinito.")
        # Imposta l'epoca da cui ripartire
        start_epoch = checkpoint['epoch']
        if parsed_args.resume is None and parsed_args.epochs <= start_epoch:
            print(f"Epoche richieste ({parsed_args.epochs}) <= epoca checkpoint ({start_epoch}). Avvio da epoca 0 per completare {parsed_args.epochs} epoca/che.")
            start_epoch = 0
        else:
            print(f"Ripresa del training dall'epoca {start_epoch + 1}")
    except (FileNotFoundError, IndexError):
        print("Nessun checkpoint trovato. Inizio addestramento da zero.")

    # Avvio dell'Addestramento
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
        checkpoint_dir=parsed_args.checkpoint_dir,
        start_epoch=start_epoch,
        conditional=conditional_mode,
        cfg_drop_rate=parsed_args.cfg_drop_rate,
        lr=parsed_args.lr,
        max_steps=parsed_args.max_steps,
        ema_unet=ema_unet,
        ema_decay=parsed_args.ema_decay,
        use_ema=parsed_args.use_ema
    )

if __name__ == "__main__":
    main()