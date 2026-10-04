import torch
import csv
import os
import json
from torch.utils.data import DataLoader
import glob
import torch
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

def main():
    # 1. Setup iniziale per la riproducibilità
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

    # 3. Split Composizionale
    splitter = CompositionalSplitter(config)
    train_idx, val_idx, ood_idx = splitter.split(raw_metadata)
    
    train_metadata = [raw_metadata[i] for i in train_idx]
    train_image_paths = [image_paths[i] for i in train_idx]
    
    # 4. Generazione Didascalie e Fit del Tokenizer SOLO sul Training Set
    caption_gen = CaptionGenerator()
    train_texts = [caption_gen.generate(m) for m in train_metadata]
    
    tokenizer = AvatarTokenizer(config)
    tokenizer.fit(train_texts)
    tokenizer.save_vocab() # Salviamo il vocabolario per l'inferenza
    
    # 5. Creazione del Dataset e DataLoader
    train_dataset = AvatarDataset(
        image_paths=train_image_paths,
        metadata=train_metadata,
        tokenizer=tokenizer,
        config=config
    )
    
    # Batch size consigliato: 32 o 64, a seconda del budget della T4
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    
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

    # 2. Inizializza l'ottimizzatore e lo scheduler QUI, nel main
    optimizer = optim.AdamW(list(unet.parameters()) + list(text_encoder.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 3. Logica di ricerca e caricamento del Checkpoint
    checkpoint_files = glob.glob(os.path.join('checkpoints', "*.pt"))

    if checkpoint_files:
        # Trova il file più recente in base alla data di creazione
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Trovato checkpoint! Ripristino da: {latest_checkpoint}")
        
        # Carica il checkpoint mappandolo sul device corretto (GPU)
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        
        # Ripristina i pesi della rete
        unet.load_state_dict(checkpoint['unet_state_dict'])
        text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        
        # Ripristina lo stato dell'ottimizzatore e dello scheduler
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Ripristina i seed per la perfetta riproducibilità del rumore (opzionale ma consigliato)
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
        if 'torch_cuda_rng_state' in checkpoint and checkpoint['torch_cuda_rng_state'] is not None:
            if torch.cuda.is_available():
                cuda_rng_states = [state.cpu() for state in checkpoint['torch_cuda_rng_state']]
                torch.cuda.set_rng_state_all(cuda_rng_states)           
        # Imposta l'epoca da cui ripartire
        start_epoch = checkpoint['epoch']
        print(f"Ripresa del training dall'epoca {start_epoch + 1}")
    else:
        print("Nessun checkpoint trovato. Inizio addestramento da zero.")
    # 8. Avvio dell'Addestramento
    # Per eseguire la baseline incondizionata, imposta conditional=False
    print("Avvio del loop di training...")
    train(
        unet=unet,
        text_encoder=text_encoder,
        forward_process=forward_process,
        dataloader=train_loader,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=50,
        device=device,
        checkpoint_dir="checkpoints",
        start_epoch=start_epoch,
        conditional=True, # Imposta a True per il conditional model
        cfg_drop_rate=0.1,
        lr=1e-4
    )

if __name__ == "__main__":
    main()