import torch
import csv
import os
import json
from torch.utils.data import DataLoader

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
    

    # ---------------------------------------------------------
    # CARICAMENTO METADATI GOOGLE CARTOON SET
    # ---------------------------------------------------------
    # Punta alla cartella principale che contiene le sottocartelle 0, 1, 2...
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
    caption_gen = CaptionGenerator(config.template)
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
    )
    
    unet = Unet(
        in_channels=3, 
        out_channels=3, 
        base_channels=64, 
        context_dim=128
    )
    
    # 7. Inizializzazione Processo di Diffusione (Forward)
    forward_process = DiffusionForwardProcess(num_time_steps=1000, device=device)
    
    # 8. Avvio dell'Addestramento
    # Per eseguire la baseline incondizionata, imposta conditional=False
    print("Avvio del loop di training...")
    train(
        unet=unet,
        text_encoder=text_encoder,
        forward_process=forward_process,
        dataloader=train_loader,
        tokenizer=tokenizer,
        epochs=50,
        device=device,
        checkpoint_dir="checkpoints",
        conditional=True, # Imposta a True per il conditional model[cite: 4]
        cfg_drop_rate=0.1,
        lr=1e-4
    )

if __name__ == "__main__":
    main()