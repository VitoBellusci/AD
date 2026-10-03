import torch
import matplotlib.pyplot as plt
import os
from typing import List

# Import fittizi basati sui moduli precedentemente analizzati
from models.unet import Unet
from models.transformer import FullTextEncoder
from models.diffusion import DiffusionReverseProcess
from preprocessing.tokenizer import AvatarTokenizer
from preprocessing.config import PreprocessingConfig

class AvatarGenerator:
    """
    Gestisce l'inferenza del modello di diffusione, permettendo la generazione di 
    immagini da prompt testuali specificando un seed, e la valutazione di prompt OOD.
    """
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.config = PreprocessingConfig(config_path)
        
        # Inizializzazione Tokenizer
        self.tokenizer = AvatarTokenizer(self.config)
        self.tokenizer.load_vocab()
        
        # Inizializzazione Modelli (i parametri devono coincidere con quelli di training)
        vocab_size = len(self.tokenizer.vocab)
        self.text_encoder = FullTextEncoder(vocab_size=vocab_size, max_seq_len=self.config.max_seq_len).to(self.device)
        
        # Es: canali base=64, context_dim=128 (d_model del transformer)
        self.unet = Unet(in_channels=3, out_channels=3, base_channels=64, context_dim=128).to(self.device)
        
        self.reverse_process = DiffusionReverseProcess(num_time_steps=1000, device=self.device)
        
        self._load_checkpoint(checkpoint_path)
        
        self.unet.eval()
        self.text_encoder.eval()

    def _load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'])
        print(f"Checkpoint caricato con successo dall'epoca {checkpoint['epoch']}")

    def generate(self, prompt: str, seed: int, guidance_scale: float = 3.0, plot: bool = True):
        """
        Genera un'immagine a partire da un prompt e un seed specifico.
        """
        # Impostazione del seed per garantire ispezionabilità e riproducibilità
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

        # 1. Preparazione del testo condizionato
        tokens = self.tokenizer.encode(prompt)
        text_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
        pad_token_id = self.tokenizer.vocab.get("<PAD>", 0)
        mask = (text_tensor != pad_token_id).unsqueeze(1).unsqueeze(2).to(self.device)
        
        # 2. Preparazione del testo incondizionato (Classifier-Free Guidance)
        uncond_tokens = torch.full((1, self.config.max_seq_len), pad_token_id, dtype=torch.long, device=self.device)
        uncond_mask = (uncond_tokens != pad_token_id).unsqueeze(1).unsqueeze(2).to(self.device)

        with torch.no_grad():
            context = self.text_encoder(text_tensor, mask)
            uncond_context = self.text_encoder(uncond_tokens, uncond_mask)
            
            # 3. Inizializzazione dal rumore puro
            # Genera il tensore iniziale con la risoluzione corretta (es. 32x32 o 64x64)
            h, w = self.config.resolution
            x = torch.randn((1, 3, h, w), device=self.device)
            
            # 4. Reverse Sampling Loop
            for t_step in reversed(range(self.reverse_process.num_time_steps)):
                t = torch.tensor([t_step], device=self.device, dtype=torch.long)
                
                # Rimuove il rumore stocastico all'ultimo step
                noise_free = (t_step == 0)
                
                x = self.reverse_process.sample(
                    model=self.unet,
                    x=x,
                    t=t,
                    context=context,
                    uncond_context=uncond_context,
                    guidance_scale=guidance_scale,
                    noise_free=noise_free
                )
        
        # 5. Post-processing per la visualizzazione
        # Riporta il tensore dal range [-1, 1] al range [0, 1]
        img_tensor = (x.squeeze(0).cpu().clamp(-1, 1) + 1) / 2
        img_np = img_tensor.permute(1, 2, 0).numpy()

        if plot:
            plt.figure(figsize=(4, 4))
            plt.imshow(img_np)
            plt.title(f"Seed: {seed}\n{prompt[:30]}...", fontsize=9)
            plt.axis("off")
            plt.show()

        return img_tensor

    def evaluate_ood_combinations(self, ood_prompts: List[str], num_seeds: int = 4):
        """
        Ispeziona le generazioni su prompt composizionali OOD testando multipli seed.
        """
        print(f"Avvio valutazione OOD su {len(ood_prompts)} combinazioni...")
        
        for prompt in ood_prompts:
            print(f"\nGenerazione per il prompt trattenuto: '{prompt}'")
            fig, axes = plt.subplots(1, num_seeds, figsize=(3 * num_seeds, 3))
            
            # Utilizza seed diversi per lo stesso prompt
            for i in range(num_seeds):
                seed = 1000 + i 
                img_tensor = self.generate(prompt, seed=seed, plot=False)
                img_np = img_tensor.permute(1, 2, 0).numpy()
                
                axes[i].imshow(img_np)
                axes[i].set_title(f"Seed: {seed}")
                axes[i].axis("off")
                
            plt.suptitle(prompt)
            plt.tight_layout()
            plt.show()


# ==========================================
# ESEMPIO DI UTILIZZO (Script Entry Point)
# ==========================================
if __name__ == "__main__":
    # Inizializza il generatore
    # NOTA: Assicurati di avere il file JSON di configurazione e i pesi salvati
    generator = AvatarGenerator(
        config_path="./preprocessing/preprocessing_config.json", 
        checkpoint_path="checkpoints/checkpoint_epoch_22.pt"
    )
    
    # 1. Test Interattivo (Singolo Prompt, Singolo Seed)
    # L'utente può inserire un prompt e un seed personalizzati da terminale
    while True:
        user_input = input("\nInserisci un prompt per generare l'avatar (o 'exit' per uscire): ")
        if user_input.lower() == 'exit':
            break

        try:
            seed_input = input("Inserisci un seed numerico (es. 42): ")
            user_seed = int(seed_input)
        except ValueError:
            print("Seed non valido. Utilizzo seed predefinito 42.")
            user_seed = 42

        print(f"Generazione in corso per: {user_input} (Seed: {user_seed})...")
        generator.generate(prompt=user_input, seed=user_seed, guidance_scale=3.5, plot=True)

    # 2. Valutazione Out-Of-Distribution (Generalizzazione Composizionale)
    # Se vuoi ancora eseguire i test automatici OOD, decommenta le righe seguenti:
    ood_test_prompts = [
        "a blue cartoon avatar with round eyes and exaggerated proportions",
        "a red avatar with standard eyes and normal proportions"
    ]
    generator.evaluate_ood_combinations(ood_test_prompts, num_seeds=4)
