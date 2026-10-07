import os
import glob
import re
import argparse
from typing import List
import torch
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from models.unet import Unet
from models.transformer import FullTextEncoder
from models.diffusion import DiffusionReverseProcess
from preprocessing.tokenizer import AvatarTokenizer
from preprocessing.config import PreprocessingConfig

def resolve_checkpoint(checkpoint_dir: str = "checkpoints", explicit_path: str = None) -> str:
    """
    Risolve il checkpoint in modo deterministico ordinando per indice numerico di epoca (DEF-20),
    evitando la fragilità di os.path.getctime e prevenendo crash su percorsi mancanti (DEF-08).
    """
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        raise FileNotFoundError(f"Checkpoint specificato non trovato: '{explicit_path}'")

    if not os.path.exists(checkpoint_dir):
        return None

    pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")
    available = glob.glob(pattern)
    if not available:
        fallback = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        if not fallback:
            return None
        return fallback[0]

    def extract_epoch(path: str) -> int:
        match = re.search(r'checkpoint_epoch_(\d+)\.pt', os.path.basename(path))
        return int(match.group(1)) if match else -1

    return max(available, key=extract_epoch)

class AvatarGenerator:
    """
    Gestisce l'inferenza del modello di diffusione, permettendo la generazione di 
    immagini da prompt testuali specificando un seed, e la valutazione di prompt OOD.
    Supporta campionamento accelerato deterministico DDIM (Song et al., 2020) per inferenza rapida.
    """
    def __init__(self, config_path: str, checkpoint_path: str = None, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.config = PreprocessingConfig(config_path)

        # Inizializzazione Tokenizer con guardia contro vocab.json mancante
        self.tokenizer = AvatarTokenizer(self.config)
        vocab_path = getattr(self.config, "vocab_path", "preprocessing/vocab.json")
        if os.path.exists(vocab_path):
            self.tokenizer.load_vocab(vocab_path)
        else:
            print(f"Avviso: File vocabolario '{vocab_path}' non trovato. Utilizzo token speciali di base.")

        # Inizializzazione Modelli Architetturali (parametri conformi al budget Tiny da 8.56M parametri)
        vocab_size = len(self.tokenizer.vocab)
        self.text_encoder = FullTextEncoder(
            vocab_size=vocab_size,
            max_seq_len=self.config.max_seq_len,
            d_model=256,
            d_ff=512,
            num_layers=4
        ).to(self.device)

        self.unet = Unet(
            in_channels=3,
            out_channels=3,
            base_channels=96,
            context_dim=256
        ).to(self.device)

        self.reverse_process = DiffusionReverseProcess(num_time_steps=1000, device=self.device)

        # Caricamento Checkpoint sicuro (supporta assenza per test/benchmark)
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
        else:
            if checkpoint_path:
                print(f"Avviso: Checkpoint specificato '{checkpoint_path}' non trovato.")
            print("Avviso: Inizializzazione del modello con pesi casuali per test/inferenza.")

        self.unet.eval()
        self.text_encoder.eval()

    def _load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        def strip_prefix(state_dict):
            return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            
        unet_weights = strip_prefix(checkpoint['unet_state_dict'])
        text_encoder_weights = strip_prefix(checkpoint['text_encoder_state_dict'])
        
        self.unet.load_state_dict(unet_weights)

        # Gestione retrocompatibilita dimensioni vocabolario tra checkpoint e tokenizer attuale
        embed_key = None
        if 'embed.embedding.weight' in text_encoder_weights:
            embed_key = 'embed.embedding.weight'
        elif 'embedding.weight' in text_encoder_weights:
            embed_key = 'embedding.weight'

        if embed_key is not None:
            ckpt_vocab_size = text_encoder_weights[embed_key].shape[0]
            embed_dim = text_encoder_weights[embed_key].shape[1]
            if ckpt_vocab_size != self.text_encoder.embedding.num_embeddings:
                self.text_encoder.embed.embedding = torch.nn.Embedding(
                    ckpt_vocab_size, embed_dim
                ).to(self.device)

            if embed_key == 'embedding.weight' and 'embed.embedding.weight' not in text_encoder_weights:
                text_encoder_weights['embed.embedding.weight'] = text_encoder_weights.pop('embedding.weight')

        self.text_encoder.load_state_dict(text_encoder_weights)
        epoch_str = checkpoint.get('epoch', 'N/A')
        print(f"Checkpoint caricato con successo da '{path}' (epoca: {epoch_str})")

    def generate(
        self,
        prompt: str = "a cartoon avatar with brown skin, wavy hair, dark eyes, no glasses, and full beard",
        seed: int = 42,
        guidance_scale: float = 3.5,
        num_steps: int = 50,
        batch_size: int = 1,
        plot: bool = False,
        output_dir: str = None
    ) -> torch.Tensor:
        """
        Genera un'immagine o un batch di immagini da prompt testuale.
        Se num_steps < 1000, esegue campionamento deterministico DDIM accelerato.
        """
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

        # 1. Preparazione del testo condizionato
        tokens = self.tokenizer.encode(prompt)
        max_valid_id = self.text_encoder.embedding.num_embeddings - 1
        pad_token_id = min(self.tokenizer.vocab.get("<PAD>", 0), max_valid_id)
        unk_token_id = min(self.tokenizer.vocab.get("<UNK>", 1), max_valid_id)
        tokens = [t if 0 <= t <= max_valid_id else unk_token_id for t in tokens]
        text_tensor = torch.tensor([tokens] * batch_size, dtype=torch.long, device=self.device)
        mask = (text_tensor != pad_token_id).unsqueeze(1).unsqueeze(2).to(self.device)

        # 2. Preparazione del testo incondizionato (Classifier-Free Guidance)
        uncond_tokens = torch.full(
            (batch_size, self.config.max_seq_len),
            pad_token_id,
            dtype=torch.long,
            device=self.device
        )
        uncond_mask = torch.ones_like(mask)

        with torch.no_grad():
            context = self.text_encoder(text_tensor, mask)
            uncond_context = self.text_encoder(uncond_tokens, uncond_mask)

            # 3. Inizializzazione dal rumore puro
            h, w = self.config.resolution
            x = torch.randn((batch_size, 3, h, w), device=self.device)

            # 4. Reverse Sampling Loop
            total_timesteps = self.reverse_process.num_time_steps

            if num_steps >= total_timesteps:
                # Campionamento DDPM standard su 1000 step
                for t_step in reversed(range(total_timesteps)):
                    t = torch.full((batch_size,), t_step, device=self.device, dtype=torch.long)
                    noise_free = (t_step == 0)
                    x = self.reverse_process.sample(
                        model=self.unet,
                        x=x,
                        t=t,
                        context=context,
                        uncond_context=uncond_context,
                        mask=mask,
                        uncond_mask=uncond_mask,
                        guidance_scale=guidance_scale,
                        noise_free=noise_free,
                        clip_denoised=True
                    )
            else:
                # Campionamento accelerato DDIM (Song et al., 2020)
                timesteps = torch.linspace(0, total_timesteps - 1, steps=num_steps).long().to(self.device)
                reversed_timesteps = timesteps.flip(0)

                for i in range(num_steps):
                    t_val = reversed_timesteps[i].item()
                    t = torch.full((batch_size,), t_val, device=self.device, dtype=torch.long)
                    t_prev_val = reversed_timesteps[i + 1].item() if (i + 1 < num_steps) else None

                    # CFG pass
                    if guidance_scale > 1.0:
                        x_input = torch.cat([x, x], dim=0)
                        t_input = torch.cat([t, t], dim=0)
                        context_input = torch.cat([context, uncond_context], dim=0)
                        mask_input = torch.cat([mask, uncond_mask], dim=0)
                        all_noise = self.unet(x_input, t_input, context=context_input, mask=mask_input)
                        eps_cond, eps_uncond = torch.chunk(all_noise, 2, dim=0)
                        predicted_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                    else:
                        predicted_noise = self.unet(x, t, context=context, mask=mask)

                    sqrt_alpha_bar_t = self.reverse_process.sqrt_alpha_bars[t].to(self.device)[:, None, None, None]
                    sqrt_one_minus_alpha_bar_t = self.reverse_process.sqrt_one_minus_alpha_bars[t].to(self.device)[:, None, None, None]

                    # Stima x_0 con dynamic range clipping per prevenire posterizzazione (DEF-17)
                    pred_x0 = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
                    pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

                    if t_prev_val is None:
                        x = pred_x0
                    else:
                        t_prev = torch.full((batch_size,), t_prev_val, device=self.device, dtype=torch.long)
                        alpha_bar_prev = self.reverse_process.alpha_bars[t_prev].to(self.device)[:, None, None, None]
                        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
                        sqrt_one_minus_alpha_bar_prev = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev, min=0.0))
                        x = sqrt_alpha_bar_prev * pred_x0 + sqrt_one_minus_alpha_bar_prev * predicted_noise

        # 5. Normalizzazione da [-1.0, 1.0] a [0.0, 1.0]
        img_tensor = (x.clamp(-1.0, 1.0) + 1.0) / 2.0

        # Salvataggio su disco se richiesto
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for b_idx in range(batch_size):
                filename = f"sample_seed{seed}_step{num_steps}_{b_idx}.png"
                out_path = os.path.join(output_dir, filename)
                save_image(img_tensor[b_idx], out_path)
                print(f"Immagine salvata in: {out_path}")

        # Visualizzazione opzionale
        if plot:
            try:
                for b_idx in range(batch_size):
                    img_np = img_tensor[b_idx].cpu().permute(1, 2, 0).numpy()
                    plt.figure(figsize=(4, 4))
                    plt.imshow(img_np)
                    plt.title(f"Seed: {seed} | Steps: {num_steps}\n{prompt[:35]}...", fontsize=9)
                    plt.axis("off")
                    plt.show()
            except Exception as e:
                print(f"Visualizzazione grafica non disponibile: {e}")

        return img_tensor

    def evaluate_ood_combinations(
        self,
        ood_prompts: List[str] = None,
        num_seeds: int = 4,
        output_dir: str = None,
        num_steps: int = 50
    ):
        """
        Ispeziona le generazioni su prompt composizionali OOD testando multipli seed (DEF-12).
        """
        if ood_prompts is None:
            ood_prompts = [
                "a cartoon avatar with brown skin, wavy hair, dark eyes, no glasses, and full beard",
                "a cartoon avatar with wavy hair and no glasses",
                "a cartoon avatar with pale skin, wavy hair, blue eyes, no glasses, and light stubble"
            ]

        print(f"\nAvvio valutazione OOD su {len(ood_prompts)} combinazioni trattenute...")
        for p_idx, prompt in enumerate(ood_prompts):
            print(f"\nGenerazione per prompt OOD trattenuto [{p_idx + 1}/{len(ood_prompts)}]: '{prompt}'")
            for i in range(num_seeds):
                seed = 1000 + i
                _ = self.generate(
                    prompt=prompt,
                    seed=seed,
                    num_steps=num_steps,
                    batch_size=1,
                    plot=False,
                    output_dir=output_dir
                )
                print(f"  [Seed {seed}] Generazione completata con successo.")

def parse_args():
    parser = argparse.ArgumentParser(description="Avatar Diffusion Inference")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a cartoon avatar with brown skin, wavy hair, dark eyes, no glasses, and full beard",
        help="Prompt testuale per condizionare la generazione dell'avatar"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per garantire riproducibilità"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Scala per Classifier-Free Guidance (CFG)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Percorso esplicito del file di checkpoint (.pt)"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Numero di step per il campionamento accelerato DDIM (default: 50)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Numero di immagini da generare in parallelo"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Cartella di destinazione per salvare le immagini generate"
    )
    parser.add_argument(
        "--test_ood",
        action="store_true",
        default=False,
        help="Esegui la suite di valutazione composizionale Out-Of-Distribution (OOD)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    checkpoint_path = resolve_checkpoint(checkpoint_dir="checkpoints", explicit_path=args.checkpoint)
    config_path = "preprocessing/preprocessing_config.json"
    if not os.path.exists(config_path):
        config_path = "./preprocessing/preprocessing_config.json"

    print("--- Inizializzazione Avatar Diffusion Inference ---")
    generator = AvatarGenerator(
        config_path=config_path,
        checkpoint_path=checkpoint_path
    )

    if args.test_ood:
        print("\nModalità OOD attivata (--test_ood).")
        generator.evaluate_ood_combinations(
            num_seeds=4,
            output_dir=args.output_dir,
            num_steps=args.num_steps
        )
    else:
        print(f"\nGenerazione per prompt: '{args.prompt}'")
        print(f"Parametri: Seed={args.seed}, Steps={args.num_steps}, Guidance={args.guidance_scale}, Batch={args.batch_size}")
        generator.generate(
            prompt=args.prompt,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            plot=False
        )
    print("\n--- Processo di inferenza completato con successo! ---")
