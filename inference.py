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
    Esegue campionamento stocastico DDPM (Ho et al., 2020), sia su catena completa (1000 step)
    sia accelerato su sotto-sequenza (eta=1.0) per qualsiasi numero di step specificato.
    """
    def __init__(self, config_path: str, checkpoint_path: str = None, device: str = None, use_ema: bool = True):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_ema = use_ema
        self.config = PreprocessingConfig(config_path)

        # Inizializzazione Tokenizer con controllo contro vocab.json mancante
        self.tokenizer = AvatarTokenizer(self.config)
        vocab_path = getattr(self.config, "vocab_path", "preprocessing/vocab.json")
        if os.path.exists(vocab_path):
            self.tokenizer.load_vocab(vocab_path)
        else:
            print(f"Avviso: File vocabolario '{vocab_path}' non trovato. Utilizzo token speciali di base.")

        # Inizializzazione Modelli Architetturali
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
        if self.use_ema:
            if 'ema_unet_state_dict' in checkpoint:
                try:
                    unet_weights = strip_prefix(checkpoint['ema_unet_state_dict'])
                    unet_weights = {k: v for k, v in unet_weights.items() if k != 'n_averaged'}
                    self.unet.load_state_dict(unet_weights)
                    unet_loaded = True
                    print("Caricamento pesi EMA UNet (ema_unet_state_dict) per inferenza.")
                except Exception as e:
                    print(f"Avviso: Caricamento ema_unet_state_dict fallito ({e}), tento fallback.")

            if not unet_loaded and 'ema_state_dict' in checkpoint:
                try:
                    unet_weights = strip_prefix(checkpoint['ema_state_dict'])
                    unet_weights = {k: v for k, v in unet_weights.items() if k != 'n_averaged'}
                    self.unet.load_state_dict(unet_weights)
                    unet_loaded = True
                    print("Caricamento pesi EMA UNet (ema_state_dict) per inferenza.")
                except Exception as e:
                    print(f"Avviso: Caricamento ema_state_dict fallito ({e}), tento fallback.")

        if not unet_loaded:
            unet_weights = strip_prefix(checkpoint['unet_state_dict'])
            unet_weights = {k: v for k, v in unet_weights.items() if k != 'n_averaged'}
            self.unet.load_state_dict(unet_weights)
            if self.use_ema:
                print("Nessun peso EMA valido trovato; caricamento pesi regolari UNet per inferenza.")
            else:
                print("Caricamento pesi regolari UNet per inferenza (--no_ema attivo).")

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
        output_dir: str = None
    ) -> torch.Tensor:
        """
        Genera un'immagine o un batch di immagini da prompt testuale.
        Esegue campionamento probabilistico stocastico DDPM (Ho et al., 2020) a prescindere dal numero
        di step specificato (catena completa se num_steps >= 1000, o sotto-sequenza stocastica eta=1.0 se num_steps < 1000).
        """
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

        # Preparazione del testo condizionato
        tokens = self.tokenizer.encode(prompt)
        max_valid_id = self.text_encoder.embedding.num_embeddings - 1
        # Serve per assicurarsi che il numero non superi mai la capienza del modello testuale e che abbia id validi
        pad_token_id = min(self.tokenizer.vocab.get("<PAD>", 0), max_valid_id)
        unk_token_id = min(self.tokenizer.vocab.get("<UNK>", 1), max_valid_id)
        tokens = [t if 0 <= t <= max_valid_id else unk_token_id for t in tokens]
        text_tensor = torch.tensor([tokens] * batch_size, dtype=torch.long, device=self.device)
        mask = (text_tensor != pad_token_id).unsqueeze(1).unsqueeze(2).to(self.device)

        # Preparazione del testo incondizionato (Classifier-Free Guidance)
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

            # Inizializzazione dal rumore puro
            h, w = self.config.resolution
            x = torch.randn((batch_size, 3, h, w), device=self.device)

            # Reverse Sampling Loop: Campionamento DDPM
            total_timesteps = self.reverse_process.num_time_steps

            # Confronto degli step richiesti con la catena completa
            if num_steps >= total_timesteps:
                # Campionamento DDPM standard su tutti i 1000 step
                for t_step in reversed(range(total_timesteps)):
                    # Allinea tutti gli elementi del batch allo step temporale corrente
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
                """
                Campionamento stocastico DDPM accelerato su sotto-sequenza
                Dato che il metodo standard è tarato per rimuovere solo 1/1000 di rumore alla volta, se provassimo
                a fare 50 step prendendo i timestep a salti di 20: al passo 980, il metodo toglie solo il rumore per arrivare a 979
                ma al passo successivo ci ritroveremmo al passo 960, con l'immagine che contiene ancora il 99.7% del rumore originario
                """
                
                # Seleziona un numero ridotto di step temporali, flippandoli. Durante l'iterazione, il sistema estrae matematicamente il valore
                # dello step corrente (t_val) e quello dello step di destinazione (t_prev_val)
                timesteps = torch.linspace(0, total_timesteps - 1, steps=num_steps).long().to(self.device)
                reversed_timesteps = timesteps.flip(0)

                for i in range(num_steps):
                    t_val = reversed_timesteps[i].item()
                    t = torch.full((batch_size,), t_val, device=self.device, dtype=torch.long)
                    t_prev_val = reversed_timesteps[i + 1].item() if (i + 1 < num_steps) else None

                    # CFG pass
                    if guidance_scale > 1.0 and context is not None and uncond_context is not None:
                        x_input = torch.cat([x, x], dim=0)
                        t_input = torch.cat([t, t], dim=0)
                        context_input = torch.cat([context, uncond_context], dim=0)
                        mask_input = None
                        if mask is not None and uncond_mask is not None:
                            mask_input = torch.cat([mask, uncond_mask], dim=0)
                        elif mask is not None:
                            mask_input = torch.cat([mask, torch.ones_like(mask)], dim=0)
                        model_out = self.unet(x_input, t_input, context=context_input, mask=mask_input)
                        v_cond, v_uncond = torch.chunk(model_out, 2, dim=0)
                        predicted_v = v_uncond + guidance_scale * (v_cond - v_uncond)
                    else:
                        predicted_v = self.unet(x, t, context=context, mask=mask)

                    # Ricostruzione x_0 e rumore dalla velocità predetta
                    pred_x0 = self.reverse_process.predict_x0_from_v(x, predicted_v, t)
                    pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

                    # Ricalcolo predicted_noise da pred_x0 clippato per coerenza numerica, a causa del taglio 
                    alpha_bar_t = self.reverse_process._extract(self.reverse_process.alpha_bars, t, x)
                    predicted_noise = (x - torch.sqrt(alpha_bar_t) * pred_x0) / torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-20))

                    # Se non siamo all'ultimo step della catena
                    if t_prev_val is None:
                        x = pred_x0
                    else:
                        t_prev = torch.full((batch_size,), t_prev_val, device=self.device, dtype=torch.long)
                        # coefficiente cumulativo del rumore per lo step precedente
                        alpha_bar_prev = self.reverse_process._extract(self.reverse_process.alpha_bars, t_prev, x)

                        # Varianza a posteriori stocastica, interpolando le informazioni del tempo corrente e precedente
                        sigma_t2 = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * (1.0 - alpha_bar_t / alpha_bar_prev)
                        sigma_t = torch.sqrt(torch.clamp(sigma_t2, min=0.0))

                        # Direzione scalata verso x_t
                        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0))

                        # Iniezione stocastica del rumore gaussiano. Il nuovo stato latente viene formato miscelando la stima di x pesata
                        # la componente di rumore direzionale e iniettando un nuovo termine di disturbo scalato
                        noise = torch.randn_like(x)
                        x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt * predicted_noise + sigma_t * noise

        # Normalizzazione da [-1.0, 1.0] a [0.0, 1.0]
        img_tensor = (x.clamp(-1.0, 1.0) + 1.0) / 2.0

        # Salvataggio su disco se richiesto
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for b_idx in range(batch_size):
                filename = f"sample_seed{seed}_step{num_steps}_{b_idx}.png"
                out_path = os.path.join(output_dir, filename)
                save_image(img_tensor[b_idx], out_path)
                print(f"Immagine salvata in: {out_path}")

        return img_tensor

    def evaluate_ood_combinations(
        self,
        ood_prompts: List[str] = None,
        num_seeds: int = 4,
        output_dir: str = None,
        num_steps: int = 50
    ):
        """
        Ispeziona le generazioni su prompt composizionali OOD testando multipli seed.
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
                    output_dir=output_dir
                )
                print(f"  [Seed {seed}] Generazione completata con successo.")

def parse_args():
    parser = argparse.ArgumentParser(description="Avatar Diffusion Inference")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a cartoon avatar with pale skin, blonde hair, dark eyes, round glasses, and no facial hair",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
    )
    parser.add_argument(
        "--test_ood",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_ema",
        dest="use_ema",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_ema",
        dest="use_ema",
        action="store_false",
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
        checkpoint_path=checkpoint_path,
        use_ema=args.use_ema
    )

    if args.test_ood:
        print("\nModalità OOD attivata (--test_ood).")
        generator.evaluate_ood_combinations(
            num_seeds=4,
            output_dir=args.output_dir,
            num_steps=args.num_steps
        )
    else:
        mode_str = "catena completa" if args.num_steps >= generator.reverse_process.num_time_steps else f"sotto-sequenza stocastica ({args.num_steps} step, eta=1.0)"
        print(f"\nGenerazione per prompt: '{args.prompt}'")
        print(f"Parametri: Seed={args.seed}, Steps={args.num_steps} [DDPM {mode_str}], Guidance={args.guidance_scale}, Batch={args.batch_size}")
        generator.generate(
            prompt=args.prompt,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
    print("\n--- Processo di inferenza completato con successo! ---")
