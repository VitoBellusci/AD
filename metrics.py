import torch
import time
import itertools
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class DiffusionEvaluator:
    """
    Classe centralizzata per il calcolo delle metriche obbligatorie del modello di diffusione.
    Gestisce l'efficienza computazionale, la qualità visiva globale e la diversità composizionale.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        
        # Metriche di Qualità Immagine (FID e KID)
        # feature=64 è un compromesso standard per immagini a bassa risoluzione (32x32 o 64x64)
        self.fid = FrechetInceptionDistance(feature=64, normalize=True).to(self.device)
        # Il subset_size del KID deve essere minore o uguale al numero di campioni forniti
        self.kid = KernelInceptionDistance(subset_size=50, normalize=True).to(self.device)
        
        # Metrica di Diversità (LPIPS usa VGG per misurare la distanza percettiva tra patch)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(self.device)

    def compute_efficiency_metrics(self, unet, text_encoder, sample_fn, dummy_noise, timesteps, context):
        """
        Calcola il numero di parametri, il tempo di campionamento e l'utilizzo di memoria VRAM.
        """
        # 1. Conteggio Parametri
        unet_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        text_enc_params = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
        total_params = unet_params + text_enc_params

        # 2. Utilizzo Memoria e Tempo di Campionamento
        max_vram_mb = 0.0
        sampling_time = 0.0

        unet.eval()
        text_encoder.eval()
        
        with torch.no_grad():
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            # Esecuzione della funzione di sampling fornita in input (es. il reverse loop completo)
            _ = sample_fn(dummy_noise, timesteps, context)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                max_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                
            sampling_time = time.time() - start_time

        return {
            "Total Parameters": total_params,
            "U-Net Parameters": unet_params,
            "Text Encoder Parameters": text_enc_params,
            "Sampling Time (s)": sampling_time,
            "Peak VRAM Usage (MB)": max_vram_mb
        }

    def update_quality_metrics(self, real_images, fake_images):
        """
        Aggiorna lo stato interno di FID e KID con nuovi batch di immagini.
        I tensori devono essere normalizzati nel range [-1, 1] o [0, 1].
        """
        # Assicuriamoci che i tensori siano nel formato atteso da normalize=True in torchmetrics (float 0.0 - 1.0)
        # Se le tue immagini escono dal range [-1, 1], riportale a [0, 1]
        if real_images.min() < 0.0:
            real_images = (real_images + 1.0) / 2.0
            fake_images = (fake_images + 1.0) / 2.0

        self.fid.update(real_images, real=True)
        self.fid.update(fake_images, real=False)
        
        self.kid.update(real_images, real=True)
        self.kid.update(fake_images, real=False)

    def compute_quality_metrics(self):
        """
        Restituisce i valori finali di FID e KID calcolati sull'intero set di validazione/test.
        """
        fid_score = self.fid.compute().item()
        kid_mean, kid_std = self.kid.compute()
        
        # Reset degli stati per valutazioni future
        self.fid.reset()
        self.kid.reset()
        
        return {
            "FID": fid_score,
            "KID (Mean)": kid_mean.item(),
            "KID (Std)": kid_std.item()
        }

    def compute_diversity_across_seeds(self, generated_images):
        """
        Misura la diversità tra immagini generate partendo dallo stesso prompt ma con seed diversi.
        generated_images: Tensore [N_seeds, C, H, W] contenente le varianti di uno stesso prompt.
        """
        if generated_images.size(0) < 2:
            raise ValueError("Sono necessari almeno 2 seed per calcolare la diversità.")

        # Trasforma il range da [-1, 1] a [0, 1] se necessario, poi a [-1, 1] richiesto da LPIPS interno,
        # ma normalize=True nel costruttore di LPIPS gestisce input [0, 1].
        if generated_images.min() < 0.0:
             generated_images = (generated_images + 1.0) / 2.0

        pairwise_distances = []
        # Calcola LPIPS per ogni possibile coppia di immagini generate
        for img1, img2 in itertools.combinations(generated_images, 2):
            img1 = img1.unsqueeze(0) # [1, C, H, W]
            img2 = img2.unsqueeze(0)
            dist = self.lpips(img1, img2).item()
            pairwise_distances.append(dist)
            
        mean_diversity = sum(pairwise_distances) / len(pairwise_distances)
        return mean_diversity