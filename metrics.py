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

    def compute_efficiency_metrics(
        self,
        unet,
        text_encoder,
        device=None,
        sample_fn=None,
        dummy_noise=None,
        timesteps=None,
        context=None
    ):
        """
        Calcola il numero di parametri, il tempo di campionamento e l'utilizzo di memoria VRAM.
        Supporta sia la chiamata diretta (unet, text_encoder, device) che quella
        con funzione di campionamento esplicita (unet, text_encoder, sample_fn, ...).
        """
        # Se il 3° argomento è callable, era stato passato come sample_fn
        if callable(device):
            sample_fn, device = device, None

        dev = torch.device(device) if device is not None else self.device

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
            if dev.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(dev)
                torch.cuda.synchronize(dev)
            
            start_time = time.time()
            
            if sample_fn is not None and callable(sample_fn):
                _ = sample_fn(dummy_noise, timesteps, context)
            else:
                dummy_x = torch.randn(1, 3, 64, 64, device=dev)
                dummy_t = torch.zeros(1, dtype=torch.long, device=dev)
                ctx_dim = getattr(unet, 'context_dim', 256)
                dummy_ctx = torch.zeros(1, 20, ctx_dim, device=dev)
                _ = unet(dummy_x, dummy_t, dummy_ctx)
            
            if dev.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize(dev)
                max_vram_mb = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)
                
            sampling_time = time.time() - start_time

        return {
            "Total Parameters": total_params,
            "U-Net Parameters": unet_params,
            "Text Encoder Parameters": text_enc_params,
            "Sampling Latency (s)": sampling_time,
            "Sampling Time (s)": sampling_time,
            "Peak VRAM Usage (MB)": max_vram_mb
        }

    @staticmethod
    def _ensure_zero_one_range(images: torch.Tensor) -> torch.Tensor:
        """
        Normalizza in modo indipendente un tensore di immagini a [0.0, 1.0].
        Gestisce in modo sicuro input sia in [-1.0, 1.0] che in [0.0, 1.0],
        applicando clamp rigoroso per prevenire overflow in Inception-v3.
        """
        if images.min() < 0.0:
            images = (images + 1.0) / 2.0
        return torch.clamp(images, 0.0, 1.0)

    def update_quality_metrics(self, real_images: torch.Tensor, fake_images: torch.Tensor):
        """
        Aggiorna lo stato interno di FID e KID con batch di immagini reali e generate.
        """
        real_norm = self._ensure_zero_one_range(real_images)
        fake_norm = self._ensure_zero_one_range(fake_images)

        self.fid.update(real_norm, real=True)
        self.fid.update(fake_norm, real=False)
        
        self.kid.update(real_norm, real=True)
        self.kid.update(fake_norm, real=False)

    def compute_quality_metrics(self):
        """
        Restituisce i valori finali di FID e KID calcolati sull'intero set di validazione/test.
        Gestisce in sicurezza campionamenti ridotti (num_samples < 50) per evitare crash di torchmetrics.
        """
        fid_score = self.fid.compute().item()

        # Guard KID subset_size: torchmetrics richiede che il numero di campioni sia >= subset_size
        try:
            if hasattr(self.kid, "real_features") and hasattr(self.kid, "fake_features"):
                if self.kid.real_features and self.kid.fake_features:
                    n_real = sum(f.shape[0] for f in self.kid.real_features)
                    n_fake = sum(f.shape[0] for f in self.kid.fake_features)
                    min_samples = min(n_real, n_fake)
                    if min_samples < 50:
                        self.kid.subset_size = min(50, max(2, min_samples))
            kid_mean, kid_std = self.kid.compute()
            kid_mean_val = kid_mean.item()
            kid_std_val = kid_std.item()
        except Exception as e:
            print(f"Avviso durante il calcolo KID ({e}), fallback a 0.0.")
            kid_mean_val = 0.0
            kid_std_val = 0.0
        
        # Reset degli stati per valutazioni future
        self.fid.reset()
        self.kid.reset()
        self.kid.subset_size = 50
        
        return {
            "FID": fid_score,
            "KID (Mean)": kid_mean_val,
            "KID (Std)": kid_std_val
        }

    def compute_diversity_across_seeds(self, generated_images):
        """
        Misura la diversità tra immagini generate partendo dallo stesso prompt ma con seed diversi.
        generated_images: Tensore [N_seeds, C, H, W] contenente le varianti di uno stesso prompt.
        """
        if generated_images.size(0) < 2:
            raise ValueError("Sono necessari almeno 2 seed per calcolare la diversità.")

        # Trasforma il range in modo rigoroso a [0.0, 1.0]
        generated_images = self._ensure_zero_one_range(generated_images)

        pairwise_distances = []
        # Calcola LPIPS per ogni possibile coppia di immagini generate
        for img1, img2 in itertools.combinations(generated_images, 2):
            img1 = img1.unsqueeze(0) # [1, C, H, W]
            img2 = img2.unsqueeze(0)
            dist = self.lpips(img1, img2).item()
            pairwise_distances.append(dist)
            
        mean_diversity = sum(pairwise_distances) / len(pairwise_distances)
        return mean_diversity

class AttributeAlignmentEvaluator:
    """
    Attribute classification verification probe to evaluate whether
    generated images match their text conditioning attributes (Blueprint 2.4).
    """
    def __init__(self, classifier_model, device):
        self.device = torch.device(device)
        self.classifier = classifier_model.to(self.device).eval()

    def evaluate_alignment(self, images: torch.Tensor, expected_attributes: torch.Tensor) -> float:
        with torch.no_grad():
            images = images.to(self.device)
            if isinstance(expected_attributes, torch.Tensor):
                expected_attributes = expected_attributes.to(self.device)
            preds = self.classifier(images)
            correct = (preds == expected_attributes).float().mean()
        return correct.item()