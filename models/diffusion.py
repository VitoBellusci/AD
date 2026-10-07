import torch
import math


class DiffusionScheduler:
    """
    Classe base per gestire lo scheduling del rumore.
    Supporta sia il Cosine Scheduler (Nichol & Dhariwal, 2021) che il Linear Scheduler (Ho et al., 2020)
    per gestire la varianza del rumore.
    """
    def __init__(self, num_time_steps=1000, s=0.008, schedule_type="cosine", beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_time_steps = num_time_steps
        self.device = torch.device(device)
        self.schedule_type = schedule_type

        if schedule_type == "cosine":
            # Calcolo di alpha_bar usando la formula del cosine scheduler:
            # f(t) = cos(((t/T + s) / (1 + s)) * (pi/2))^2
            steps = torch.arange(num_time_steps + 1, dtype=torch.float32, device=self.device)
            f_t = torch.cos(((steps / num_time_steps + s) / (1 + s)) * (math.pi / 2))**2

            # Questo tensore viene normalizzato, assicurando che allo step t = 0 il valore sia esattamente 1.0.
            # Esso contiene la frazione totale del segnale originale, sopravvissuta dopo t iniezioni di rumore
            ab_full = f_t / f_t[0]    

            # Prendiamo gli step da 1 a T (escludendo lo step 0)
            self.alpha_bars = ab_full[1:].to(self.device)

            # alpha_bar_t = alpha_bar_{t-1} * alpha_t  => alpha_t = alpha_bar_t / alpha_bar_{t-1}
            # Escludendo l'ultimo elemento, si prendono tutti i valori fino a T - 1
            ab_prev = ab_full[:-1].to(self.device)
            alphas = self.alpha_bars / ab_prev
            betas = 1 - alphas    # betas contiene le quote di informazione visiva cancellate

            # Clipping delle beta per evitare instabilità numeriche (singolarità vicino a t=T)
            self.betas = torch.clamp(betas, max=0.999).to(self.device)

            # Ricalcolo di alphas e alpha_bars dopo il clipping per coerenza matematica
            self.alphas = 1 - self.betas
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        elif schedule_type == "linear":
            # Linear beta schedule di Ho et al. (2020) (1e-4 a 0.02)
            self.betas = torch.linspace(beta_start, beta_end, num_time_steps, dtype=torch.float32, device=self.device)
            self.alphas = 1.0 - self.betas
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        else:
            raise ValueError(f"Tipo di schedule non supportato: '{schedule_type}'. Scegliere 'cosine' o 'linear'.")

        # Pre-calcolo per il Forward Process (add_noise)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

        # Pre-calcolo per il Reverse Process (sample)
        self.inv_sqrt_alphas = 1 / torch.sqrt(self.alphas)
        self.beta_over_sqrt_one_minus_alpha_bar = self.betas / torch.sqrt(1 - self.alpha_bars)

        # Pre-calcolo per la formulazione corretta di reverse sampling con clipping di x_0 (Ho et al. 2020 Eq. 12)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alpha_bars[:-1]]).to(self.device)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alpha_bars)).to(self.device)
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bars)).to(self.device)

class DiffusionForwardProcess(DiffusionScheduler):

    # Questo metodo riceve l'immagine originale, il rumore gaussiano e il tensore t
    def add_noise(self, original, noise, t):
        """
        Applica il rumore direttamente al tempo t partendo da x_0.
        Formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        # Assicuriamoci che i coefficienti siano sullo stesso device dell'immagine
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].to(original.device)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].to(original.device)

        # Broadcasting per [batch, channel, height, width]
        sqrt_alpha_bar_t = sqrt_alpha_bar_t[:, None, None, None]
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t[:, None, None, None]

        return (sqrt_alpha_bar_t * original) + (sqrt_one_minus_alpha_bar_t * noise)

class DiffusionReverseProcess(DiffusionScheduler):
    """
    Gestisce la rimozione del rumore (Processo di Generazione).
    """
    def sample(self, model, x, t, context=None, uncond_context=None, 
               mask=None, uncond_mask=None, guidance_scale=3.0, noise_free=False, clip_denoised=True):
        """
        Esegue un singolo step di campionamento inverso con Classifier-Free Guidance e Dynamic Range Clipping.
        """
        # 1. Classifier-Free Guidance (CFG) con concatenazione sicura delle maschere
        if guidance_scale > 1.0 and context is not None and uncond_context is not None:
            x_input = torch.cat([x, x], dim=0)
            t_input = torch.cat([t, t], dim=0)
            context_input = torch.cat([context, uncond_context], dim=0)
            
            mask_input = None
            if mask is not None and uncond_mask is not None:
                mask_input = torch.cat([mask, uncond_mask], dim=0)
            elif mask is not None:
                uncond_m = torch.ones_like(mask)
                mask_input = torch.cat([mask, uncond_m], dim=0)
                
            all_noise = model(x_input, t_input, context=context_input, mask=mask_input)
            eps_cond, eps_uncond = torch.chunk(all_noise, 2, dim=0)
            predicted_noise = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            predicted_noise = model(x, t, context=context, mask=mask)

        # 2. Coefficienti per lo step t corrente
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].to(x.device)[:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].to(x.device)[:, None, None, None]
        beta_t = self.betas[t].to(x.device)[:, None, None, None]

        # 3. Stima di x_0 pulita (Ho et al. 2020 Eq. 12 / Nichol & Dhariwal 2021)
        pred_x0 = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t

        # 4. Dynamic Range Clipping: ancora la traiettoria al dominio dell'immagine [-1.0, 1.0]
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # 5. Calcolo della media a posteriori mu_theta da pred_x0 clippato
        coef1 = self.posterior_mean_coef1[t].to(x.device)[:, None, None, None]
        coef2 = self.posterior_mean_coef2[t].to(x.device)[:, None, None, None]
        mean = coef1 * pred_x0 + coef2 * x

        # 6. Condizione terminale per t=0
        if (t == 0).all() or noise_free:
            return mean

        # 7. Iniezione di rumore Langevin (stabilità con clamp min=1e-20)
        z = torch.randn_like(x)
        sigma_t = torch.sqrt(torch.clamp(beta_t, min=1e-20))
        return mean + sigma_t * z