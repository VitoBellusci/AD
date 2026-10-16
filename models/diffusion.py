import torch
import math


class DiffusionScheduler:
    """
    Classe base per gestire lo scheduling del rumore.
    Supporta sia il Cosine Scheduler che il Linear Scheduler per gestire la varianza del rumore.
    Il Cosine Scheduler è l'approccio predefinito, introdotto per prevenire una distruzione troppo rapida 
    dell'immagine nei primi step.
    """
    def __init__(self, num_time_steps=1000, s=0.008, schedule_type="cosine", beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_time_steps = num_time_steps
        self.device = torch.device(device)
        self.schedule_type = schedule_type

        if schedule_type == "cosine":
            # Calcolo di alpha_bar usando la formula del cosine scheduler:
            # f(t) = cos(((t/T + s) / (1 + s)) * (pi/2))^2
            # L'utilizzo di steps nell'equazione, vettorializza il calcolo, permettendo di calcolare la curva base
            # per la distruzione dell'immagine simultaneamente per tutti gli istanti temporali, evitando l'uso di cicli iterativi lenti
            steps = torch.arange(num_time_steps + 1, dtype=torch.float32, device=self.device)
            f_t = torch.cos(((steps / num_time_steps + s) / (1 + s)) * (math.pi / 2))**2

            # Questo tensore viene normalizzato, assicurando che allo step t = 0 il valore sia esattamente 1.0, cioè che il processo di diffusione
            # cominci con il 100% del segnale originale intatto, senza alcuna corruzione preventiva.
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
            # Qui beta_t può avvicinarsi a 1.0, causando instabilità numeriche o singolarità
            self.betas = torch.clamp(betas, max=0.999).to(self.device)

            # Ricalcolo di alphas e alpha_bars dopo il clipping per coerenza matematica
            self.alphas = 1 - self.betas
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        elif schedule_type == "linear":
            # Linear beta schedule
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

        # Pre-calcolo per la formulazione corretta di reverse sampling con clipping di x_0.
        # Si costruisce il tensore shiftato concatenando il valore 1.0 all'inizio della sequenza alpha_bars.
        # Questi coefficienti sono necessari per ricondizionare l'immagine denoised

        """
        Durante la generazione, ricondizionare significa calcolare matematicamente lo stato intermedio corretto x_(t-1) fondendo la previsione
        dell'immagine finale completamente pulita con lo stato rumoroso attuale. La media a posteriori calcola un punto di equilibrio esatto.
        Nel metodo sample, coef1 tira l'immagine verso la destinazione finala stimata dal modello neurale, mentre il secondo coefficiente ancora
        l'immagine allo stato rumoroso corrente, che funge da ponte strutturale sicuro
        """
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alpha_bars[:-1]]).to(self.device)
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alpha_bars)).to(self.device)
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bars)).to(self.device)

    def _extract(self, coef_tensor, t, target_tensor):
        """
        Estrae i coefficienti al timestep t e ne effettua il broadcast
        compatibile con la dimensione e il tipo di target_tensor.
        Controlla che t non sia già un tensore ed in tal caso lo trasforma 
        """
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=torch.long, device=coef_tensor.device)
        else:
            t = t.to(device=coef_tensor.device, dtype=torch.long)

        # Appiattimento del tensore per estrarre facilmente i valori tramite indicizzazione
        t = t.flatten()
        coef = coef_tensor[t].to(device=target_tensor.device, dtype=target_tensor.dtype)

        # Ciclo di broadcasting: finché il numero di dimensioni del coefficiente è minore di quello dell'immagine
        # bersaglio, aggiunge dimensioni vuote alla fine del tensore
        while coef.ndim < target_tensor.ndim:
            coef = coef.unsqueeze(-1)
        return coef

    def get_velocity(self, original, noise, t):
        """
        Calcola il target di velocita (v) per v-prediction.
        Formula: v = sqrt(alpha_bar_t) * noise - sqrt(1 - alpha_bar_t) * original
        """

        # Check device e dtype
        if noise.device != original.device or noise.dtype != original.dtype:
            noise = noise.to(device=original.device, dtype=original.dtype)
        sqrt_alpha_bar_t = self._extract(self.sqrt_alpha_bars, t, original)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alpha_bars, t, original)
        return (sqrt_alpha_bar_t * noise) - (sqrt_one_minus_alpha_bar_t * original)

    def predict_x0_from_v(self, x, v, t):
        """
        Ricostruisce x_0 dalla velocita predetta per v-parameterization, essendo un'operazione invertibile:
        x_0 = sqrt(alpha_bar_t) * x - sqrt(1 - alpha_bar_t) * v
        """
        if v.device != x.device or v.dtype != x.dtype:
            v = v.to(device=x.device, dtype=x.dtype)
        sqrt_alpha_bar_t = self._extract(self.sqrt_alpha_bars, t, x)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alpha_bars, t, x)
        return (sqrt_alpha_bar_t * x) - (sqrt_one_minus_alpha_bar_t * v)


class DiffusionForwardProcess(DiffusionScheduler):

    """
    Questo metodo riceve l'immagine originale, il rumore gaussiano e il tensore t ed implementa il processo
    di corruzione dell'immagine    
    """
    def add_noise(self, original, noise, t):
        """
        Applica il rumore direttamente al tempo t partendo da x_0, invece di applicarlo iterativamente passo
        dopo passo da 0 a t.
        Formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise.device != original.device or noise.dtype != original.dtype:
            noise = noise.to(device=original.device, dtype=original.dtype)
        sqrt_alpha_bar_t = self._extract(self.sqrt_alpha_bars, t, original)
        sqrt_one_minus_alpha_bar_t = self._extract(self.sqrt_one_minus_alpha_bars, t, original)
        return (sqrt_alpha_bar_t * original) + (sqrt_one_minus_alpha_bar_t * noise)


class DiffusionReverseProcess(DiffusionScheduler):
    """
    Gestisce la rimozione del rumore nella fase di inferenza.
    """
    def sample(self, model, x, t, context=None, uncond_context=None, 
               mask=None, uncond_mask=None, guidance_scale=3.0, noise_free=False, clip_denoised=True):

        # Normalizzazione difensiva di t a 1D tensor compatibile con batch e device di x
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, device=x.device, dtype=torch.long)
        else:
            t = t.to(device=x.device, dtype=torch.long)
        t = t.flatten()

        # la funzione numel(): number of elements, conta quanti valori sono contenuti nel tensore.
        # La condizione verifica se è stato passato un singolo instante temporale per l'intera chiamata. In tal caso,
        # il vettore viene clonato per riempire un nuovo vettore della stessa lunghezza del batch.
        # E' necessario sapere a che istante temporale si trovano tutte le immagini processate nel batch
        if t.numel() == 1:
            t = t.repeat(x.shape[0])
        elif t.shape[0] != x.shape[0]:
            raise ValueError(f"Timestep tensor batch size ({t.shape[0]}) does not match input x batch size ({x.shape[0]}).")

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
                
            model_out = model(x_input, t_input, context=context_input, mask=mask_input)
            v_cond, v_uncond = torch.chunk(model_out, 2, dim=0)
            predicted_v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            predicted_v = model(x, t, context=context, mask=mask)

        # 2. Coefficienti per lo step t corrente
        beta_t = self._extract(self.betas, t, x)

        # 3. Ricostruzione di x_0 da velocita (v) predetta (v-parameterization, R2)
        pred_x0 = self.predict_x0_from_v(x, predicted_v, t)

        # 4. Dynamic Range Clipping: ancora la traiettoria al dominio dell'immagine [-1.0, 1.0]
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # 5. Calcolo della media a posteriori mu_theta da pred_x0 clippato
        coef1 = self._extract(self.posterior_mean_coef1, t, x)
        coef2 = self._extract(self.posterior_mean_coef2, t, x)
        mean = coef1 * pred_x0 + coef2 * x

        # 6. Condizione terminale per t=0 e iniezione di rumore Langevin (stabilità con clamp min=1e-20)
        if noise_free:
            return mean

        z = torch.randn_like(x)
        nonzero_mask = (t > 0).to(dtype=x.dtype)
        while nonzero_mask.ndim < x.ndim:
            nonzero_mask = nonzero_mask.unsqueeze(-1)
        sigma_t = torch.sqrt(torch.clamp(beta_t, min=1e-20))
        return mean + nonzero_mask * sigma_t * z
