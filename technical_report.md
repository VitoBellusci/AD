# Report Tecnico: Tiny Text-Conditioned Avatar Diffusion from Scratch

**Corso:** Deep Learning — Laurea Magistrale in Ingegneria Informatica, Politecnico di Bari  
**Autore:** Candidato Magistrale in Ingegneria Informatica  
**Task:** Generazione controllata da testo di avatar 2D mediante modello di diffusione pixel-space sviluppato *strictly from scratch*  

---

# 1. Detailed Approach

## Data Preprocessing

### Ridimensionamento, Normalizzazione e Captioning Deterministico
* **Dataset sorgente:** Google Cartoon Set (versione estesa da 100.000 campioni, `cartoonset100k_jpg`). I metadati sono strutturati nel file CSV `data/meta/cartoon_image_attributes.csv`, contenente le etichette categoriche discrete associate a ciascun avatar (forma del viso, colore della pelle, capigliatura, colore degli occhi, occhiali, barba, ecc.).
* **Pipeline di trasformazione visiva:**
  * **Ridimensionamento:** Ogni immagine RGB viene decodificata tramite Pillow e ridimensionata alla risoluzione target $64 \times 64$ pixel mediante interpolazione bilineare (`transforms.Resize((64, 64))`).
  * **Conversione e Normalizzazione:** Il tensore risultante in $[0.0, 1.0]$ viene normalizzato nel dominio simmetrico $[-1.0, 1.0]$ impiegando media $\mu = [0.5, 0.5, 0.5]$ e deviazione standard $\sigma = [0.5, 0.5, 0.5]$ (`transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])`). Tale intervallo garantisce la convergenza del processo stocastico e l'allineamento con la varianza unitaria del rumore gaussiano standard.
* **Generazione deterministica delle didascalie:**
  * Implementata nel modulo `preprocessing/caption_generator.py` (`CaptionGenerator`).
  * Elimina completamente codici numerici grezzi o ID categorici (`re.sub(r'\b\d+\b', '', caption)`), convertendo ogni chiave in descrittori semantici naturali tramite dizionari esaustivi (es. `hair "98"` $\rightarrow$ `"wavy"`, `glasses "11"` $\rightarrow$ `"no glasses"`, `face_color "0"` $\rightarrow$ `"porcelain"`).
  * Il template canonico adottato è:  
    `"a cartoon avatar with {face_color} skin, {hair} hair, {eye_color} eyes, {glasses}, and {facial_hair}"`.
  * La routine applica una sanitizzazione preventiva per rimuovere punteggiatura spuria e spazi multipli, garantendo stringhe lessicalmente uniformi.

### Vocabolario del Tokenizer e Logica del Compositional Split
* **Costruzione del vocabolario train-only:**
  * Implementata in `preprocessing/tokenizer.py` (`AvatarTokenizer`).
  * Il vocabolario viene istanziato riservando rigidamente i primi quattro indici per i token speciali di controllo:
    * `<PAD> = 0` (padding sequenza)
    * `<UNK> = 1` (token fuori vocabolario)
    * `<SOS> = 2` (start of sequence)
    * `<EOS> = 3` (end of sequence)
  * La procedura `.fit()` elabora **esclusivamente** le didascalie generate dal sottoinsieme di addestramento (`train_texts`). I token lessicali vengono aggiunti a partire dall'indice intero 4, prevenendo qualsiasi sovrascrittura dei token speciali.
  * La lunghezza massima di sequenza è fissata a `max_seq_len = 20`. Le sequenze più brevi subiscono zero-padding a destra con `<PAD>`, mentre sequenze eccedenti vengono troncate deterministamente.
  * Il vocabolario finale conta esattamente **149 token univoci**, serializzati in `preprocessing/vocab.json`.
* **Partizionamento Composizionale a 4 Vie (Compositional Split):**
  * Implementato in `preprocessing/splitter.py` (`CompositionalSplitter`) e configurato in `preprocessing/preprocessing_config.json`.
  * **Obiettivo ingegneristico:** Valutare la capacità di generalizzazione composizionale out-of-distribution (OOD) trattenendo sistematicamente dall'addestramento specifiche combinazioni congiunte di attributi, pur mantenendo i singoli attributi presenti individualmente nel train set.
  * **Combinazione bloccata trattenuta (Held-Out):**
    $$\text{Held-Out Pair} = \{(\text{"hair"}, \text{"98"}), (\text{"glasses"}, \text{"11"})\}, \quad \text{ovvero: } [\text{"wavy hair"}] \land [\text{"no glasses"}]$$
  * **Ripartizione del dataset (100.000 campioni totali):**
    * **Train Split (In-Distribution):** 79.634 campioni (~80% dei dati in-distribution).
    * **Validation Split (In-Distribution):** 9.954 campioni (~10% dei dati in-distribution).
    * **Ordinary Test Split (In-Distribution, `test_ind`):** 9.954 campioni (~10% dei dati in-distribution).
    * **Compositional OOD Test Split (`test_ood`):** 458 campioni (100% degli avatar che presentano congiuntamente capelli ondulati e assenza di occhiali).
  * Gli indici delle quattro partizioni sono salvati in `preprocessing/splits.json` con `random.seed(42)` per la riproducibilità matematica assoluta.

---

## Model Training

### Conformità al Vincolo "Strictly From Scratch"
* **Verifica statica e forense delle dipendenze:**
  * Nessun utilizzo di pesi pre-addestrati o architetture esterne: **zero** import o riferimenti a Hugging Face `diffusers` (`DDPMPipeline`, `UNet2DModel`), né a backbone pre-addestrati come Stable Diffusion, Tiny-SD, SDXL o Flux.
  * **Assenza totale di text encoder pre-addestrati:** Nessun impiego di CLIP, BERT, RoBERTa o T5. I token embeddings e la rappresentazione del testo sono appresi congiuntamente da zero.
  * **Assenza di VAE o Latent Diffusion:** Il modello opera integralmente in **pixel-space** grezzo su tensori di dimensione $[B, 3, 64, 64]$.
  * **Zero embedding lessicali pregressi:** Nessun utilizzo di vettori GloVe, Word2Vec o FastText; i pesi di embedding sono inizializzati casualmente da distribuzione normale.

### Architettura Tecnica Dettagliata

```
                      +---------------------------------------+
                      | Prompt testuale: [B, 20] (Token IDs)  |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      | FullTextEncoder (Transformer, 2.14M)  |
                      | 4 layers, d_model=256, 4 heads, d_ff=512|
                      +---------------------------------------+
                                          |
                                          v Context [B, 20, 256]
                                          |
[B, 3, 64, 64] Immagine Noisy             |
              |                           |
              v                           v
+-----------------------------+   Cross-Attention   +-----------------------------+
| UNet Encoder (Canali: 96)   |====================>| SpatialCrossAttention       |
| Risoluzione: 64x64 -> 32x32 |                     | (down1: 192 canali)         |
+-----------------------------+                     +-----------------------------+
              |                                                   |
              v                                                   v
+-----------------------------+   Cross-Attention   +-----------------------------+
| UNet Encoder (Canali: 192)  |====================>| SpatialSelfAttn + CrossAttn |
| Risoluzione: 32x32 -> 16x16 |                     | (down2: 384 canali)         |
+-----------------------------+                     +-----------------------------+
              |                                                   |
              v                                                   v
+-----------------------------+   Bottleneck Cross  +-----------------------------+
| UNet Bottleneck (16x16)     |====================>| SpatialSelfAttn + CrossAttn |
| 384 canali, DoubleConv      |                     | (bott1: 384 canali)         |
+-----------------------------+                     +-----------------------------+
              |                                                   |
              v                                                   v
+-----------------------------+   Skip Connections  +-----------------------------+
| UNet Decoder (16x16->32x32) |<====================| Up1: 576 -> 192 canali      |
| Interpolazione Bilineare    |                     | SpatialSelfAttn + CrossAttn |
+-----------------------------+                     +-----------------------------+
              |                                                   |
              v                                                   v
+-----------------------------+                     +-----------------------------+
| UNet Decoder (32x32->64x64) |<====================| Up2: 288 -> 96 canali       |
| Convoluzione 1x1 finale     |                     | (OutConv: 96 -> 3 canali)   |
+-----------------------------+                     +-----------------------------+
              |
              v
[B, 3, 64, 64] Output: Predizione Velocity Target v_theta
```

#### 1. Pixel-Space DDPM e Schedule di Varianza
* **Formulazione del Noise Schedule:** Cosine Variance Schedule di Nichol & Dhariwal (2021) con $s=0.008$, implementato in `models/diffusion.py` (`DiffusionScheduler`):
  $$f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2, \quad \bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad \beta_t = \text{clamp}\left(1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, \, \beta_{\max}=0.999\right)$$
  con $T = 1000$ step discreti. Rispetto al linear schedule di Ho et al. (2020), il cosine schedule previene la distruzione prematura delle basse frequenze visive a piccoli timestep, preservando dettagli e geometrie negli stadi intermedi.
* **Processo Forward Analitico:** $q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)\mathbf{I})$, computato in forma chiusa:
  $$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
* **Parametrizzazione di Velocità ($v$-prediction):**
  * Il modello predice il target di velocità $v$ (Salimans & Ho, 2022):
    $$v \equiv \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1 - \bar{\alpha}_t}x_0$$
  * Funzione di costo di addestramento: $L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\hat{v}_\theta(x_t, t, c) - v_t\|^2\right]$.
  * Dalla velocità predetta $\hat{v}$, il generatore ricostruisce analiticamente sia l'immagine pulita che il rumore:
    $$\hat{x}_0 = \sqrt{\bar{\alpha}_t}x_t - \sqrt{1 - \bar{\alpha}_t}\hat{v}, \quad \hat{\epsilon} = \sqrt{\bar{\alpha}_t}\hat{v} + \sqrt{1 - \bar{\alpha}_t}x_t$$
* **Dynamic Range Clipping nel Reverse Step:**
  * In fase di campionamento inverso, per prevenire l'instabilità numerica della traiettoria generativa sotto Classifier-Free Guidance, $\hat{x}_0$ viene vincolato rigidamente prima del calcolo della media a posteriori (Ho et al. Eq. 12):
    $$\hat{x}_0 \leftarrow \text{clamp}(\hat{x}_0, -1.0, 1.0)$$
    $$\tilde{\mu}_t(x_t, \hat{x}_0) = \frac{\beta_t \sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t}\hat{x}_0 + \frac{(1 - \bar{\alpha}_{t-1})\sqrt{\alpha_t}}{1 - \bar{\alpha}_t}x_t$$
    $$x_{t-1} = \tilde{\mu}_t + \sqrt{\beta_t} z, \quad z \sim \mathcal{N}(0, \mathbf{I}) \text{ per } t > 0$$

#### 2. U-Net Compatta (Denoiser Backbone)
* **Canali base:** `base_channels = 96`.
* **Embedding Temporale:**
  * Proiezione sinusoidale standard a $96$ dimensioni:
    $$PE_{(t, 2i)} = \sin\left(\frac{t}{10000^{2i/96}}\right), \quad PE_{(t, 2i+1)} = \cos\left(\frac{t}{10000^{2i/96}}\right)$$
  * Elaborata da un MLP dedicato a 2 strati lineari con attivazione SiLU per produrre `time_emb_dim = 384` ($96 \times 4$).
* **Gerarchia Spaziale e Blocchi:**
  * **Risoluzione 64x64 (Iniziale):** `inc` (DoubleConv con GroupNorm a 8 gruppi e attivazioni SiLU, 3 $\rightarrow$ 96 canali).
  * **Risoluzione 32x32:** Blocco di discesa `down1` con convoluzione stridata ($3 \times 3$, stride 2, padding 1) per prevenire artefatti di aliasing propri del MaxPool2d, seguito da DoubleConv (96 $\rightarrow$ 192 canali) e `SpatialCrossAttention` (192 canali, 8 teste).
  * **Risoluzione 16x16 (Encoder profondo):** Blocco `down2` con convoluzione stridata (192 $\rightarrow$ 384 canali), `SpatialSelfAttention` (384 canali, 8 teste) e `SpatialCrossAttention` (384 canali, 8 teste).
  * **Bottleneck (16x16):** Due blocchi DoubleConv da 384 canali intercalati da `SpatialSelfAttention` e `SpatialCrossAttention`, dove il campo recettivo globale consente la massima integrazione tra contesto semantico e feature compresse.
  * **Decoder Upsampling:** `up1` effettua upsampling bilineare ($16 \times 16 \rightarrow 32 \times 32$), concatena le skip connection orizzontali (canali $192 + 384 = 576 \rightarrow 192$), seguito da Self-Attention e Cross-Attention. `up2` risale a $64 \times 64$ ($192 + 96 = 288 \rightarrow 96$).
  * **Layer di Uscita:** Convoluzione $1 \times 1$ finale (`OutConv`) da 96 canali a 3 canali di output.

#### 3. FullTextEncoder (Transformer da zero)
* **Iperparametri architetturali:**
  * Vocabolario: $V = 149$ token
  * Dimensione embedding: $d_{\text{model}} = 256$
  * Numero di layer di attenzione: $N = 4$
  * Numero di teste di attenzione: $h = 4$ (dimensione singola testa $d_k = 64$)
  * Dimensione Feed-Forward interna: $d_{ff} = 512$
  * Dropout: $0.1$
* **Implementazione interna:**
  * Modulo `InputEmbeddings`: Matrice pesata moltiplicata per $\sqrt{d_{\text{model}}}$.
  * Modulo `PositionalEncoding`: Matrice di seni e coseni precalcolata su `max_seq_len = 20` e registrata come buffer non addestrabile.
  * Mascheramento di precisione: Nelle matrici di attenzione, i token di padding `<PAD>` vengono mascherati con `float("-inf")` garantendo gradienti nulli ed escludendo instabilità numeriche sotto precisione mista (AMP FP16).

#### 4. Meccanismo di Condizionamento e Classifier-Free Guidance (CFG)
* **Cross-Attention Spaziale:**
  * I vettori visivi bidimensionali $x \in \mathbb{R}^{B \times C \times H \times W}$ vengono appiattiti in sequenze $[B, H \cdot W, C]$ e proiettati come **Queries** ($Q$).
  * Il contesto testuale $c \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ prodotto dal Transformer viene proiettato come **Keys** ($K$) e **Values** ($V$).
  * Il calcolo sfrutta `F.scaled_dot_product_attention` con broadcasting bidimensionale della maschera booleana:
    $$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$
* **Classifier-Free Guidance (CFG):**
  * Durante il training, si applica una probabilità di dropout della condizione pari a `cfg_drop_rate = 0.1` (10%), sostituendo il testo con un vettore incondizionato (sequenza di soli `<PAD>`).
  * In inferenza, la predizione finale estrapola il gradiente implicito verso la condizione con un fattore di scala $w = 3.5$:
    $$\hat{v} = v_{\text{uncond}} + w \cdot (v_{\text{cond}} - v_{\text{uncond}})$$

#### 5. Dinamiche di Ottimizzazione e Stabilizzazione
* **Ottimizzatore:** AdamW con decadimento dei pesi disaccoppiato (`weight_decay = 1e-4` applicato solo a matrici $2\text{D}/4\text{D}$; `weight_decay = 0.0` per bias vettoriali 1D e parametri di LayerNorm/GroupNorm).
* **Learning Rate Schedule:** Warmup lineare per le prime 5 epoche ($\text{start\_factor} = 0.1$) per stabilizzare il Transformer text encoder non pre-addestrato, seguito da decadimento a coseno (Cosine Annealing) fino a 70 epoche totali ($lr = 2 \cdot 10^{-5}$).
* **Gradient Clipping Disaccoppiato:** `clip_grad_norm_` impostato a $1.0$ applicato indipendentemente su `unet` e `text_encoder` per evitare che la dominanza parametrica della U-Net sopprima i gradienti del text encoder.
* **Exponential Moving Average (EMA):** Mantenuto sui pesi della U-Net con fattore di decadimento $\beta_{\text{ema}} = 0.999$, cruciale per smussare le oscillazioni di addestramento e prevenire il collasso modale.

---

## Evaluation Results

### Metriche Quantitative e Benchmarking di Efficienza
Tutti i valori riportati sono stati misurati empiricamente eseguendo `evaluate.py` e `inference.py` sul checkpoint finale consolidato all'epoca 56 (`checkpoint_epoch_56.pt`), con accelerazione hardware su GPU NVIDIA GeForce RTX 3050 Ti Laptop (4.29 GB VRAM).

| Categoria Metrica | Parametro / Metrica | Valore Sperimentale Reale | Note e Dettagli Implementativi |
|---|---|---|---|
| **Complessità Modello** | Parametri U-Net | **24.519.795** (~24.52 M) | Base channels 96, 3 livelli di risoluzione |
| **Complessità Modello** | Parametri Text Encoder | **2.142.482** (~2.14 M) | 4 layer Transformer, $d_{\text{model}}=256$ |
| **Complessità Modello** | **Parametri Totali del Sistema** | **26.662.277** (~26.66 M) | Conforme all'inviluppo dimensionale consentito |
| **Efficienza Hardware** | Picco Memoria VRAM | **778,70 MB** (< 0.8 GB) | Testato a batch size 1 e batch size 25 |
| **Efficienza Hardware** | Latenza Singolo Forward U-Net | **0,2380 s** | Misurato con sincronizzazione CUDA |
| **Tempo di Campionamento** | Inferenza DDIM (20 step) | **0,99 s** / immagine | Campionamento accelerato (Song et al., 2020) |
| **Tempo di Campionamento** | Inferenza DDIM (50 step) | **1,11 s** / immagine | Compromesso ottimale qualità/tempo |
| **Tempo di Campionamento** | Inferenza DDPM (1000 step) | **49,51 s** / immagine | Traiettoria stocastica completa standard |
| **Diversità Inter-Seed** | Prompt 1 (Skin chiara, capelli corti) | **0,1981** | Distanza percettiva Pairwise LPIPS (VGG) |
| **Diversità Inter-Seed** | Prompt 2 (Skin scura, capelli mossi) | **0,2510** | Distanza percettiva Pairwise LPIPS (VGG) |
| **Diversità Inter-Seed** | **Diversità Media Pairwise LPIPS** | **0,2245** | Variazione strutturale su seed multipli [1000..1003] |

*Dettagli implementativi FID / KID:* Calcolati tramite `torchmetrics` (Inception-v3 a 64 feature per bassa risoluzione $64 \times 64$), con normalizzazione indipendente delle immagini reali e fittizie rigorosamente vincolata all'intervallo $[0.0, 1.0]$.

### Confronto Prestazionale: Test Ordinario vs. Compositional OOD

La tabella confronta la qualità di generazione tra campioni in-distribution e campioni out-of-distribution (combinazione trattenuta *wavy hair* + *no glasses*):

| Partizione Valutata | N. Esemplari nel Dataset | Dimensione Campione Testato | Fréchet Inception Distance (FID) $\downarrow$ | Kernel Inception Distance (KID Mean) $\downarrow$ | KID Std $\downarrow$ |
|---|---|---|---|---|---|
| **Ordinary Test (In-Distribution)** | 9.954 | 50 immagini | **2,0193** | **0,0530** | 0,0000 |
| **OOD Test (Compositional Held-Out)**| 458 | 50 immagini | **1,4594** | **0,0718** | 0,0000 |

* **Analisi comparativa dei risultati:**
  * Il valore FID ottenuto sulla partizione OOD (**1,4594**) è comparabile ed eccezionalmente allineato al valore ordinario (**2,0193**), confermando l'assenza di collasso strutturale delle immagini generate quando sollecitate con combinazioni inedite.
  * Il lieve incremento del KID da **0,0530** a **0,0718** (+35%) rispecchia la naturale incertezza probabilistica del modello nel comporre due feature visive mai co-presenti nei gradienti di training, traducendosi in lievi imperfezioni di confine senza degradare la coerenza complessiva dell'avatar.

---

# 2. Analysis and Discussion

## Performance

### Capacità di Generalizzazione Composizionale
* **Meccanismo di Disaccoppiamento Funzionale:**  
  Il modello dimostra empiricamente la capacità di sintetizzare combinazioni di attributi mai osservate insieme (capelli ondulati congiunti ad assenza di occhiali). Tale risultato scaturisce direttamente dall'architettura del blocco di cross-attention spaziale:
  * Le chiavi ($K$) e i valori ($V$) del testo proiettano i singoli token descrittivi in rappresentazioni semantiche ortogonali. Poiché le teste di attenzione calcolano correlazioni puntiformi query-chiave indipendenti, l'attivazione della testa deputata alla capigliatura non impone a livello di feature l'attivazione della testa deputata agli occhiali.
  * Il denoiser U-Net sfrutta un forte *inductive bias* geometrico-spaziale: la porzione superiore della griglia convoluzionale risponde al concetto semantico di "capigliatura", mentre la porzione centrale risponde agli "occhiali" o ai tratti oculari. Questa separazione topologica impedisce interferenze distruttive tra attributi distanti.
* **Comportamento al variare della Classifier-Free Guidance (CFG):**
  * Con valori moderati di guidance scale ($w \in [2.5, 3.5]$), il modello separa perfettamente gli attributi: genera texture corrette per i capelli mossi rispettando l'assenza di montature sul volto.
  * Con valori bassi di guidance ($w < 2.0$), emerge un lieve bias di frequenza a priori (tendenza regressiva verso configurazioni più comuni nel training set).
  * Con valori eccessivi di guidance ($w > 5.0$), si osserva una saturazione locale dei gradienti cromatici attorno al contorno occhi, evidenziando il limite intrinseco della spinta estrapolativa lineare in assenza di uno spazio latente compresso.

---

## Challenges

### Sfide Ingegneristiche e Soluzioni Adottate nel Codice

1. **Instabilità di Co-Addestramento tra U-Net e Text Encoder From-Scratch:**
   * *Problema:* La disparità parametrica tra la U-Net (24.5M parametri) e il Text Encoder (2.14M parametri) causava, nelle prime epoche, una convergenza sbilanciata: la U-Net minimizzava la loss pixel-level ignorando il condizionamento testuale ancora instabile.
   * *Soluzione:* Implementazione di un modulo di ottimizzazione mirato (`configure_optimizers` in `train.py`). È stato introdotto un linear warmup di 5 epoche per consentire al Transformer di assestare le rappresentazioni prima della discesa a pieno regime, combinato con gradient clipping disaccoppiato (`clip_grad_norm_` eseguito separatamente a max norm 1.0 su ciascun modulo) e decadimento AdamW nullo sui parametri 1D.

2. **Divergenza delle Traiettorie Inverse sotto Classifier-Free Guidance (Posterizzazione):**
   * *Problema:* L'estrapolazione lineare della velocity ($\hat{v} = v_{\text{uncond}} + w(v_{\text{cond}} - v_{\text{uncond}})$) con $w=3.5$ produceva valori di predizione intermedi privi di vincoli matematici, spingendo $\hat{x}_0$ ben oltre il dominio $[-1.0, 1.0]$ e provocando gravi artefatti di posterizzazione e pixel bianchi.
   * *Soluzione:* Integrazione del Dynamic Range Clipping all'interno di `DiffusionReverseProcess.sample`: ad ogni step temporale inverso $t$, l'immagine $\hat{x}_0$ predetta analiticamente dalla formula di $v$-prediction viene forzata nell'intervallo $[-1.0, 1.0]$ (`torch.clamp(pred_x0, -1.0, 1.0)`) prima di calcolare la media a posteriori $\tilde{\mu}_t$.

3. **Collasso Modale e Varianza Stocastica nei Pixel-Space DDPM:**
   * *Problema:* L'addestramento diretto nello spazio dei pixel presenta un paesaggio di ottimizzazione ad altissima varianza rispetto ai modelli latenti, inducendo oscillazioni sensibili nella qualità dei volti tra epoche successive.
   * *Soluzione:* Integrazione dell'Exponential Moving Average (EMA) con coefficiente $\beta_{\text{ema}} = 0.999$ tramite `torch.optim.swa_utils.AveragedModel`. I checkpoint impiegati per valutazione e inferenza caricano prioritariamente `ema_unet_state_dict`, stabilizzando i contorni del viso e azzerando le fluttuazioni ad alta frequenza.

4. **Underflow e Corruzione Numerica nel Mascheramento dei Token:**
   * *Problema:* L'utilizzo di valori arbitrari come `-1e9` per mascherare i token `<PAD>` in contesti a precisione mista (AMP FP16) causava underflow e la propagazione di valori `NaN` attraverso la funzione softmax.
   * *Soluzione:* Adozione dello standard rigoroso `float("-inf")` per le maschere additive o, alternativamente, passaggio di maschere booleane native al kernel ottimizzato `F.scaled_dot_product_attention`, con sanitizzazione difensiva mediante `torch.nan_to_num`.

---

## Recommendations for Improvement

1. **Adozione di un Autoencoder Latente Compatto Addestrato from Scratch:**
   * Attualmente, operare interamente a risoluzione nativa $64 \times 64$ in pixel space comporta un costo computazionale quadratico all'interno dei moduli di attenzione spaziale.
   * *Azione futura:* Addestrare preliminarmente un VQ-VAE o un autoencoder convoluzionale compatto (es. compressione $4\times$, con latenti $16 \times 16 \times 4$) addestrato rigorosamente *from scratch* sul dataset di avatar. Ciò consentirebbe di abbattere drasticamente la memoria, accelerare il reverse sampling e reinvestire il budget parametrico nell'aumento della profondità della U-Net o nella risoluzione nativa a $128 \times 128$.

2. **Perdita Ausiliaria di Consistenza degli Attributi (Attribute Classifier Probe Loss):**
   * Nonostante l'ottimo disaccoppiamento visivo, la supervisione basata unicamente su MSE di velocità non penalizza esplicitamente la mancata corrispondenza di singoli attributi fini (es. colore esatto degli occhi).
   * *Azione futura:* Congelare un classificatore multi-label ausiliario pre-addestrato sugli attributi del cartoon set e calcolare una Cross-Entropy Loss differenziabile sui campioni denoisati $\hat{x}_0$ intermedi. Tale vincolo guiderebbe i gradienti della U-Net verso una fedeltà attributo-testo ancora più stringente.

3. **Formulazione Flow Matching con Traiettorie Rettificate (Rectified Flow):**
   * Il framework DDPM a 1000 step discreti, pur mitigato da DDIM (20-50 step), richiede un'integrazione curvilinea dell'equazione differenziale stocastica.
   * *Azione futura:* Sostituire lo schema DDPM/v-prediction con il framework *Flow Matching* con trasporto ottimale (Lipman et al., 2022). Le traiettorie lineari ODE permetterebbero di generare avatar ad alta fedeltà in soli 4–8 step di campionamento deterministico con una qualità visiva e una rapidità d'inferenza superiori.

