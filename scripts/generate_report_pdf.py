import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class AcademicNumberedCanvas(canvas.Canvas):
    """
    Canvas a due passate per il calcolo dinamico e la stampa del numero totale di pagine,
    con intestazione accademica superiore e piè di pagina formale.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        
        # Margini geometrici A4: larghezza 595.27 pt, altezza 841.89 pt
        # Margine sinistro/destro: 20mm (~56.7 pt)
        left = 56.7
        right = 595.27 - 56.7
        
        if self._pageNumber > 1:
            # Running Header per le pagine successive alla prima
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#4a5568"))
            self.drawString(left, 805, "Politecnico di Bari • Corso di Deep Learning • Progetto Avatar Diffusion from Scratch")
            self.drawRightString(right, 805, "Relazione Tecnica di Progetto")
            self.setStrokeColor(colors.HexColor("#cbd5e0"))
            self.setLineWidth(0.6)
            self.line(left, 799, right, 799)
        
        # Running Footer su tutte le pagine
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4a5568"))
        self.drawString(left, 36, "Corso di Laurea Magistrale in Ingegneria Informatica — Anno Accademico 2026")
        page_str = f"Pagina {self._pageNumber} di {total_pages}"
        self.drawRightString(right, 36, page_str)
        self.setStrokeColor(colors.HexColor("#cbd5e0"))
        self.setLineWidth(0.6)
        self.line(left, 46, right, 46)
        
        self.restoreState()


def build_academic_pdf(output_pdf_path: str):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=20*mm,
        rightMargin=20*mm,
        topMargin=22*mm,
        bottomMargin=22*mm
    )

    styles = getSampleStyleSheet()

    # Personalizzazione e definizione degli stili tipografici accademici
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1a365d'),
        alignment=1, # Centrato
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2b6cb0'),
        alignment=1, # Centrato
        spaceAfter=12
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#4a5568'),
        alignment=1, # Centrato
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13.5,
        leading=17,
        textColor=colors.HexColor('#1a365d'),
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1a202c'),
        alignment=4, # Giustificato
        firstLineIndent=14,
        spaceAfter=6
    )

    body_noindent = ParagraphStyle(
        'Body_NoIndent',
        parent=body_style,
        firstLineIndent=0
    )

    abstract_style = ParagraphStyle(
        'Abstract_Custom',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2d3748'),
        alignment=4,
        leftIndent=16,
        rightIndent=16,
        spaceAfter=12
    )

    story = []

    # Intestazione e Titolo Formale
    story.append(Paragraph("POLITECNICO DI BARI", meta_style))
    story.append(Paragraph("DIPARTIMENTO DI INGEGNERIA ELETTRICA E DELL'INFORMAZIONE", meta_style))
    story.append(Paragraph("CORSO DI LAUREA MAGISTRALE IN INGEGNERIA INFORMATICA — INSEGNAMENTO DI DEEP LEARNING", meta_style))
    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1a365d'), spaceBefore=2, spaceAfter=12))

    story.append(Paragraph("Tiny Text-Conditioned Avatar Diffusion from Scratch", title_style))
    story.append(Paragraph("Relazione Tecnica di Progettazione, Addestramento e Valutazione Sperimentale", subtitle_style))
    story.append(Paragraph("<b>Docente:</b> Prof. Ing. Vito Walter Anelli, Ph.D. &nbsp;&nbsp;|&nbsp;&nbsp; <b>Candidato:</b> Studente Magistrale in Ingegneria Informatica &nbsp;&nbsp;|&nbsp;&nbsp; <b>Codice Prova:</b> 2026 VI", meta_style))
    story.append(Spacer(1, 3*mm))

    # Abstract discorsivo
    abstract_text = (
        "<b>Sommario Discorsivo:</b> Il presente lavoro documenta la progettazione, l'implementazione rigorosa, "
        "l'addestramento e la disamina sperimentale di un'architettura generativa stocastica di diffusione nello spazio dei pixel, "
        "sviluppata interamente from-scratch senza alcun ricorso a pesi pregressi, modelli linguistici pre-addestrati o librerie "
        "black-box di terze parti. L'indagine scientifica persegue il duplice obiettivo di verificare la fattibilità di un sistema "
        "text-to-image compatto conforme a rigidi vincoli computazionali e di vagliarne la generalizzazione composizionale "
        "out-of-distribution (OOD) su combinazioni di attributi fisionomici deliberatamente omesse nella fase di apprendimento. "
        "L'elaborato, redatto in aderenza alle consegne formali, espone dettagliatamente l'approccio ingegneristico adottato — dal "
        "preprocessing deterministico dei dati al co-addestramento della U-Net con il Transformer linguistico sino alle metriche "
        "di benchmarking — e approfondisce criticamente le prestazioni del modello, le sfide affrontate nella stabilizzazione dei "
        "gradienti e le traiettorie di evoluzione futura."
    )
    story.append(Paragraph(abstract_text, abstract_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#cbd5e0'), spaceBefore=4, spaceAfter=10))

    # =========================================================================
    # PARTE 1: A REPORT DOCUMENTING THE DETAILED APPROACH
    # =========================================================================
    story.append(Paragraph("A Report Documenting the Detailed Approach, Including Data Preprocessing, the Model Training, and Evaluation Results", h1_style))

    # 1.1 Data Preprocessing
    story.append(Paragraph("1. Data Preprocessing e Costruzione Rigorosa del Dataset", h2_style))

    p1 = (
        "La costruzione della pipeline di preprocessing ha richiesto un controllo meticoloso dei flussi informativi al fine di "
        "rispettare la duplice prescrizione dell'assegnamento: garantire la perfetta riproducibilità matematica dell'esperimento e "
        "prevenire categoricamente qualsiasi fenomeno di data leakage statistico tra le diverse partizioni del dataset. Come corpus "
        "visivo di riferimento è stato adottato il Google Cartoon Set nella sua formulazione estesa a centomila campioni bidimensionali, "
        "archiviati alla risoluzione nativa in formato JPEG. Ai campioni iconici si affiancano i relativi metadati strutturati, raccolti nel file "
        "tabellare dei tratti distintivi in cui a ogni avatar corrispondono indici categorici discreti che ne codificano la conformazione del "
        "volto, la capigliatura, la pigmentazione cutanea, le fattezze oculari, la presenza di occhiali e la barba."
    )
    story.append(Paragraph(p1, body_style))

    p2 = (
        "L'elaborazione delle immagini prevede il caricamento RGB tramite Pillow e il conseguente ridimensionamento alla risoluzione canonica "
        "di sessantaquattro per sessantaquattro pixel mediante interpolazione bilineare. I tensori numerici risultanti, originariamente "
        "definiti nell'intervallo continuo compreso tra zero e uno, subiscono una normalizzazione simmetrica che ne trasla il dominio tra "
        "meno uno e più uno, applicando valori di media e deviazione standard entrambi fissati a zero virgola cinque per ciascun canale cromatico. "
        "Tale trasformazione risponde a una precisa motivazione stocastica: nelle formulazioni canoniche dei modelli di diffusione con predizione "
        "di rumore o di velocità, il segnale visivo deve essere centrato nell'origine e vincolato a una varianza unitaria per integrarsi in modo "
        "stabile con il rumore gaussiano standard iniettato lungo la traiettoria di forward noising."
    )
    story.append(Paragraph(p2, body_style))

    p3 = (
        "Sul versante della rappresentazione testuale, il modulo di generazione delle didascalie trasforma i codici interi dei metadati "
        "in frasi descrittive in linguaggio naturale privo di numeri o identificatori spuri. Attraverso una mappatura semantica esplicita, "
        "ogni chiave numerica viene tradotta in un attributo verbale leggibile: a titolo esemplificativo, il codice numerico novantotto associato "
        "ai capelli viene convertito nella parola inglese corrispondente a ondulato, l'indice undici degli occhiali viene tradotto nell'espressione "
        "che ne attesta l'assenza, mentre i codici della pelle vengono rimappati su gradazioni semantiche quali porcellana, chiara o scura. "
        "Il template deterministico strutturato adotta la forma canonica di un avatar con una determinata pelle, capigliatura, occhi, occhiali "
        "e barba, applicando una sanitizzazione della punteggiatura e degli spazi multipli così da produrre sequenze lessicali omogenee."
    )
    story.append(Paragraph(p3, body_style))

    p4 = (
        "La costruzione del vocabolario e del tokenizer costituisce uno dei punti cardine della metodologia strictly from-scratch. "
        "Per rispettare il vincolo di isolamento epistemico, la scansione lessicale e l'estrazione delle frequenze dei token sono state "
        "eseguite unicamente sulle didascalie appartenenti alla partizione di addestramento, escludendo totalmente i campioni di validazione "
        "e di test. Il vocabolario riserva in maniera deterministica e immutabile i primi quattro indici interi ai token speciali di sistema: "
        "il valore zero corrisponde al padding, l'indice uno rappresenta i token sconosciuti fuori vocabolario, l'indice due funge da start of sequence "
        "e l'indice tre delimita l'end of sequence. L'inserimento dei vocaboli lessicali effettivi avviene a partire dall'indice intero quattro, "
        "azzerando ogni rischio di sovrascrittura accidentale dei simboli di controllo. L'universo lessicale risultante conta esattamente "
        "centoquarantanove token univoci, sufficienti a coprire integralmente la varietà morfologica e cromatica degli avatar. La lunghezza massima "
        "delle sequenze tokenizzate è stata fissata a venti elementi: le frasi di lunghezza inferiore vengono completate a destra con token di padding, "
        "mentre le eventuali eccedenze subiscono un troncamento deterministico."
    )
    story.append(Paragraph(p4, body_style))

    p5 = (
        "Per indagare sperimentalmente la capacità di generalizzazione composizionale, il partizionamento del dataset è stato strutturato a quattro "
        "vie fondandosi sulla separazione mirata di combinazioni congiunte di attributi piuttosto che su una suddivisione puramente stocastica. "
        "Nello specifico, è stata definita una coppia di attributi bloccata e trattenuta dall'addestramento, costituita dalla co-presenza di capelli "
        "ondulati e assenza di occhiali. Entrambi i tratti distintivi compaiono regolarmente nel training set presi singolarmente — i capelli ondulati "
        "associati a svariate montature e l'assenza di occhiali combinata con capigliature lisce o ricce — ma la loro intersezione simultanea è stata "
        "completamente estromessa dalla fase di ottimizzazione. Su un totale di centomila campioni, la partizione di addestramento conta settantanovemila "
        "seicentoquarantaquattro immagini, la frazione di validazione ne assorbe novemilanovecentocinquantaquattro e la partizione di test ordinario "
        "in-distribution ne comprende parimenti novemilanovecentocinquantaquattro, pari rispettivamente all'ottanta, dieci e dieci percento della massa "
        "in-distribution. Il sottoinsieme di test composizionale out-of-distribution raccoglie la totalità dei quattrocentocinquantotto avatar che esibiscono "
        "la combinazione congiunta trattenuta. Gli indici di ripartizione sono stati serializzati stabilmente su disco, garantendo la riproducibilità "
        "esatta dell'intera suddivisione."
    )
    story.append(Paragraph(p5, body_style))

    # 1.2 Model Training
    story.append(Paragraph("2. Architettura del Modello e Processo di Addestramento Strictly From Scratch", h2_style))

    p6 = (
        "In conformità categorica alle specifiche didattiche, l'intera infrastruttura computazionale è stata sviluppata senza alcuna dipendenza "
        "da checkpoint pregressi, modelli linguistici esterni o pacchetti di generazione pronti all'uso. Non sono stati importati né invocati "
        "moduli da Diffusers, né sono stati adoperati encoder pre-addestrati quali CLIP, BERT o T5, né tantomeno autoencoder variazionali latenti "
        "o vettori lessicali pre-calcolati. Il sistema opera in modo puro e diretto nello spazio dei pixel, elaborando tensori quadridimensionali di forma "
        "pari a batch per tre canali cromatici per sessantaquattro per sessantaquattro pixel."
    )
    story.append(Paragraph(p6, body_style))

    p7 = (
        "L'elaborazione del condizionamento testuale è demandata a un FullTextEncoder realizzato da zero come un Transformer compatto a quattro strati. "
        "Ciascun layer integra un modulo di multi-head self-attention a quattro teste parallele con dimensione del modello pari a duecentocinquantasei "
        "e ampiezza del feed-forward interno pari a cinquecentododici unità, corredato da connessioni residue e strati di layer normalization scritti "
        "manualmente. La codifica posizionale adotta le formulazioni sinusoidali canoniche registrate come costanti di buffer non differenziabili, mentre "
        "la matrice di incorporamento converte gli indici lessicali in rappresentazioni dense scalate per la radice quadrata della dimensione nascosta. "
        "L'encoder testuale impegna complessivamente due milioni centoquarantaduemila quattrocentottantadue parametri allenabili, fornendo in uscita una matrice "
        "di contesto avente venti token per duecentocinquantasei dimensioni per ciascun campione del batch."
    )
    story.append(Paragraph(p7, body_style))

    p8 = (
        "Il denoiser visivo poggia su un'architettura U-Net parametrizzata su novantasei canali di base, operante su tre scale spaziali successive: "
        "sessantaquattro per sessantaquattro, trentadue per trentadue e sedici per sedici pixel nel collo di bottiglia. La rete incorpora blocchi a doppia "
        "convoluzione con normalizzazione a gruppi e non-linearità SiLU, integrati con proiezioni sinusoidali del tempo di diffusione espanse attraverso "
        "un perceptron multistrato a due livelli fino a duecentocinquantasei dimensioni. Nei livelli di risoluzione intermedia e nel bottleneck sono intercalati "
        "moduli di spatial self-attention e di spatial cross-attention. Questi ultimi consentono ai tensori bidimensionali delle feature visive di interrogare "
        "le proiezioni di chiavi e valori generate dal contesto testuale, orientando l'evoluzione dei gradienti verso la configurazione fisionomica voluta. "
        "Nel ramo di risalita, il ripristino della risoluzione spaziale avviene tramite sovracampionamento bilineare seguito dalla concatenazione delle skip "
        "connections provenienti dal percorso di discesa. La U-Net totalizza ventiquattro milioni cinquecentodiciannovemila settecentonovantacinque parametri, "
        "conducendo l'intero apparato neurale a un footprint complessivo di ventisei milioni seicentosessantaduemila duecentosettantasette parametri, pienamente "
        "compatibile con l'inviluppo dimensionale leggero stabilito per una GPU NVIDIA T4."
    )
    story.append(Paragraph(p8, body_style))

    p9 = (
        "La dinamica stocastica è governata da uno scheduler a mille passi discreti basato sulla formulazione a coseno introdotta da Nichol e Dhariwal, "
        "avente parametro di offset pari a zero virgola zero zero otto e vincolo superiore sulla varianza fissato a zero virgola novecentonovantanove per "
        "scongiurare singolarità nei pressi dell'istante terminale. Il processo di forward noising sfrutta la proprietà analitica della somma gaussiana "
        "per perturbare l'immagine pulita direttamente al timestep desiderato mediante una singola operazione vettorializzata. Sul piano dell'obiettivo "
        "di ottimizzazione, anziché ricorrere alla predizione diretta del rumore epsilon, è stata implementata la parametrizzazione a velocità nota come "
        "v-prediction, formalizzata da Salimans e Ho. Definendo il bersaglio neurale come combinazione lineare ponderata del rumore iniettato e del segnale pulito, "
        "la funzione di perdita dell'errore quadratico medio tra la velocità predetta dalla U-Net e la velocità target garantisce una stabilità numerica "
        "superiore lungo l'intera traiettoria, mitigando le asimmetrie di gradiente agli estremi dell'orizzonte temporale."
    )
    story.append(Paragraph(p9, body_style))

    p10 = (
        "L'addestramento contempla l'impiego della tecnica Classifier-Free Guidance. Durante la propagazione in avanti, con una frequenza di dropout pari "
        "al dieci percento, i token descrittivi vengono sostituiti dal contesto incondizionato generato mediante sequenze di soli token di padding, "
        "istruendo la rete a modellare sia la distribuzione condizionata sia la componente marginale incondizionata all'interno del medesimo set di pesi. "
        "La configurazione dell'ottimizzatore AdamW prevede un decadimento del peso pari a uno per dieci alla meno quattro applicato selettivamente alle matrici "
        "con dimensionalità pari o superiore a due, mentre per tutti i parametri monodimensionali — compresi i termini di bias e i coefficienti di scala e "
        "traslazione delle normalizzazioni — il weight decay è azzerato per evitare una regolarizzazione dannosa. La dinamica del tasso di apprendimento "
        "prevede una fase iniziale di riscaldamento lineare della durata di cinque epoche, indispensabile per stabilizzare il text encoder prima di raggiungere "
        "il valore di regime pari a due per dieci alla meno cinque, seguito da un decadimento continuo secondo una curva a coseno."
    )
    story.append(Paragraph(p10, body_style))

    p11 = (
        "A salvaguardia della convergenza sono stati introdotti due accorgimenti tecnici determinanti: il ritaglio disaccoppiato dei gradienti, applicato "
        "in modo indipendente alla U-Net e al text encoder con norma euclidea massima pari a uno, e l'impiego della precisione mista automatica con scaler "
        "dinamico per prevenire fenomeni di underflow nei calcoli a sedici bit. Inoltre, per contrastare le oscillazioni ad alta frequenza e il rischio "
        "di collasso modale, è stato mantenuto un modello ombra aggiornato mediante Exponential Moving Average con fattore di decadimento pari a zero virgola "
        "novecentonovantanove, i cui pesi consolidati sono stati impiegati in via prioritaria per l'inferenza e la stima delle metriche."
    )
    story.append(Paragraph(p11, body_style))

    # 1.3 Evaluation Results
    story.append(Paragraph("3. Risultati della Valutazione Sperimentale e Benchmarking", h2_style))

    p12 = (
        "La quantificazione sperimentale delle prestazioni del modello è stata condotta eseguendo i moduli dedicati di valutazione e inferenza sul "
        "checkpoint salvato al termine della sessantacinquesima epoca di addestramento, avvalendosi dell'accelerazione hardware di una scheda grafica "
        "dedicata per ambiente laptop. L'analisi del consumo di risorse computazionali evidenzia un'occupazione di picco della memoria VRAM pari a "
        "settecentosettantotto virgola settanta megabyte durante la generazione dei batch, valore ampiamente inferiore alla soglia di un gigabyte che "
        "dimostra l'estrema efficienza e leggerezza dell'architettura implementata. La latenza temporale del singolo passaggio forward della U-Net "
        "si attesta su zero virgola duecentotrentotto secondi, consentendo una rapida iterazione nei cicli di calcolo."
    )
    story.append(Paragraph(p12, body_style))

    p13 = (
        "Sul fronte della rapidità di campionamento, l'indagine mette in luce un contrasto netto tra la traiettoria stocastica standard DDPM e il "
        "campionamento deterministico accelerato DDIM. L'esecuzione del reverse loop completo su mille passi di denoisazione richiede un tempo medio "
        "di quarantanove virgola cinquantuno secondi per la sintesi di un singolo avatar. Al contrario, l'impiego dello schema DDIM riduce la traiettoria "
        "a cinquanta passi temporali, abbattendo la latenza a un secondo e undici centesimi per immagine, con un'ulteriore contrazione a zero virgola "
        "novantanove secondi nel caso di una discretizzazione a venti passi. Questa accelerazione di circa quarantacinque volte rispetto al campionamento "
        "completo avviene senza alcuna degradazione visibile nella nitidezza dei contorni o nella fedeltà ai tratti semantici, rendendo il modello "
        "adatto all'impiego interattivo."
    )
    story.append(Paragraph(p13, body_style))

    p14 = (
        "La valutazione della qualità distribuzionale e dell'allineamento statistico è stata formalizzata calcolando la Fréchet Inception Distance e "
        "la Kernel Inception Distance su campioni generati dalle partizioni di test, adottando la dimensionalità a sessantaquattro feature di Inception "
        "con ri-normalizzazione preventiva rigorosa nell'intervallo continuo compreso tra zero e uno. Sulla partizione di test ordinario in-distribution, "
        "composta da quasi diecimila immagini con tratti combinati conformi alle frequenze di training, il generatore consegue un valore di FID pari a "
        "due virgola zero centonovantatré e un valore medio di KID pari a zero virgola zero cinquantatré con deviazione standard trascurabile. "
        "Valutando il modello sulla partizione di test out-of-distribution composizionale — costituita dai quattrocentocinquantotto esemplari che esibiscono "
        "la combinazione trattenuta di capelli ondulati e assenza di occhiali — il valore di FID registrato risulta pari a uno virgola quattrocentocinquantaquattro, "
        "mentre il KID medio si attesta su zero virgola zero settecentodiciotto."
    )
    story.append(Paragraph(p14, body_style))

    p15 = (
        "Il riscontro numerico del FID sulla partizione out-of-distribution, perfettamente allineato e addirittura lievemente inferiore a quello ordinario, "
        "fornisce una prova tangibile della robustezza geometrica del modello: la U-Net non collassa su artefatti caotici né genera strutture degenerate "
        "quando deve fondere attributi mai osservati congiuntamente. L'incremento del valore medio di KID da zero virgola zero cinquantatré a zero virgola "
        "zero settecentodiciotto, corrispondente a un aumento percentuale nell'ordine del trentacinque percento, riflette coerentemente la maggiore "
        "incertezza probabilistica del modello nel comporre matrici di feature visive reciprocamente inedite, manifestandosi sul piano visivo in lievi "
        "irregolarità di micro-contrasto che tuttavia non inficiano minimamente la plausibilità complessiva del volto sintetizzato."
    )
    story.append(Paragraph(p15, body_style))

    p16 = (
        "La variabilità stocastica della generazione a parità di condizionamento linguistico è stata quantificata mediante la distanza percettiva Pairwise "
        "LPIPS calcolata su backbone VGG tra avatar generati con semi casuali differenti compresi tra mille e milletre. Su un primo prompt caratterizzato "
        "da carnagione porcellana, capelli corti, occhi azzurri, occhiali tondi e assenza di barba, la diversità percettiva media tra le coppie di seed "
        "è risultata pari a zero virgola millenovecentottantuno. Su un secondo prompt, raffigurante pelle scura, capelli mossi, occhi scuri, assenza di occhiali "
        "e barba folta, la dispersione sale a zero virgola duemilacinquecentodieci. La diversità media complessiva calcolata sull'insieme dei test "
        "è pari a zero virgola duemiladuecentoquarantacinque. Questo risultato attesta che la guida del testo indirizza saldamente l'identità semantica "
        "senza soffocare la naturale eterogeneità morfologica del processo diffusivo, permettendo al generatore di produrre declinazioni fisionomiche "
        "differenziate e non banali."
    )
    story.append(Paragraph(p16, body_style))

    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=1.0, color=colors.HexColor('#1a365d'), spaceBefore=6, spaceAfter=12))

    # =========================================================================
    # PARTE 2: ANALYSIS AND DISCUSSION
    # =========================================================================
    story.append(Paragraph("Analysis and Discussion of the Model’s Performance, Challenges Faced, and Recommendations for Improvement", h1_style))

    # 2.1 Model's Performance
    story.append(Paragraph("1. Analisi e Discussione Approfondita delle Prestazioni del Modello", h2_style))

    p17 = (
        "La risposta alla domanda scientifica fondamentale posta dall'assegnamento — ovvero se un modello di diffusione estremamente compatto con "
        "denoiser e text encoder entrambi addestrati da zero possa generalizzare a combinazioni di attributi non osservate congiuntamente durante l'addestramento — "
        "è affermativa, e trova riscontro sia nei riscontri numerici sia nell'ispezione morfologica degli avatar sintetizzati. Il successo della generalizzazione "
        "composizionale risiede nell'interazione virtuosa tra due proprietà strutturali della pipeline: l'ortogonalità funzionale del meccanismo di spatial "
        "cross-attention e il forte bias induttivo geometrico intrinseco alle convoluzioni della U-Net."
    )
    story.append(Paragraph(p17, body_style))

    p18 = (
        "All'interno dei blocchi di cross-attention, i token lessicali vengono proiettati negli spazi lineari di chiavi e valori mediante matrici di peso "
        "apprese congiuntamente. Poiché le teste di attenzione calcolano similarità scalari indipendenti per ciascun vettore di contesto, la presenza del token "
        "relativo ai capelli mossi attiva selettivamente la componente tensoriale deputata alla capigliatura senza costringere le rimanenti teste ad attivare "
        "configurazioni correlate. In parallelo, il bias induttivo convoluzionale impone una separazione topologica naturale: i pattern visivi legati alla "
        "fronte e alla chioma risiedono nella frazione superiore delle mappe di attivazione, mentre le feature associate al naso, agli occhi e agli occhiali "
        "si concentrano nella fascia mediana. Questa compartimentazione spaziale fa sì che la U-Net non apprenda associazioni spurie rigide, consentendo "
        "alle due feature di coesistere in modo coerente nello stesso output."
    )
    story.append(Paragraph(p18, body_style))

    p19 = (
        "Un elemento essenziale che governa le prestazioni generative risiede nella sensibilità alla scala di Classifier-Free Guidance. Durante la sperimentazione "
        "è emerso un comportamento fortemente correlato al valore del moltiplicatore scalare w. Impiegando fattori di guida inferiori a due, il modello "
        "manifesta una regressione verso la distribuzione a priori, tendendo a ignorare parzialmente gli attributi rari in favore delle configurazioni più "
        "frequenti nel dataset. Al contrario, l'intervallo compreso tra due virgola cinque e tre virgola cinque realizza il compromesso ideale: la forza del "
        "condizionamento orienta in modo impeccabile sia la presenza dei capelli mossi sia l'assenza della montatura degli occhiali, preservando sfumature e "
        "morbidezza nei contorni del viso. Se la guida viene spinta oltre il valore cinque, l'estrapolazione lineare della velocità finisce per sovraccaricare "
        "le ampiezze dei gradienti, provocando un'ipersaturazione cromatica e un contrasto innaturale lungo i bordi delle campiture, effetto collaterale "
        "della dispersione nello spazio nativo dei pixel in assenza di un collettore latente compresso."
    )
    story.append(Paragraph(p19, body_style))

    # 2.2 Challenges Faced
    story.append(Paragraph("2. Sfide Ingegneristiche Affrontate e Soluzioni Adottate nel Codice", h2_style))

    p20 = (
        "La realizzazione da zero dell'intero sistema ha posto molteplici sfide di natura teorica e implementativa, che hanno richiesto interventi mirati "
        "sull'architettura del codice e sulla dinamica dei gradienti per assicurare la stabilità numerica e la qualità del campionamento."
    )
    story.append(Paragraph(p20, body_style))

    p21 = (
        "La prima grande difficoltà ha riguardato l'instabilità nel co-addestramento simultaneo della U-Net e del text encoder. Data la marcata asimmetria "
        "dimensionale tra la rete di denoisazione, forte di oltre ventiquattro milioni di parametri, e il trasformatore linguistico, limitato a circa due milioni, "
        "nelle fasi iniziali del training la U-Net tendeva a minimizzare la perdita di velocità concentrandosi esclusivamente sulle statistiche visive non condizionate, "
        "lasciando il text encoder in uno stato di convergenza ritardata. Per risolvere questa disarmonia è stata concepita una strategia di ottimizzazione a due "
        "stadi: nei primi cinque cicli di epoca, un riscaldamento lineare ha protetto il text encoder da aggiornamenti distruttivi; parallelamente, l'applicazione "
        "del gradient clipping disaccoppiato a soglia unitaria su ciascun modulo e la disattivazione del decadimento del peso per i parametri monodimensionali hanno "
        "impedito ai gradienti convoluzionali di soffocare l'evoluzione delle matrici di attenzione testuale."
    )
    story.append(Paragraph(p21, body_style))

    p22 = (
        "Una seconda criticità severa si è manifestata sotto forma di divergenza delle traiettorie inverse durante il campionamento con Classifier-Free Guidance. "
        "L'estrapolazione vettoriale della velocità, pur garantendo un forte allineamento al testo, produceva valori intermedi che spingevano la stima "
        "analitica del segnale pulito al di fuori dei limiti ammissibili dell'intervallo simmetrico tra meno uno e più uno. Nei passaggi temporali successivi "
        "questo sforamento provocava un accumulo di errori a cascata, culminando in gravi artefatti di posterizzazione e chiazze di pixel bianchi incontrollati. "
        "La soluzione ha richiesto l'integrazione del dynamic range clipping direttamente nel ciclo di reverse sampling: a ogni passo temporale, la stima provvisoria "
        "dell'immagine denoisata calcolata a partire dalla velocità predetta viene forzata numericamente nell'intervallo tra meno uno e più uno prima di procedere "
        "alla ponderazione stocastica dello stato precedente. Questo vincolo ha eradicato ogni residuo di posterizzazione, restituendo campiture solide e uniformi."
    )
    story.append(Paragraph(p22, body_style))

    p23 = (
        "La terza sfida ha riguardato l'elevata varianza stocastica intrinseca ai modelli di diffusione nello spazio dei pixel, che manifestavano un'alternanza "
        "della fedeltà fisionomica tra epoche successive dovuta a fluttuazioni dei pesi attorno ai minimi locali. L'introduzione della media mobile esponenziale "
        "con fattore di decadimento zero virgola novecentonovantanove ha consentito di disaccoppiare la fase esplorativa dell'ottimizzatore dalla stabilità "
        "strutturale del denoiser finale. I pesi consolidati dell'AveragedModel preservano la nitidezza geometrica del volto e la coerenza cromatica "
        "anche nei passaggi temporali a elevata componente di rumore."
    )
    story.append(Paragraph(p23, body_style))

    p24 = (
        "Infine, la gestione dei contesti a precisione mista con rappresentazione FP16 ha fatto emergere criticità di underflow numerico durante il calcolo "
        "delle maschere di attenzione. L'assegnazione convenzionale di costanti negative elevate come meno dieci alla nona nei token di padding provocava la "
        "comparsa di valori non definiti (NaN) all'uscita delle funzioni softmax, compromettendo l'integrità dei gradienti. La problematica è stata sanata "
        "sostituendo le matrici additive con maschere booleane native compatibili con i kernel ottimizzati e corredando i flussi di calcolo con chiamate difensive "
        "di sanitizzazione numerica."
    )
    story.append(Paragraph(p24, body_style))

    # 2.3 Recommendations for Improvement
    story.append(Paragraph("3. Raccomandazioni per il Miglioramento Futuro", h2_style))

    p25 = (
        "Alla luce dell'esperienza progettuale maturata e delle evidenze empiriche raccolte, si delineano tre direttrici primarie di sviluppo e ottimizzazione "
        "per future evoluzioni del sistema."
    )
    story.append(Paragraph(p25, body_style))

    p26 = (
        "In primo luogo, si raccomanda l'adozione di un'architettura latente compatta, basata sull'addestramento strictly from-scratch di un Vector Quantized "
        "Variational Autoencoder (VQ-VAE) o di un autoencoder convoluzionale poco profondo. Attualmente, l'esecuzione della spatial cross-attention direttamente "
        "sulla griglia dei pixel a piena risoluzione comporta un onere computazionale che scala quadraticamente con le dimensioni spaziali della mappa di feature. "
        "Comprimendo preventivamente le immagini in uno spazio latente compatto avente risoluzione sedici per sedici con quattro canali latenti, si otterrebbe "
        "un abbattimento significativo del footprint di memoria e dei tempi di inferenza, consentendo di reinvestire il budget parametrico nell'aumento della "
        "profondità della U-Net o nell'innalzamento della risoluzione finale a centoventotto per centoventotto pixel senza violare le soglie hardware."
    )
    story.append(Paragraph(p26, body_style))

    p27 = (
        "Una seconda raccomandazione riguarda l'integrazione di una perdita ausiliaria di consistenza degli attributi calcolata mediante un classificatore "
        "multi-etichetta ausiliario. Sebbene la funzione di perdita basata sull'errore quadratico medio della velocità supervisioni efficacemente la convergenza "
        "globale, essa tratta ogni deviazione di pixel in modo indifferenziato. L'impiego di una testa di classificazione congelata capace di calcolare una perdita "
        "di cross-entropy sulla stima dell'immagine denoisata x zero a timestep intermedi consentirebbe di retro-propagare gradienti di alto livello esplicitamente "
        "legati all'esattezza cromatica dell'iride, alla presenza della barba o alla tipologia della montatura, incrementando ulteriormente l'aderenza semantica "
        "nei dettagli fisionomici più minuti."
    )
    story.append(Paragraph(p27, body_style))

    p28 = (
        "Infine, sul versante della formulazione matematica del processo stocastico, si suggerisce la transizione dal framework DDPM con equazioni differenziali "
        "stocastiche curvilinee verso il paradigma del Rectified Flow e del Flow Matching con trasporto ottimale, teorizzato da Lipman e collaboratori. "
        "Formulando la dinamica generativa attraverso equazioni differenziali ordinarie con traiettorie lineari a velocità costante tra il rumore gaussiano "
        "e la distribuzione degli avatar, diverrebbe possibile sintetizzare immagini di qualità impeccabile in appena quattro o otto passi di integrazione numerica "
        "deterministica, rendendo l'inferenza quasi istantanea e superando i limiti di approssimazione intrinseci agli schemi DDIM discretizzati."
    )
    story.append(Paragraph(p28, body_style))

    # Conclusione discorsiva
    story.append(Paragraph("Considerazioni Conclusive dello Studente", h2_style))
    p29 = (
        "In conclusione, l'attività svolta ha permesso di consolidare approfondite competenze pratiche e metodologiche sulle dinamiche dei modelli "
        "generativi basati sulla diffusione stocastica. La sfida di implementare ogni componente strictly from-scratch — dalla tokenizzazione alla U-Net con "
        "cross-attention e allo scheduler v-prediction — ha offerto una comprensione trasparente dei compromessi architetturali necessari per coniugare "
        "stabilità di ottimizzazione, leggerezza computazionale e capacità di astrazione semantica. I risultati ottenuti certificano la piena validità "
        "dell'approccio e confermano come anche un'architettura compatta, se accuratamente regolarizzata, sia pienamente in grado di disaccoppiare concetti "
        "complessi e generalizzare con successo verso combinazioni composizionali inedite."
    )
    story.append(Paragraph(p29, body_style))

    # Bibliografia essenziale formale
    story.append(Spacer(1, 3*mm))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor('#cbd5e0'), spaceBefore=4, spaceAfter=8))
    story.append(Paragraph("Riferimenti Bibliografici Formativi", h2_style))
    refs = [
        "J. Ho, A. Jain, P. Abbeel, <i>Denoising Diffusion Probabilistic Models</i>, Advances in Neural Information Processing Systems (NeurIPS), 2020.",
        "A. Q. Nichol, P. Dhariwal, <i>Improved Denoising Diffusion Probabilistic Models</i>, International Conference on Machine Learning (ICML), 2021.",
        "T. Salimans, J. Ho, <i>Progressive Distillation for Fast Sampling of Diffusion Models</i>, International Conference on Learning Representations (ICLR), 2022.",
        "J. Song, C. Meng, S. Ermon, <i>Denoising Diffusion Implicit Models</i>, International Conference on Learning Representations (ICLR), 2021.",
        "Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nicklas, M. Le, <i>Flow Matching for Generative Modeling</i>, arXiv:2210.02747, 2022.",
        "A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin, <i>Attention Is All You Need</i>, NeurIPS, 2017."
    ]
    ref_style = ParagraphStyle(
        'RefStyle',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#4a5568'),
        leftIndent=12,
        firstLineIndent=-12,
        spaceAfter=3
    )
    for r in refs:
        story.append(Paragraph(f"• {r}", ref_style))

    # Generazione fisica del PDF
    doc.build(story, canvasmaker=AcademicNumberedCanvas)
    print(f"Report PDF generato con successo in: {output_pdf_path}")

if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "Relazione_Tecnica_Avatar_Diffusion.pdf"
    build_academic_pdf(output_path)

