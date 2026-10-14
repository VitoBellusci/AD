import json
from typing import List, Tuple, Set

class PreprocessingConfig:
    """
    Classe wrapper che carica la configurazione da un file JSON,
    ricostruisce le strutture dati native di Python e valida gli iperparametri.
    """
    def __init__(self, json_path: str):
        # Caricamento del file JSON
        # Il blocco with garantisce che, non appena l'operazione di lettura è terminata
        # ('r'), o anche se si verifica un errore improvviso, il file venga chiuso automaticamente 
        # e in modo sicuro
        with open(json_path, 'r') as f:
            raw_config = json.load(f)

        # Mappatura diretta e Type Hinting
        self.resolution: Tuple[int, int] = tuple(raw_config["resolution"])
        self.norm_min: float = raw_config["norm_min"]
        self.norm_max: float = raw_config["norm_max"]
        self.mean: List[float] = raw_config["mean"]
        self.std: List[float] = raw_config["std"]
        self.max_seq_len: int = raw_config["max_seq_len"]
        self.vocab_path: str = raw_config["vocab_path"]
        self.splits_path: str = raw_config.get("splits_path", "preprocessing/splits.json")

        # Ricostruzione delle strutture complesse non gestibili nel file JSON, che supporta solo
        # stringhe, numeri, booleani, dizionari e liste e non tipi di dati complessi
        # Converte [[["color", "blue"], ["proportion", "exaggerated"]]] 
        # in [{('color', 'blue'), ('proportion', 'exaggerated')}]
        # Le combinazioni del file JSON vengono convertite in liste di set di tuple, questo perché
        # l'operazione issubset() usata in splitter.py ha una complessità computazionale bassissima
        self.ood_blocked_combinations: List[Set[Tuple[str, str]]] = []
        for combination in raw_config.get("ood_blocked_combinations", []):
            rebuilt_set = set(tuple(attr) for attr in combination)
            self.ood_blocked_combinations.append(rebuilt_set)

        # Validazione istantanea (Fail-Fast), per evitare che il codice crashi subito piuttosto che
        # dopo ore di training
        self._validate()

    def _validate(self):
        """Verifica la coerenza matematica degli iperparametri."""
        if self.norm_min >= self.norm_max:
            raise ValueError("norm_min deve essere strettamente minore di norm_max")