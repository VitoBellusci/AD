import json
import os
import re
from typing import List, Dict

class AvatarTokenizer:
    """
    Tokenizer deterministico per didascalie generate da metadati di avatar.
    Preserva i token speciali nei primi 4 ID (0..3) e garantisce coerenza
    totale di sanitizzazione (lowercase, rimozione punteggiatura, split whitespace)
    tra fit ed encode.
    """
    def __init__(self, config=None):
        self.config = config
        self.vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.inverse_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}

    def _tokenize(self, text: str) -> List[str]:
        """
        Sanitizzazione e tokenizzazione uniforme:
        Converte in minuscolo, rimuove punteggiatura residua (es. virgole '1,', '98,')
        e separa per spazi bianchi.
        """
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        return clean_text.split()

    def fit(self, training_texts: List[str]):
        """
        Costruisce il vocabolario preservando i token speciali (ID 0..3).
        I nuovi token partono tassativamente da ID 4.
        """
        word_count = max(self.vocab.values())  # Inizia da 3 -> il primo token aggiunto avra ID 4
        for text in training_texts:
            tokens = self._tokenize(text)
            for token in tokens:
                if token not in self.vocab:
                    word_count += 1
                    self.vocab[token] = word_count

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        """
        Converte una stringa in una sequenza di indici interi.
        Utilizza esattamente la stessa routine _tokenize di fit() per garantire coerenza totale.
        Applica padding o troncamento a max_seq_len.
        """
        tokens = self._tokenize(text)
        encoded = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

        # Padding o Truncation al valore max_seq_len
        max_seq_len = 20
        if self.config is not None:
            if isinstance(self.config, dict):
                max_seq_len = self.config.get("max_seq_len", 20)
            else:
                max_seq_len = getattr(self.config, "max_seq_len", 20)

        if len(encoded) < max_seq_len:
            encoded += [self.vocab["<PAD>"]] * (max_seq_len - len(encoded))
        else:
            encoded = encoded[:max_seq_len]

        return encoded

    def decode(self, ids: List[int]) -> str:
        """
        Decodifica una sequenza di indici in una stringa di testo.
        """
        return " ".join([self.inverse_vocab.get(i, "<UNK>") for i in ids])

    def save_vocab(self, vocab_path: str = None):
        """
        Salva il vocabolario su file JSON.
        """
        if vocab_path is None:
            if self.config is not None:
                if isinstance(self.config, dict):
                    vocab_path = self.config.get("vocab_path", "preprocessing/vocab.json")
                else:
                    vocab_path = getattr(self.config, "vocab_path", "preprocessing/vocab.json")
            else:
                vocab_path = "preprocessing/vocab.json"

        dir_name = os.path.dirname(vocab_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, indent=2)

    def load_vocab(self, vocab_path: str = None):
        """
        Carica il vocabolario da file JSON e ricostruisce inverse_vocab
        garantendo chiavi intere.
        """
        if vocab_path is None:
            if self.config is not None:
                if isinstance(self.config, dict):
                    vocab_path = self.config.get("vocab_path", "preprocessing/vocab.json")
                else:
                    vocab_path = getattr(self.config, "vocab_path", "preprocessing/vocab.json")
            else:
                vocab_path = "preprocessing/vocab.json"

        with open(vocab_path, 'r', encoding='utf-8') as f:
            loaded_vocab = json.load(f)

        # Assicura che i valori del vocabolario siano interi
        self.vocab = {k: int(v) if str(v).isdigit() else v for k, v in loaded_vocab.items()}
        self.inverse_vocab = {
            int(v) if str(v).isdigit() else v: k 
            for k, v in self.vocab.items()
        }

    def __len__(self) -> int:
        return len(self.vocab)
