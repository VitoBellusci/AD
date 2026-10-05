import json
import os
import random
from typing import List, Dict, Tuple

class CompositionalSplitter:
    """
    Partiziona il dataset in 4 split disgiunti:
    - Train (80% dei dati in-distribution)
    - Val (10% dei dati in-distribution)
    - Test Ordinario (test_ind, 10% dei dati in-distribution)
    - Test OOD (test_ood, campioni con combinazioni di attributi bloccate)
    """
    def __init__(self, config):
        self.config = config

    def split(self, metadata_list: List[Dict]) -> Tuple[List[int], List[int], List[int], List[int]]:
        train_indices, ood_indices = [], []
        
        # Estrazione e normalizzazione delle combinazioni bloccate OOD
        raw_blocked = getattr(self.config, "ood_blocked_combinations", [])
        if isinstance(self.config, dict):
            raw_blocked = self.config.get("ood_blocked_combinations", [])
            
        blocked_sets = []
        for combo in raw_blocked:
            if isinstance(combo, set):
                blocked_sets.append({(str(k), str(v)) for k, v in combo})
            else:
                blocked_sets.append({(str(x[0]), str(x[1])) for x in combo})

        for idx, meta in enumerate(metadata_list):
            current_comb = {(str(k), str(v)) for k, v in meta.items()}
            is_ood = any(
                blocked_set.issubset(current_comb) 
                for blocked_set in blocked_sets
            )
            if is_ood:
                ood_indices.append(idx)
            else:
                train_indices.append(idx)

        # 4-Way Partition: Train (80%), Val (10%), Ordinary Test (10%), OOD Test (Held-out)
        random.seed(42)
        random.shuffle(train_indices)
        n_total = len(train_indices)
        val_size = int(n_total * 0.1)
        test_size = int(n_total * 0.1)
        
        val_indices = train_indices[:val_size]
        test_ind_indices = train_indices[val_size:val_size + test_size]
        final_train_indices = train_indices[val_size + test_size:]

        # Salva le partizioni su disco per esatta riproducibilità
        splits = {
            "train": final_train_indices,
            "val": val_indices,
            "test_ind": test_ind_indices,
            "test_ood": ood_indices
        }
        splits_path = getattr(self.config, "splits_path", None)
        if isinstance(self.config, dict):
            splits_path = self.config.get("splits_path", splits_path)
        if not splits_path:
            splits_path = "preprocessing/splits.json"

        splits_dir = os.path.dirname(splits_path)
        if splits_dir:
            os.makedirs(splits_dir, exist_ok=True)
            
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)

        return final_train_indices, val_indices, test_ind_indices, ood_indices
