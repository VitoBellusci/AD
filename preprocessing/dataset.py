import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Dict, Any, Tuple
from preprocessing.config import PreprocessingConfig
from preprocessing.caption_generator import CaptionGenerator
from preprocessing.tokenizer import AvatarTokenizer

class AvatarDataset(Dataset):
    """
    Creiamo il dataset effettivo che integra l'immagine ed il testo. Questa classe si assicura che l'immagine sia
    normalizzata tra [-1, 1] e che la caption sia tokenizzata usando soltanto il vocabolario costruito con il set di training
    """

    def __init__(
        self,
        image_paths: List[str],
        metadata: List[Dict[str, Any]],
        tokenizer: AvatarTokenizer,
        config: PreprocessingConfig,
        use_ram_cache: bool = True
    ):
        self.image_paths = image_paths
        self.metadata = metadata
        self.tokenizer = tokenizer
        self.config = config
        self.caption_gen = CaptionGenerator()
        self.use_ram_cache = use_ram_cache

        self.transform = transforms.Compose([
            transforms.Resize(self.config.resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.mean,
                std=self.config.std
            )
        ])

        if self.use_ram_cache and len(self.image_paths) > 0:
            from tqdm import tqdm
            from torch.utils.data import DataLoader
            print(f"Pre-caricamento parallelo di {len(self.image_paths)} elementi in RAM. Attendere...")
            
            self.use_ram_cache = False  # Disabilita temporaneamente per forzare la lettura dal disco
            
            # Scopri le dimensioni dei tensori
            first_img = Image.open(self.image_paths[0]).convert("RGB")
            tensor_shape = self.transform(first_img).shape
            max_seq_len = getattr(self.config, 'max_seq_len', 20)
            
            # Alloca la memoria
            self.cached_images = torch.empty((len(self.image_paths), *tensor_shape), dtype=torch.float32)
            self.cached_captions = torch.empty((len(self.image_paths), max_seq_len), dtype=torch.long)
            
            # Usiamo PyTorch DataLoader per parallelizzare il caricamento su più processi CPU
            cache_loader = DataLoader(
                self, 
                batch_size=512, 
                shuffle=False, 
                num_workers=4,
                drop_last=False
            )
            
            idx = 0
            for imgs, caps in tqdm(cache_loader, desc="Caching Parallelo in RAM"):
                b_size = imgs.size(0)
                self.cached_images[idx : idx + b_size] = imgs
                self.cached_captions[idx : idx + b_size] = caps
                idx += b_size
                
            self.use_ram_cache = True  # Riattiva il cache
            
            # Condivide la memoria
            self.cached_images.share_memory_()
            self.cached_captions.share_memory_()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if getattr(self, 'use_ram_cache', False):
            return self.cached_images[idx].clone(), self.cached_captions[idx].clone()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        meta = self.metadata[idx]
        caption_text = self.caption_gen.generate(meta)
        caption_tokens = self.tokenizer.encode(
            caption_text
        )
        caption_tensor = torch.tensor(caption_tokens, dtype=torch.long)

        return image_tensor, caption_tensor
