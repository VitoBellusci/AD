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
            print(f"Pre-caricamento di {len(self.image_paths)} elementi in RAM. Attendere...")
            
            # Usa tensori PyTorch che supportano la memoria condivisa tra i worker
            first_img = Image.open(self.image_paths[0]).convert("RGB")
            tensor_shape = self.transform(first_img).shape
            
            self.cached_images = torch.empty((len(self.image_paths), *tensor_shape), dtype=torch.float32)
            
            max_seq_len = getattr(self.config, 'max_seq_len', 20)
            self.cached_captions = torch.empty((len(self.image_paths), max_seq_len), dtype=torch.long)
            
            for i, (img_path, meta) in enumerate(tqdm(zip(self.image_paths, self.metadata), total=len(self.image_paths), desc="Caching Dataset")):
                img = Image.open(img_path).convert("RGB")
                self.cached_images[i] = self.transform(img)
                
                caption_text = self.caption_gen.generate(meta)
                caption_tokens = self.tokenizer.encode(caption_text)
                self.cached_captions[i] = torch.tensor(caption_tokens, dtype=torch.long)
                
            # Condivide la memoria per l'accesso simultaneo da parte dei worker del DataLoader
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
