import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


class BrodenConceptDataset(Dataset):

    def __init__(self, broden_root, concept_name, split='train',
                 index_path='concept_index.pkl', transform=None,
                 max_samples=None):
        self.broden_root = broden_root
        self.concept_name = concept_name
        self.transform = transform or IMAGENET_TRANSFORM

        # Find this concept's mask code and category
        self.mask_code, self.mask_col = self._find_mask_code(
            broden_root, concept_name
        )

        # Load pre-built index
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Index not found at {index_path}. "
                f"Run build_concept_index() first."
            )
        with open(index_path, 'rb') as f:
            concept_index = pickle.load(f)

        # Try tuple key (v2 index) first, fall back to integer key (v1)
        key = (self.mask_code, self.mask_col)
        if key in concept_index:
            self.entries = concept_index[key][split]
        elif self.mask_code in concept_index:
            self.entries = concept_index[self.mask_code][split]
        else:
            raise ValueError(
                f"Concept '{concept_name}' (code {self.mask_code}, "
                f"col {self.mask_col}) not found in index."
            )

        # Optionally cap the number of images
        if max_samples is not None and len(self.entries) > max_samples:
            import random
            random.seed(42)
            self.entries = random.sample(self.entries, max_samples)

        print(f"  '{concept_name}' ({split}): {len(self.entries)} images")

    def _find_mask_code(self, broden_root, concept_name):
        """Find pixel code and mask column for this concept."""
        col_map = {
            'part':     'c_part.csv',
            'color':    'c_color.csv',
            'object':   'c_object.csv',
            'part':     'c_part.csv',
            'material': 'c_material.csv',
        }
        for col, csv_name in col_map.items():
            path = os.path.join(broden_root, csv_name)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            match = df[df['name'] == concept_name]
            if len(match) > 0:
                return int(match.iloc[0]['code']), col
        raise ValueError(
            f"Could not find mask code for '{concept_name}'."
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        img = Image.open(entry['image_path']).convert('RGB')
        img_tensor = self.transform(img)

        mask_arr = np.array(Image.open(entry['mask_path']))
        if mask_arr.ndim == 3:
            mask_channel = mask_arr[:, :, 0]
        else:
            mask_channel = mask_arr

        binary_mask = (mask_channel == self.mask_code).astype(np.float32)

        return img_tensor, binary_mask