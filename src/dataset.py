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
                 index_path='concept_index.pkl', transform=None):
        self.broden_root = broden_root
        self.concept_name = concept_name
        self.transform = transform or IMAGENET_TRANSFORM

        labels = pd.read_csv(os.path.join(broden_root, 'label.csv'))
        match = labels[labels['name'] == concept_name]
        if len(match) == 0:
            raise ValueError(
                f"Concept '{concept_name}' not found in label.csv."
            )

        self.label_number = int(match.iloc[0]['number'])
        self.mask_code = self._find_mask_code(broden_root, concept_name)

        # Load pre-built index
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"concept_index.pkl not found at {index_path}. "
                f"Run build_concept_index() first."
            )
        with open(index_path, 'rb') as f:
            concept_index = pickle.load(f)

        if self.mask_code not in concept_index:
            raise ValueError(
                f"Concept '{concept_name}' (code {self.mask_code}) "
                f"not found in index."
            )

        self.entries = concept_index[self.mask_code][split]
        print(f"  '{concept_name}' ({split}): {len(self.entries)} images")

    def _find_mask_code(self, broden_root, concept_name):
        """Find the pixel code used in mask files for this concept."""
        for csv_name in ['c_object.csv', 'c_part.csv',
                         'c_material.csv', 'c_color.csv']:
            path = os.path.join(broden_root, csv_name)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            match = df[df['name'] == concept_name]
            if len(match) > 0:
                return int(match.iloc[0]['code'])
        raise ValueError(
            f"Could not find mask code for '{concept_name}' "
            f"in any c_*.csv file."
        )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Load and transform image
        img = Image.open(entry['image_path']).convert('RGB')
        img_tensor = self.transform(img)

        # Load mask and extract binary mask for this concept
        mask_arr = np.array(Image.open(entry['mask_path']))
        if mask_arr.ndim == 3:
            mask_channel = mask_arr[:, :, 0]
        else:
            mask_channel = mask_arr

        binary_mask = (mask_channel == self.mask_code).astype(np.float32)

        return img_tensor, binary_mask