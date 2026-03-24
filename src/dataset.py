import os
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
    def __init__(self, broden_root, concept_name, split='train', transform=None):
        self.broden_root = broden_root
        self.concept_name = concept_name
        self.transform = transform or IMAGENET_TRANSFORM

        labels = pd.read_csv(os.path.join(broden_root, 'label.csv'))
        match = labels[labels['name'] == concept_name]
        if len(match) == 0:
            raise ValueError(
                f"Concept '{concept_name}' not found in label.csv. "
                f"Check spelling — use label.csv to find the exact name."
            )
        self.label_number = int(match.iloc[0]['number'])

        index = pd.read_csv(os.path.join(broden_root, 'index.csv'))
        index = index[index['split'] == split].reset_index(drop=True)

        mask_columns = ['color', 'object', 'part', 'material']
        self.entries = []

        for _, row in index.iterrows():
            for col in mask_columns:
                if pd.isna(row[col]):
                    continue

                mask_path = os.path.join(broden_root, 'images', row[col])
                if not os.path.exists(mask_path):
                    continue

                mask_arr = np.array(Image.open(mask_path))
                if mask_arr.ndim == 3:
                    mask_channel = mask_arr[:, :, 0]
                else:
                    mask_channel = mask_arr

                if self.label_number in np.unique(mask_channel):
                    self.entries.append({
                        'image_path': os.path.join(
                            broden_root, 'images', row['image']
                        ),
                        'mask_path':  mask_path,
                        'mask_col':   col,
                    })
                    break

        print(f"  '{concept_name}' ({split}): {len(self.entries)} images found")

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

        binary_mask = (mask_channel == self.label_number).astype(np.float32)

        return img_tensor, binary_mask