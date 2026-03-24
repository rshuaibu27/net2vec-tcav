import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm


def compute_thresholds(model, broden_root, layer_name,
                       tau=0.005, max_images=10000, batch_size=64):

    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    index = pd.read_csv(os.path.join(broden_root, 'index.csv'))
    train_paths = index[index['split'] == 'train']['image'].tolist()
    train_paths = train_paths[:max_images]

    print(f"Computing thresholds for {layer_name} "
          f"over {len(train_paths)} images...")

    all_activations = []

    model.eval()
    batch_imgs = []

    def process_batch(batch):
        batch_tensor = torch.stack(batch).to(model.device)
        with torch.no_grad():
            _ = model(batch_tensor)
        acts = model.get_activations()[layer_name]  # (B, K, H, W)
        B, K, H, W = acts.shape
        # Reshape to (K, B*H*W) — all spatial locations for each filter
        acts_flat = acts.permute(1, 0, 2, 3).reshape(K, -1)
        all_activations.append(acts_flat.cpu().numpy())

    for i, img_rel_path in enumerate(tqdm(train_paths)):
        img_path = os.path.join(broden_root, 'images', img_rel_path)
        try:
            img = Image.open(img_path).convert('RGB')
            batch_imgs.append(transform(img))
        except Exception:
            continue

        if len(batch_imgs) == batch_size:
            process_batch(batch_imgs)
            batch_imgs = []

    if batch_imgs:
        process_batch(batch_imgs)

    all_activations = np.concatenate(all_activations, axis=1)
    print(f"  Activation matrix shape: {all_activations.shape}")

    thresholds = np.quantile(all_activations, 1.0 - tau, axis=1)
    print(f"  Threshold range: [{thresholds.min():.4f}, {thresholds.max():.4f}]")

    return thresholds