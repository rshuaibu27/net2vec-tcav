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

def compute_single_filter_iou(model, dataset, thresholds,
                               layer_name, batch_size=32):
    
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2)

    thresholds_t = torch.tensor(thresholds, dtype=torch.float32)

    total_intersection = None
    total_union        = None

    model.eval()
    for img_batch, mask_batch in tqdm(loader, desc=f"IoU {layer_name}"):
        img_batch = img_batch.to(model.device)
        mask_batch = mask_batch.to(model.device) 

        with torch.no_grad():
            _ = model(img_batch)

        acts = model.get_activations()[layer_name]
        B, K, H, W = acts.shape

        thresh = thresholds_t.to(model.device)
        binary_acts = (acts > thresh[None, :, None, None]).float()

        target_h, target_w = mask_batch.shape[1], mask_batch.shape[2]

        flat = binary_acts.view(B * K, 1, H, W)
        upsampled = F.interpolate(flat, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)
        upsampled = (upsampled > 0.5).float()       # re-binarise
        upsampled = upsampled.view(B, K, target_h, target_w)

        gt = mask_batch.unsqueeze(1).expand(-1, K, -1, -1) 

        intersection = (upsampled * gt).sum(dim=(0, 2, 3)) 
        union = ((upsampled + gt) > 0.5).float().sum(dim=(0, 2, 3)) 

        if total_intersection is None:
            total_intersection = intersection
            total_union        = union
        else:
            total_intersection += intersection
            total_union        += union

    # Set IoU: divide once after summing across all images
    iou = torch.where(
        total_union > 0,
        total_intersection / total_union,
        torch.zeros_like(total_intersection)
    )

    return iou.cpu().numpy()