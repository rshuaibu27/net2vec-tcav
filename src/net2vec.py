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

def train_multi_filter_probe(model, dataset, thresholds,
                              layer_name, n_epochs=30,
                              lr=1e-4, momentum=0.9, batch_size=64):
    """
    Learn a weight vector w ∈ R^K to combine thresholded filter
    activations for concept segmentation.

    From the paper eq. 2:
        M(x; w) = sigmoid(sum_k wk * I(Ak(x) > Tk))

    Trained with size-weighted binary cross-entropy (eq. 4):
        L = -[α * M * Lc + (1-α) * (1-M) * (1-Lc)]
    where α = 1 - mean_foreground_fraction

    Args:
        model:      AlexNetProbe instance
        dataset:    BrodenConceptDataset for this concept (train split)
        thresholds: numpy array of shape (K,)
        layer_name: e.g. 'conv5'
        n_epochs:   training epochs (paper uses 30)
        lr:         learning rate (paper uses 1e-4)
        momentum:   SGD momentum (paper uses 0.9)
        batch_size: images per batch (paper uses 64)

    Returns:
        weights: numpy array of shape (K,) — the concept embedding
        losses:  list of per-epoch losses (for debugging)
    """
    import torch.nn as nn

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=2)

    thresholds_t = torch.tensor(thresholds, dtype=torch.float32)

    # Get number of filters K from the thresholds
    K = len(thresholds)

    # Learnable weights — one per filter
    weights = torch.zeros(K, requires_grad=True,
                          device=model.device)
    nn.init.normal_(weights, mean=0.0, std=0.01)
    weights = weights.clone().detach().requires_grad_(True)

    optimizer = torch.optim.SGD([weights], lr=lr, momentum=momentum)

    # Compute α from the dataset's foreground fraction
    # α = 1 - mean fraction of foreground pixels
    # We estimate this from the first few batches
    foreground_fractions = []
    for imgs, masks in loader:
        foreground_fractions.append(masks.mean().item())
        if len(foreground_fractions) >= 5:
            break
    alpha = 1.0 - np.mean(foreground_fractions)
    alpha = float(np.clip(alpha, 0.01, 0.99))
    print(f"  Alpha (class balance): {alpha:.4f}")

    losses = []
    model.eval()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for img_batch, mask_batch in loader:
            img_batch  = img_batch.to(model.device)   # (B, 3, 227, 227)
            mask_batch = mask_batch.to(model.device)  # (B, 113, 113)

            with torch.no_grad():
                _ = model(img_batch)

            acts = model.get_activations()[layer_name]  # (B, K, H, W)
            B, K_acts, H, W = acts.shape

            # Threshold activations: I(Ak(x) > Tk)
            thresh = thresholds_t.to(model.device)
            binary_acts = (acts > thresh[None, :, None, None]).float()
            # binary_acts: (B, K, H, W)

            # Upsample to mask resolution
            target_h, target_w = mask_batch.shape[1], mask_batch.shape[2]
            flat = binary_acts.view(B * K_acts, 1, H, W)
            upsampled = F.interpolate(flat, size=(target_h, target_w),
                                      mode='bilinear', align_corners=False)
            upsampled = upsampled.view(B, K_acts, target_h, target_w)
            # upsampled: (B, K, 113, 113)

            # Weighted combination: sum_k wk * upsampled_k
            # weights: (K,) → broadcast to (B, K, H, W)
            weighted = (upsampled * weights[None, :, None, None]).sum(dim=1)
            # weighted: (B, 113, 113)

            # Sigmoid to get probability mask
            pred = torch.sigmoid(weighted)  # (B, 113, 113)

            # Size-weighted binary cross-entropy (eq. 4)
            loss = -(
                alpha * mask_batch * torch.log(pred + 1e-8)
                + (1 - alpha) * (1 - mask_batch) * torch.log(1 - pred + 1e-8)
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  loss: {avg_loss:.4f}")

    return weights.detach().cpu().numpy(), losses


def evaluate_multi_filter_iou(model, dataset, thresholds,
                               weights, layer_name, batch_size=32):
    """
    Evaluate the multi-filter probe's set IoU on a dataset split.

    Args:
        weights: numpy array of shape (K,) from train_multi_filter_probe()

    Returns:
        iou: float — set IoU on this split
    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2)

    thresholds_t = torch.tensor(thresholds, dtype=torch.float32)
    weights_t    = torch.tensor(weights,    dtype=torch.float32)

    total_intersection = 0.0
    total_union        = 0.0

    model.eval()
    with torch.no_grad():
        for img_batch, mask_batch in loader:
            img_batch  = img_batch.to(model.device)
            mask_batch = mask_batch.to(model.device)

            _ = model(img_batch)
            acts = model.get_activations()[layer_name]
            B, K, H, W = acts.shape

            thresh  = thresholds_t.to(model.device)
            weights_d = weights_t.to(model.device)

            binary_acts = (acts > thresh[None, :, None, None]).float()

            target_h, target_w = mask_batch.shape[1], mask_batch.shape[2]
            flat = binary_acts.view(B * K, 1, H, W)
            upsampled = F.interpolate(flat, size=(target_h, target_w),
                                      mode='bilinear', align_corners=False)
            upsampled = upsampled.view(B, K, target_h, target_w)

            weighted = (upsampled * weights_d[None, :, None, None]).sum(dim=1)
            pred = torch.sigmoid(weighted)
            pred_binary = (pred > 0.5).float()

            total_intersection += (pred_binary * mask_batch).sum().item()
            total_union += ((pred_binary + mask_batch) > 0.5).float().sum().item()

    if total_union == 0:
        return 0.0
    return total_intersection / total_union