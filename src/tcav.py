import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def collect_activations_pooled(model, dataset, layer_name,
                                batch_size=64, max_samples=500):
    
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=2)

    all_acts = []
    n_processed = 0

    model.eval()
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(model.device)
            _ = model(imgs)
            acts = model.get_activations()[layer_name]  # (B, K, H, W)

            # Global average pool: (B, K, H, W) -> (B, K)
            pooled = acts.mean(dim=(2, 3))
            all_acts.append(pooled.cpu().numpy())

            n_processed += imgs.shape[0]
            if n_processed >= max_samples:
                break

    return np.vstack(all_acts)


def collect_random_activations(model, broden_root, layer_name,
                                n_samples=500, batch_size=64):
    
    import pandas as pd
    import os
    import random
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    index = pd.read_csv(os.path.join(broden_root, 'index.csv'))
    all_paths = index['image'].tolist()
    random.seed(42)
    random.shuffle(all_paths)
    selected = all_paths[:n_samples]

    all_acts = []
    batch_imgs = []

    model.eval()
    with torch.no_grad():
        for img_path in selected:
            full_path = os.path.join(broden_root, 'images', img_path)
            try:
                img = Image.open(full_path).convert('RGB')
                batch_imgs.append(transform(img))
            except Exception:
                continue

            if len(batch_imgs) == batch_size:
                batch = torch.stack(batch_imgs).to(model.device)
                _ = model(batch)
                acts = model.get_activations()[layer_name]
                pooled = acts.mean(dim=(2, 3))
                all_acts.append(pooled.cpu().numpy())
                batch_imgs = []

        if batch_imgs:
            batch = torch.stack(batch_imgs).to(model.device)
            _ = model(batch)
            acts = model.get_activations()[layer_name]
            pooled = acts.mean(dim=(2, 3))
            all_acts.append(pooled.cpu().numpy())

    return np.vstack(all_acts)


def train_cav(concept_acts, random_acts, random_state=42):

    # Balance the sets
    n = min(len(concept_acts), len(random_acts))
    concept_acts = concept_acts[:n]
    random_acts  = random_acts[:n]

    X = np.vstack([concept_acts, random_acts])
    y = np.hstack([np.ones(n), np.zeros(n)])

    # Standardise features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2,
        random_state=random_state, stratify=y
    )

    clf = LogisticRegression(
        C=1.0, max_iter=1000,
        random_state=random_state, solver='lbfgs'
    )
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))

    # CAV = normalised weight vector
    cav_raw = clf.coef_.squeeze()
    cav = cav_raw / (np.linalg.norm(cav_raw) + 1e-8)

    return cav, accuracy


def compute_tcav_score(model, images, cav, target_class_idx, layer_name):
    model.eval()
    device = model.device
    cav_t = torch.tensor(cav, dtype=torch.float32, device=device)

    all_derivatives = []

    for i in range(0, len(images), 16):
        batch = images[i:i+16].to(device)

        with torch.enable_grad():
            # Forward pass to populate activation cache
            _ = model(batch)
            h = model.get_activations()[layer_name]  # (B, K, H, W)

            # Detach and re-attach as leaf for gradient computation
            h_leaf = h.detach().requires_grad_(True)

            # Resume forward pass from this layer
            out = _forward_from(model.alexnet, h_leaf, layer_name)
            score = out[:, target_class_idx].sum()
            grad = torch.autograd.grad(score, h_leaf)[0]  # (B, K, H, W)

        # Pool gradient to (B, K) and dot with CAV
        grad_pooled = grad.mean(dim=(2, 3))           # (B, K)
        dd = (grad_pooled * cav_t[None, :]).sum(dim=1)  # (B,)
        all_derivatives.append(dd.detach().cpu().numpy())

    derivatives = np.concatenate(all_derivatives)
    tcav_score = float((derivatives > 0).mean())

    return tcav_score, derivatives


def _forward_from(alexnet, h, from_layer):
    # Map layer name to the index in features where it ends
    layer_end = {
        'conv1': 2,
        'conv2': 5,
        'conv3': 8,
        'conv4': 10,
        'conv5': 12,
    }
    start = layer_end[from_layer]

    x = h
    # Continue through remaining feature layers
    for i in range(start, len(alexnet.features)):
        x = alexnet.features[i](x)

    x = alexnet.avgpool(x)
    x = torch.flatten(x, 1)
    x = alexnet.classifier(x)
    return x


def compute_tcav_with_significance(model, concept_dataset,
                                    broden_root, layer_name,
                                    target_class_idx,
                                    n_trials=5, max_samples=500):
    from scipy import stats

    # Collect concept activations once
    print(f"    Collecting concept activations...")
    concept_acts = collect_activations_pooled(
        model, concept_dataset, layer_name, max_samples=max_samples
    )

    # Collect random activations once
    print(f"    Collecting random activations...")
    random_acts = collect_random_activations(
        model, broden_root, layer_name, n_samples=max_samples
    )

    # Collect evaluation images (use concept images)
    loader = DataLoader(concept_dataset, batch_size=64, shuffle=False)
    eval_images = []
    for imgs, _ in loader:
        eval_images.append(imgs)
        if sum(b.shape[0] for b in eval_images) >= 200:
            break
    eval_images = torch.cat(eval_images)[:200]

    # Run n_trials with different random seeds
    rng = np.random.default_rng(42)
    trial_scores = []

    for trial in range(n_trials):
        seed = int(rng.integers(0, 10000))

        # Bootstrap sample
        n = min(200, len(concept_acts), len(random_acts))
        idx_c = rng.choice(len(concept_acts), size=n, replace=False)
        idx_r = rng.choice(len(random_acts),  size=n, replace=False)

        cav, acc = train_cav(
            concept_acts[idx_c],
            random_acts[idx_r],
            random_state=seed
        )

        # Skip unreliable CAVs
        if acc < 0.55:
            trial_scores.append(0.5)
            continue

        score, _ = compute_tcav_score(
            model, eval_images, cav, target_class_idx, layer_name
        )
        trial_scores.append(score)
        print(f"    Trial {trial+1}: TCAV={score:.3f}, CAV acc={acc:.3f}")

    scores = np.array(trial_scores)
    mean_score = float(scores.mean())
    std_score  = float(scores.std())

    # Binomial test: are scores consistently != 0.5?
    n_above = int((scores > 0.5).sum())
    try:
        p_val = float(stats.binomtest(n_above, n=n_trials, p=0.5).pvalue)
    except AttributeError:
        p_val = float(stats.binom_test(n_above, n=n_trials, p=0.5))

    is_significant = (p_val < 0.05) and (abs(mean_score - 0.5) > 0.1)

    return {
        'mean_tcav_score': mean_score,
        'std':             std_score,
        'all_scores':      scores.tolist(),
        'is_significant':  is_significant,
        'p_value':         p_val,
    }