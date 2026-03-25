import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def build_concept_index(broden_root, save_path):
    """
    Scan all Broden mask files once and build a lookup:
    (mask_code, mask_col) -> {'train': [...], 'val': [...]}

    Using (code, col) as the key avoids collisions between
    object/part/material/color codes which are not globally unique.
    """
    index = pd.read_csv(os.path.join(broden_root, 'index.csv'))
    mask_columns = ['color', 'object', 'part', 'material']
    concept_index = {}

    for _, row in tqdm(index.iterrows(), total=len(index),
                       desc="Building index"):
        for col in mask_columns:
            if pd.isna(row[col]):
                continue
            mask_path = os.path.join(broden_root, 'images', row[col])
            if not os.path.exists(mask_path):
                continue

            mask_arr = np.array(Image.open(mask_path))
            mask_channel = mask_arr[:, :, 0] if mask_arr.ndim == 3 \
                           else mask_arr

            for code in np.unique(mask_channel):
                if code == 0:
                    continue
                key = (int(code), col)
                if key not in concept_index:
                    concept_index[key] = {'train': [], 'val': []}
                concept_index[key][row['split']].append({
                    'image_path': os.path.join(
                        broden_root, 'images', row['image']
                    ),
                    'mask_path': mask_path,
                    'mask_col':  col,
                })

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(concept_index, f)

    print(f"\nIndex built: {len(concept_index)} (code, category) pairs")
    print(f"Saved to: {save_path}")
    return concept_index


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--broden_root', default='broden1_227')
    p.add_argument('--save_path',
                   default='net2vec-tcav/concept_index_v2.pkl')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build_concept_index(args.broden_root, args.save_path)