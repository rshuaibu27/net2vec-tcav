import os
import sys
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import load_model
from src.dataset import BrodenConceptDataset
from src.net2vec import (
    compute_thresholds,
    compute_single_filter_iou,
    train_multi_filter_probe,
    evaluate_multi_filter_iou,
)

CONCEPTS = ['sky', 'grass', 'building', 'person',
            'car', 'wheel', 'water', 'wood']

LAYERS = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--broden_root',  default='broden1_227')
    p.add_argument('--results_dir',  default='results')
    p.add_argument('--index_path',   default='concept_index.pkl')
    p.add_argument('--n_epochs',     type=int, default=20)
    p.add_argument('--max_images',   type=int, default=5000)
    p.add_argument('--batch_size',   type=int, default=64)
    p.add_argument('--max_samples', type=int, default=3000, help='Max images per concept per split')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading model...")
    model = load_model()

    # ----------------------------------------------------------------
    # Step 1: compute thresholds for every layer
    # Cache to disk — no need to recompute on every run
    # ----------------------------------------------------------------
    thresholds_path = os.path.join(args.results_dir, 'thresholds.pkl')

    if os.path.exists(thresholds_path):
        print("\nLoading cached thresholds...")
        with open(thresholds_path, 'rb') as f:
            thresholds = pickle.load(f)
        print(f"Loaded: {list(thresholds.keys())}")
    else:
        print("\n=== Computing thresholds (one-time) ===")
        thresholds = {}
        for layer in LAYERS:
            thresholds[layer] = compute_thresholds(
                model=model,
                broden_root=args.broden_root,
                layer_name=layer,
                max_images=args.max_images,
                batch_size=args.batch_size,
            )
        with open(thresholds_path, 'wb') as f:
            pickle.dump(thresholds, f)
        print(f"Thresholds saved to {thresholds_path}")

    # ----------------------------------------------------------------
    # Step 2: load existing results if any (resume after interruption)
    # ----------------------------------------------------------------
    results_path = os.path.join(args.results_dir, 'net2vec_results.pkl')

    if os.path.exists(results_path):
        print("\nLoading existing results (resuming)...")
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Already done: {list(results.keys())}")
    else:
        results = {}

    # ----------------------------------------------------------------
    # Step 3: run probes for each concept × layer
    # ----------------------------------------------------------------
    print("\n=== Running Net2Vec probes ===")

    for concept in CONCEPTS:

        # Skip if this concept is fully done
        if concept in results and len(results[concept]) == len(LAYERS):
            print(f"\nSkipping {concept} — already complete")
            continue

        print(f"\n{'='*50}")
        print(f"Concept: {concept}")
        print(f"{'='*50}")

        if concept not in results:
            results[concept] = {}

        # Load datasets once per concept
        try:
            train_set = BrodenConceptDataset(
            args.broden_root, concept,
            split='train', index_path=args.index_path,
            max_samples=args.max_samples
            )
            val_set = BrodenConceptDataset(
            args.broden_root, concept,
            split='val', index_path=args.index_path,
            max_samples=args.max_samples // 3
)
        except ValueError as e:
            print(f"  Skipping — {e}")
            continue

        if len(train_set) < 10:
            print(f"  Skipping — too few examples ({len(train_set)})")
            continue

        for layer in LAYERS:

            # Skip if this layer is already done for this concept
            if layer in results[concept]:
                print(f"  Skipping {layer} — already done")
                continue

            print(f"\n  Layer: {layer}")
            thresh = thresholds[layer]

            # Single-filter: select best on train, evaluate on val
            iou_train = compute_single_filter_iou(
                model, train_set, thresh, layer
            )
            best_k = int(iou_train.argmax())

            iou_val = compute_single_filter_iou(
                model, val_set, thresh, layer
            )
            single_val_iou = float(iou_val[best_k])
            print(f"  Single-filter: best_k={best_k}, "
                  f"val_IoU={single_val_iou:.4f}")

            # Multi-filter: train on train, evaluate on val
            weights, losses = train_multi_filter_probe(
                model=model,
                dataset=train_set,
                thresholds=thresh,
                layer_name=layer,
                n_epochs=args.n_epochs,
            )
            multi_val_iou = evaluate_multi_filter_iou(
                model, val_set, thresh, weights, layer
            )
            print(f"  Multi-filter:  val_IoU={multi_val_iou:.4f}")

            results[concept][layer] = {
                'best_k':          best_k,
                'single_val_iou':  single_val_iou,
                'multi_val_iou':   multi_val_iou,
                'weights':         weights,
                'losses':          losses,
                'all_filter_ious': iou_val,
            }

            # Save after every layer — safe to interrupt anytime
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"  Saved checkpoint.")

    # ----------------------------------------------------------------
    # Step 4: print summary table
    # ----------------------------------------------------------------
    print(f"\n\nDone. Results saved to {results_path}")
    print("\nSummary — single-filter val IoU:")
    print(f"{'Concept':12s}  " + "  ".join(f"{l:7s}" for l in LAYERS))
    print("-" * 65)
    for concept in CONCEPTS:
        if concept not in results:
            continue
        row = f"{concept:12s}  "
        for layer in LAYERS:
            if layer in results[concept]:
                row += f"{results[concept][layer]['single_val_iou']:.4f}   "
            else:
                row += "  ---    "
        print(row)

    print("\nSummary — multi-filter val IoU:")
    print(f"{'Concept':12s}  " + "  ".join(f"{l:7s}" for l in LAYERS))
    print("-" * 65)
    for concept in CONCEPTS:
        if concept not in results:
            continue
        row = f"{concept:12s}  "
        for layer in LAYERS:
            if layer in results[concept]:
                row += f"{results[concept][layer]['multi_val_iou']:.4f}   "
            else:
                row += "  ---    "
        print(row)


if __name__ == '__main__':
    main()