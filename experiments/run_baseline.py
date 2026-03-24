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
    p.add_argument('--n_epochs',     type=int, default=30)
    p.add_argument('--max_images',   type=int, default=10000)
    p.add_argument('--batch_size',   type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading model")
    model = load_model()

    print("\nComputing thresholds")
    thresholds = {}
    for layer in LAYERS:
        thresholds[layer] = compute_thresholds(
            model=model,
            broden_root=args.broden_root,
            layer_name=layer,
            max_images=args.max_images,
            batch_size=args.batch_size,
        )

    print("\nRunning Net2Vec probes")
    results = {}

    for concept in CONCEPTS:
        print(f"\n{'='*50}")
        print(f"Concept: {concept}")
        print(f"{'='*50}")
        results[concept] = {}

        try:
            train_set = BrodenConceptDataset(
                args.broden_root, concept, split='train'
            )
            val_set = BrodenConceptDataset(
                args.broden_root, concept, split='val'
            )
        except ValueError as e:
            print(f"  Skipping — {e}")
            continue

        if len(train_set) < 10:
            print(f"  Skipping — too few training examples ({len(train_set)})")
            continue

        for layer in LAYERS:
            print(f"\n  Layer: {layer}")
            thresh = thresholds[layer]

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

            # Save after every concept × layer in case of interruption
            save_path = os.path.join(args.results_dir, 'net2vec_results.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)

    print(f"\n\nDone. Results saved to {save_path}")
    print("\nSummary (val IoU):")
    print(f"{'Concept':12s}  " + "  ".join(f"{l:6s}" for l in LAYERS))
    print("-" * 60)
    for concept in CONCEPTS:
        if concept not in results:
            continue
        row = f"{concept:12s}  "
        for layer in LAYERS:
            if layer in results[concept]:
                row += f"{results[concept][layer]['single_val_iou']:.4f}  "
            else:
                row += "  ---   "
        print(row)


if __name__ == '__main__':
    main()