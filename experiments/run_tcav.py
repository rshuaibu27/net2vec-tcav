import os
import sys
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import load_model
from src.dataset import BrodenConceptDataset
from src.tcav import compute_tcav_with_significance

CONCEPTS = ['sky', 'grass', 'building', 'person',
            'car', 'wheel', 'water', 'wood']

LAYERS = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--broden_root',     default='broden1_227')
    p.add_argument('--results_dir',     default='results')
    p.add_argument('--index_path',      default='concept_index_v2.pkl')
    p.add_argument('--target_class_idx',type=int, default=817)
    p.add_argument('--n_trials',        type=int, default=5)
    p.add_argument('--max_samples',     type=int, default=500)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    print("Loading model...")
    model = load_model()

    # Load existing results if any (resume support)
    results_path = os.path.join(args.results_dir, 'tcav_results.pkl')
    if os.path.exists(results_path):
        print("Loading existing TCAV results (resuming)...")
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        print(f"Already done: {list(results.keys())}")
    else:
        results = {}

    for concept in CONCEPTS:
        if concept not in results:
            results[concept] = {}

        print(f"\n{'='*50}")
        print(f"Concept: {concept}")
        print(f"{'='*50}")

        # Load concept dataset
        try:
            dataset = BrodenConceptDataset(
                args.broden_root, concept,
                split='train',
                index_path=args.index_path,
                max_samples=args.max_samples,
            )
        except ValueError as e:
            print(f"  Skipping — {e}")
            continue

        if len(dataset) < 10:
            print(f"  Skipping — too few images ({len(dataset)})")
            continue

        for layer in LAYERS:

            # Skip if already done
            if layer in results[concept]:
                print(f"  Skipping {layer} — already done")
                continue

            print(f"\n  Layer: {layer}")

            result = compute_tcav_with_significance(
                model=model,
                concept_dataset=dataset,
                broden_root=args.broden_root,
                layer_name=layer,
                target_class_idx=args.target_class_idx,
                n_trials=args.n_trials,
                max_samples=args.max_samples,
            )

            results[concept][layer] = result
            sig = '✓' if result['is_significant'] else '✗'
            print(f"  TCAV={result['mean_tcav_score']:.3f} "
                  f"± {result['std']:.3f}  significant: {sig}")

            # Save checkpoint after every layer
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)

    # Print summary table
    print(f"\n\nDone. Results saved to {results_path}")
    print(f"\nTCAV scores (target class {args.target_class_idx}):")
    print(f"{'Concept':12s}  " + "  ".join(f"{l:7s}" for l in LAYERS))
    print("-" * 65)
    for concept in CONCEPTS:
        if concept not in results:
            continue
        row = f"{concept:12s}  "
        for layer in LAYERS:
            if layer in results[concept]:
                score = results[concept][layer]['mean_tcav_score']
                row += f"{score:.3f}    "
            else:
                row += "  ---    "
        print(row)


if __name__ == '__main__':
    main()