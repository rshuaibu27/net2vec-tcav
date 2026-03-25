#!/bin/bash
# Full reproduction script for Google Colab.
# Run this after cloning the repo.
#
# Usage:
#   bash run_colab.sh
#
# Steps:
#   1. Download Broden dataset
#   2. Build concept index
#   3. Run Net2Vec baseline
#   4. Run TCAV
#   5. Generate figures

set -e

BRODEN_ROOT="broden1_227"
RESULTS_DIR="net2vec-tcav/results"
FIGURES_DIR="net2vec-tcav/figures"
INDEX_PATH="net2vec-tcav/concept_index_v2.pkl"

# Step 1: Download Broden 
if [ ! -d "$BRODEN_ROOT" ]; then
    echo "Downloading Broden dataset (~930 MB)..."
    wget -q http://netdissect.csail.mit.edu/data/broden1_227.zip
    unzip -q broden1_227.zip
    rm broden1_227.zip
    echo "Broden downloaded."
else
    echo "Broden already exists — skipping download."
fi

# Step 2: Build concept index
if [ ! -f "$INDEX_PATH" ]; then
    echo "Building concept index..."
    python net2vec-tcav/experiments/build_index.py \
        --broden_root "$BRODEN_ROOT" \
        --save_path "$INDEX_PATH"
else
    echo "Concept index already exists — skipping."
fi

# Step 3: Net2Vec baseline
echo "Running Net2Vec baseline..."
python net2vec-tcav/experiments/run_baseline.py \
    --broden_root "$BRODEN_ROOT" \
    --results_dir "$RESULTS_DIR" \
    --index_path  "$INDEX_PATH" \
    --n_epochs 20 \
    --max_samples 3000 \
    --batch_size 64

# Step 4: TCAV
echo "Running TCAV..."
python net2vec-tcav/experiments/run_tcav.py \
    --broden_root      "$BRODEN_ROOT" \
    --results_dir      "$RESULTS_DIR" \
    --index_path       "$INDEX_PATH" \
    --target_class_idx 817 \
    --n_trials 5 \
    --max_samples 500

# Step 5: Figures
echo "Generating figures..."
python net2vec-tcav/experiments/plot_results.py \
    --results_dir "$RESULTS_DIR" \
    --output_dir  "$FIGURES_DIR"

echo ""
echo "Done. Figures saved to $FIGURES_DIR"