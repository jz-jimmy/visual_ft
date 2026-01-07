#!/bin/bash
set -e

# Configuration
# Allow override via environment variable, else check default locations
if [ -n "$SDG_DATA_DIR" ]; then
    SOURCE_DIR="$SDG_DATA_DIR"
elif [ -d "../sdg_dataset" ]; then
    SOURCE_DIR="../sdg_dataset"
elif [ -d "../sdg_output" ]; then
    SOURCE_DIR="../sdg_output"
else
    echo "Error: Could not find sdg_dataset or sdg_output directory."
    echo "Please set SDG_DATA_DIR environment variable or ensure data is in ../sdg_dataset"
    exit 1
fi

echo "========================================================"
echo "Starting Standard Visual Fine-Tuning (SDG Only)"
echo "Source Data: $SOURCE_DIR"
echo "========================================================"

# Step 1: Prepare Data
echo "[1/2] Preparing data..."
python3 prepare_data.py --source_dir "$SOURCE_DIR" --output_dir "data"

# Step 2: Run Training
echo "[2/2] Running YOLO Training..."
python3 train_yolo.py \
    --data "data/data.yaml" \
    --epochs 50 \
    --batch 16 \
    --name "yolo_sdg_finetune" \
    --project "visual_fine_tune"

echo "Fine-tuning complete!"
