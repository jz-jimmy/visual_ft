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

COCO_RATIO=1.0  # Mix 1:1 ratio of COCO to SDG data

echo "========================================================"
echo "Starting Co-Fine-Tuning (SDG + COCO)"
echo "Source Data: $SOURCE_DIR"
echo "COCO Ratio: $COCO_RATIO"
echo "========================================================"

# Step 1: Prepare Data with Co-FT enabled
echo "[1/2] Preparing data (downloading/mixing COCO)..."
python3 prepare_data.py \
    --source_dir "$SOURCE_DIR" \
    --output_dir "data" \
    --co_fine_tune \
    --coco_ratio $COCO_RATIO

# Step 2: Run Training with merged config
echo "[2/2] Running YOLO Training..."
python3 train_yolo.py \
    --data "data/data_co_ft.yaml" \
    --epochs 50 \
    --batch 16 \
    --name "yolo_co_finetune" \
    --project "visual_fine_tune"

echo "Co-Fine-tuning complete!"
