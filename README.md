# Visual Fine-Tuning Pipeline

This project provides tools to fine-tune YOLO-World and evaluate Grounding DINO on the generated synthetic dataset.

## Project Structure

This directory (`visual_ft/`) is part of the larger Tiamat codebase. It requires the dataset to be generated in sibling directories.

```
Tiamat/
├── scripts/                  # Data Generation (sdg_pipeline.py)
├── sdg_output/               # Output from quick test runs
├── sdg_dataset/              # Output from full generation runs (Main Target)
│   ├── images/
│   └── annotations/
└── visual_ft/                # YOU ARE HERE (Fine-tuning pipeline)
    ├── data/                 # Local workspace (created by prepare_data.py)
    ├── prepare_data.py       # Data management script
    ├── train_yolo.py         # Training script
    └── rules/                # Configuration files
```

## Setup

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Login to WandB**:
   ```bash
   wandb login
   # Paste your API key when prompted
   ```

## Workflow

### 1. Quick Start (Using Helper Scripts)
We provide helper scripts for typical workflows. These are also used as entry points for Docker containers.

**Standard Fine-Tuning** (SDG Data Only):
```bash
./run_finetune.sh
```

**Choose Model Version**:
You can select different YOLO-World versions by setting the `MODEL` environment variable.
Available options: `yolov8s-world.pt` (Small), `yolov8m-world.pt` (Medium), `yolov8l-world.pt` (Large), `yolov8x-world.pt` (XLarge).

```bash
MODEL=yolov8m-world.pt ./run_finetune.sh
```

**Co-Fine-Tuning** (SDG + COCO Data):
```bash
./run_co_finetune.sh
```

*Note: These scripts automatically search for `../sdg_dataset` or `../sdg_output`. You can override the location by setting the `SDG_DATA_DIR` environment variable.*

### 2. Manual Workflow

#### A. Prepare Data
Extracts the raw SDG output, splits it into Train/Val/Test, and formats it for YOLO.

**Option A: Standard SDG Fine-Tuning**
Only uses your synthetic data.
```bash
python prepare_data.py --source_dir ../sdg_output --output_dir data
```

**Option B: Co-Fine-Tuning (SDG + COCO)**
Mixes your synthetic data with real-world images (COCO) to improve generalizability (inspired by RT-2).
This will automatically download COCO val2017 if needed.
```bash
python prepare_data.py --source_dir ../sdg_output --output_dir data --co_fine_tune
```

### 2. Fine-tune YOLO-World
Fine-tunes the open-vocabulary YOLO-World model on your specific classes.

**Run SDG-Only Fine-Tuning:**
```bash
python train_yolo.py --data data/data.yaml --name yolo_sdg_only
```

**Run Co-Fine-Tuning:**
(Requires running Option B in step 1 first)
```bash
python train_yolo.py --data data/data_co_ft.yaml --name yolo_co_ft
```

### 3. Evaluate Grounding DINO
Runs Zero-Shot evaluation using the same class list.
Currently outputs visualizations for verification.

```bash
python eval_gdino.py --data data/data.yaml
```

## Directory Structure
- `data/`: Processed dataset (images symlinked to save space).
- `runs/`: Training outputs (weights, logs).
- `gdino_results/`: Visualizations from Grounding DINO.
