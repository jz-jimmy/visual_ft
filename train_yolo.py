#!/usr/bin/env python3
import argparse
from ultralytics import YOLOWorld
import wandb
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO-World")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument("--project", type=str, default="visual_fine_tune", help="WandB project name")
    parser.add_argument("--name", type=str, default="yolo_world_finetune", help="Run name")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if data exists
    if not os.path.exists(args.data):
        print(f"Error: {args.data} not found. Please run prepare_data.py first.")
        return

    # Initialize WandB
    # Ultralytics will automatically log to WandB if installed, 
    # but we initialize here to control project name and config.
    wandb.init(project=args.project, name=args.name, job_type="training", config=vars(args))

    # Load a pretrained YOLOv8s-World model
    print("Loading YOLOv8s-World model...")
    model = YOLOWorld('yolov8s-world.pt')  # or yolov8m-world.pt, l, x

    # 1. Set the classes from data.yaml (Offline Vocabulary)
    # Actually, when training, we just pass the data argument, and Ultralytics
    # automatically sets up the head for the classes in data.yaml.
    # But explicitly setting them ensures the embeddings are ready.
    # We need to read classes from data.yaml ourselves if we want to call set_classes before training,
    # but strictly speaking model.train() handles it.
    
    # Train the model
    # Key arguments for Production/Quality:
    # - close_mosaic: Disables mosaic augmentation in last few epochs for better convergence
    # - workers: data loading threads
    # - patience: Early stopping
    print(f"Starting training on {args.device}...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,          # Save plots
        save=True,           # Save checkpoints
        exist_ok=True,       # Overwrite existing run dir
        close_mosaic=10,     # Disable mosaic for last 10 epochs
        optimizer='AdamW',   # High quality optimizer
        lr0=0.001,           # Initial LR
    )

    print("Training Complete.")
    print(f"Best map50-95: {results.box.map}")

    # Validation & Visualization
    print("Running Validation and Visualization...")
    # The val() method runs validation on the split defined in yaml
    val_results = model.val(
        data=args.data,
        split='val',
        plots=True # Generates confusion matrix, PR curve, etc.
    )
    
    # Save a few specific inferences for visual check
    # Pick a few images from validation set
    import yaml
    import glob
    import random
    
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    
    val_images_path = Path(data_cfg['path']) / data_cfg['val']
    val_images = list(glob.glob(str(val_images_path / "*.png"))) + list(glob.glob(str(val_images_path / "*.jpg")))
    
    if val_images:
        sample_images = random.sample(val_images, min(5, len(val_images)))
        print(f"Predicting on {len(sample_images)} sample images...")
        model.predict(
            sample_images, 
            save=True, 
            project=args.project, 
            name=f"{args.name}_vis", 
            conf=0.25
        )

    wandb.finish()

if __name__ == "__main__":
    main()
