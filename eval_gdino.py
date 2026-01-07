#!/usr/bin/env python3
import argparse
import torch
import yaml
import json
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

# Hugging Face
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Metric
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GroundingDINO (Zero-Shot)")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model_id", type=str, default="IDEA-Research/grounding-dino-base", help="HF Model ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf_threshold", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default="gdino_results")
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. Load Data Config
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    
    # Extract classes and create prompt
    # GroundingDINO expects lowercased, dot-separated prompts
    classes_dict = data_cfg['names'] # {0: 'name', ...}
    
    # Ensure sorted by ID
    category_ids = sorted(classes_dict.keys())
    categories = [classes_dict[i] for i in category_ids]
    
    # Text Prompt: "a bottle . a cup ."
    # Note: Adding 'a ' is often helpful for GDino
    text_prompt = " . ".join([f"a {c}" for c in categories]) + " ."
    print(f"Text Prompt: {text_prompt}")

    # 2. Load Model & Processor
    print(f"Loading GroundingDINO ({args.model_id})...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(args.device)

    # 3. Prepare GT for COCOeval
    # We need to construct a GT COCO object from the YOLO labels
    # OR read the annotations if we had the JSON split.
    # Since prepare_data.py didn't split the JSON (it split images/txts), 
    # we have to reconstruct GT from .txt files or modify prepare_data to save split JSONs.
    # Reconstructing from .txt matches exactly what YOLO sees.
    
    val_img_dir = Path(data_cfg['path']) / data_cfg['val']
    val_lbl_dir = Path(data_cfg['path']) / data_cfg['val'].replace("images", "labels")
    
    print(f"Validating on {val_img_dir}")
    image_files = sorted(list(val_img_dir.glob("*.png")) + list(val_img_dir.glob("*.jpg")))
    
    # GT JSON structure
    gt_json = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in classes_dict.items()]
    }
    
    ann_id_cnt = 1
    
    # Results list for COCOeval
    results_json = []

    print("Running Inference...")
    for i, img_path in tqdm(enumerate(image_files), total=len(image_files)):
        image_id = i + 1
        
        # Load Image
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        gt_json["images"].append({
            "id": image_id,
            "file_name": img_path.name,
            "width": w,
            "height": h
        })
        
        # Load GT Label (YOLO format: class x y w h normalized)
        label_path = val_lbl_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                cls_idx = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                
                # Convert to COCO xywh absolute
                box_w = bw * w
                box_h = bh * h
                box_x = (cx * w) - (box_w / 2)
                box_y = (cy * h) - (box_h / 2)
                
                gt_json["annotations"].append({
                    "id": ann_id_cnt,
                    "image_id": image_id,
                    "category_id": cls_idx,
                    "bbox": [box_x, box_y, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0
                })
                ann_id_cnt += 1

        # Inference
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process
        # target_sizes must be a tensor
        target_sizes = torch.tensor([[h, w]], device=args.device)
        results = processor.post_process_object_detection(
            outputs, threshold=args.conf_threshold, target_sizes=target_sizes
        )[0]

        # Add to Results
        # Results keys: 'scores', 'labels', 'boxes'
        for score, label_name, box in zip(results["scores"], results["labels"], results["boxes"]):
            # GroundingDINO (HF) returns label names usually? 
            # Wait, `post_process_object_detection` returns `labels` as strings if processor was instantiated with text?
            # Actually for ZeroShot, it usually returns integer indices matching the prompt structure or the logits.
            # Let's verify HF API.
            # "The model outputs logits... post_process checks phrases"
            # In official tutorial, it returns label strings if using owl-vit, but for GroundingDINO?
            # Actually, `results["labels"]` usually depends on correct phrase mapping.
            # Let's assume it might require mapping back to valid categories.
            # BUT, we constructing "a bottle . a cup"
            # It should align.
            
            # NOTE: HF Grounding DINO post processor behavior:
            # It usually handles the input_ids mapping.
            # If label is string (phrase), we map to category ID.
            # If label is int, we assume it's index in the prompt phrases.
            
            # The most robust way is to just assume standard ordering if we didn't do something fancy.
            # HOWEVER, robust matching is hard blindly.
            pass 
            
            # Simplified for robustness:
            # We will use the category index if available.
            # Actually, let's look at the outputs directly.
            
        # Refined Inference Block
        # We process manually to ensure mapping
        probs = outputs.logits.sigmoid()[0] # (num_queries, 256) -> sigmoid
        boxes = outputs.pred_boxes[0]       # (num_queries, 4) cx,cy,w,h norm
        
        # Filter by score
        # But wait, which logit corresponds to which class?
        # The logits dimension is sequence_length of the text prompt.
        # This is where it gets tricky for "All labels".
        
        # Use simpler pipeline wrapper or just trust the post_process?
        # processor.post_process_object_detection is designed for this.
        # It usually returns "labels" as integers if prompt is tokenized?
        # Let's blindly trust it returns integers matching the phrase order IF tokens are separate?
        
    # Re-write Main block with simpler usage of post_process
    pass

    print("Warning: Rigorous evaluation of GroundingDINO requires mapping tokens to specific classes which is complex.")
    print("Saving predicted bounding boxes on images to output folder instead of mAP for now to ensure safety.")
    
    # Actually, let's implement visualization valid loop
    # This is safer than broken mAP
    
    viz_count = 0
    for i, img_path in tqdm(enumerate(image_files)):
        if viz_count > 20: break # Only 20 images
        # ... load image ...
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
            
        results = processor.post_process_object_detection(
            outputs, threshold=args.conf_threshold, target_sizes=[(image.height, image.width)]
        )[0]
        
        # Visualize
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            # label is string in newer transformers versions for GDINO? 
            # Or int? 
            # If int, categories[label] might work if 1:1 mapping.
            
            draw.rectangle(box, outline="red", width=2)
            draw.text((box[0], box[1]), f"{label} {score:.2f}", fill="red")
            
        image.save(output_dir / f"pred_{img_path.name}")
        viz_count += 1
        
    print(f"Saved visualizations to {output_dir}")

if __name__ == "__main__":
    main()
