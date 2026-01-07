#!/usr/bin/env python3
import json
import os
import argparse
import random
import shutil
import zipfile
import urllib.request
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import yaml

COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_TRAINVAL_ANNS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for Visual Fine-Tuning")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to sdg_dataset or sdg_output")
    parser.add_argument("--output_dir", type=str, default="data", help="Where to output organized data")
    parser.add_argument("--split_ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1], help="Train/Val/Test ratios")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--co_fine_tune", action="store_true", help="Enable Co-Fine-Tuning with COCO dataset")
    parser.add_argument("--coco_dir", type=str, default="coco_data", help="Directory to download/store COCO data")
    parser.add_argument("--coco_ratio", type=float, default=1.0, help="Ratio of COCO images to use relative to SDG (default 1.0 = equal amount)")
    return parser.parse_args()

def download_file(url, target_path):
    if target_path.exists():
        return
    print(f"Downloading {url} to {target_path}...")
    with tqdm(unit='B', unit_scale=True, miniters=1, desc=target_path.name) as t:
        urllib.request.urlretrieve(url, target_path, reporthook=lambda b, bsize, tsize: t.update(bsize))

def setup_coco(coco_dir):
    """Download and extract COCO Val2017 data if needed."""
    coco_path = Path(coco_dir)
    coco_path.mkdir(exist_ok=True, parents=True)
    
    img_zip = coco_path / "val2017.zip"
    ann_zip = coco_path / "annotations_trainval2017.zip"
    
    # Download
    try:
        if not (coco_path / "val2017").exists():
            download_file(COCO_VAL_IMAGES_URL, img_zip)
            print("Extracting images...")
            with zipfile.ZipFile(img_zip, 'r') as zf:
                zf.extractall(coco_path)
                
        if not (coco_path / "annotations").exists():
            download_file(COCO_TRAINVAL_ANNS_URL, ann_zip)
            print("Extracting annotations...")
            with zipfile.ZipFile(ann_zip, 'r') as zf:
                zf.extractall(coco_path)
    except Exception as e:
        print(f"Error setting up COCO: {e}")
        # Clean up partials
        if img_zip.exists(): img_zip.unlink()
        if ann_zip.exists(): ann_zip.unlink()
        raise

    return coco_path / "val2017", coco_path / "annotations" / "instances_val2017.json"

def convert_coco_bbox_to_yolo(bbox, img_width, img_height):
    """
    COCO: [x_min, y_min, width, height]
    YOLO: [x_center, y_center, width, height] (normalized)
    """
    x_min, y_min, w, h = bbox
    
    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return [x_center, y_center, w_norm, h_norm]

def main():
    args = parse_args()
    random.seed(args.seed)

    source_path = Path(args.source_dir)
    output_path = Path(__file__).parent / args.output_dir
    output_path.mkdir(exist_ok=True, parents=True)

    # --- 1. Load SDG Data ---
    ann_file = source_path / "annotations" / "instances_sdg.json"
    if not ann_file.exists():
        ann_file = source_path / "annotations" / "instances.json"
    if not ann_file.exists():
        ann_file = source_path / "dataset" / "_annotations.coco.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"Could not find instances_sdg.json in {source_path}")

    print(f"Loading SDG annotations from {ann_file}...")
    with open(ann_file, 'r') as f:
        sdg_coco = json.load(f)

    # --- 2. Load COCO Data (Optional) ---
    coco_instances = {'images': [], 'annotations': [], 'categories': []}
    coco_img_dir = None
    
    if args.co_fine_tune:
        print("\n[Co-Fine-Tuning Mode Enabled]")
        coco_img_dir, coco_ann_file = setup_coco(args.coco_dir)
        print(f"Loading COCO annotations from {coco_ann_file}...")
        with open(coco_ann_file, 'r') as f:
            coco_instances = json.load(f)

    # --- 3. Merge Categories ---
    # We want to preserve SDG IDs if possible, or just create a master list
    # RT-2 Style: We train on Union of vocabularies.
    
    sdg_cats = {c['id']: c['name'] for c in sdg_coco['categories']}
    coco_cats = {c['id']: c['name'] for c in coco_instances.get('categories', [])}
    
    # Create unified category list (Name -> ID)
    # Priority: SDG categories first (keep them at 0..N)
    combined_cats = []
    seen_names = set()
    
    # Add SDG
    sorted_sdg_ids = sorted(sdg_cats.keys())
    for cid in sorted_sdg_ids:
        name = sdg_cats[cid]
        combined_cats.append(name)
        seen_names.add(name)
        
    # Add COCO (if name not already present)
    sorted_coco_ids = sorted(coco_cats.keys())
    for cid in sorted_coco_ids:
        name = coco_cats[cid]
        if name not in seen_names:
            combined_cats.append(name)
            seen_names.add(name)
            
    print(f"\nCombined Vocabulary Size: {len(combined_cats)}")
    print(f"SDG specific: {len(sdg_cats)}, Total unique: {len(combined_cats)}")
    
    # Map from original ID to New Index
    cat_name_to_idx = {name: i for i, name in enumerate(combined_cats)}
    
    sdg_id_map = {cid: cat_name_to_idx[name] for cid, name in sdg_cats.items()}
    coco_id_map = {cid: cat_name_to_idx[name] for cid, name in coco_cats.items() if name in cat_name_to_idx}

    # Save classes list
    with open(output_path / "classes.txt", "w") as f:
        f.write("\n".join(combined_cats))

    # --- 4. Process Images & Labels ---
    
    # Helper to process a dataset
    def process_dataset(dataset_coco, img_source_dir, id_map, prefix="sdg"):
        # Group Anns
        img_to_anns = defaultdict(list)
        for ann in dataset_coco['annotations']:
            img_to_anns[ann['image_id']].append(ann)
            
        images = dataset_coco['images']
        
        # Filter: If Co-FT, maybe limit COCO size?
        if prefix == "coco":
            # Target size: Ratio * SDG Size
            target_size = int(len(sdg_coco['images']) * args.coco_ratio)
            # COCO val is 5000. If SDG is 20k, we might want all COCO.
            # If SDG is 100, we don't want 5000 COCO.
            # But COCO is regularization. 
            if len(images) > target_size:
                print(f"Subsampling COCO from {len(images)} to {target_size} (Ratio {args.coco_ratio})")
                random.shuffle(images)
                images = images[:target_size]
        else:
            random.shuffle(images)

        # Split
        n_total = len(images)
        n_train = int(n_total * args.split_ratio[0])
        n_val = int(n_total * args.split_ratio[1])
        
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train+n_val],
            'test': images[n_train+n_val:]
        }
        
        for split_name, split_images in splits.items():
            if not split_images: continue
            
            split_img_dir = output_path / "images" / split_name
            split_lbl_dir = output_path / "labels" / split_name
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_lbl_dir.mkdir(parents=True, exist_ok=True)
            
            for img_info in tqdm(split_images, desc=f"Processing {prefix} {split_name}"):
                fname = img_info['file_name']
                
                # Resolve Source Path
                if prefix == "sdg":
                    # Try heuristic paths
                    src = img_source_dir / "images" / fname
                    if not src.exists(): src = img_source_dir / "dataset" / "images" / fname
                    if not src.exists(): src = img_source_dir / fname
                else:
                    src = img_source_dir / fname
                    
                if not src.exists():
                    continue

                # Unique Filename for destination (avoid collision if SDG and COCO share names)
                dst_name = f"{prefix}_{fname}"
                dst_img_path = split_img_dir / dst_name

                # Symlink
                if dst_img_path.exists(): dst_img_path.unlink()
                try:
                    os.symlink(src.absolute(), dst_img_path)
                except OSError:
                    shutil.copy2(src, dst_img_path)
                    
                # Annotations
                anns = img_to_anns.get(img_info['id'], [])
                lines = []
                for ann in anns:
                    if ann['category_id'] not in id_map: continue
                    
                    new_idx = id_map[ann['category_id']]
                    bbox = convert_coco_bbox_to_yolo(ann['bbox'], img_info['width'], img_info['height'])
                    
                    # Clamp
                    bbox = [max(0.0, min(1.0, x)) for x in bbox]
                    lines.append(f"{new_idx} {' '.join(f'{x:.6f}' for x in bbox)}")
                    
                with open(split_lbl_dir / f"{Path(dst_name).stem}.txt", 'w') as f:
                    f.write("\n".join(lines))

    # Run Processing
    print("\n--- Processing SDG Dataset ---")
    process_dataset(sdg_coco, source_path, sdg_id_map, "sdg")
    
    if args.co_fine_tune and coco_img_dir:
        print("\n--- Processing COCO Dataset ---")
        process_dataset(coco_instances, coco_img_dir, coco_id_map, "coco")

    # 5. Generate data.yaml
    yaml_name = "data_co_ft.yaml" if args.co_fine_tune else "data.yaml"
    data_yaml = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(combined_cats)}
    }
    
    with open(output_path / yaml_name, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"\nPreparation complete. Configuration: {output_path}/{yaml_name}")
