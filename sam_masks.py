import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
from PIL import Image
import glob

print("Loading SAM model...")
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        pred_iou_thresh=0.92,
        stability_score_thresh=0.95,
        min_mask_region_area=200,
    )
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: SAM checkpoint file not found at '{sam_checkpoint}'")
    print("Please download the model checkpoint and place it in the correct directory.")
    exit()

input_folder = ""
output_folder = "tarchi_masks"
os.makedirs(output_folder, exist_ok=True)

image_paths = sorted(glob.glob(os.path.join(input_folder, '*.jpg')))
if not image_paths:
    print(f"No JPG files found in '{input_folder}'. Please check the path.")
    exit()

print(f"Found {len(image_paths)} JPG files to process.")

mask_counter = 0

for image_path in image_paths:
    print(f"\nProcessing image: {os.path.basename(image_path)}")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Could not read image, skipping: {image_path}")
        continue
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("... generating masks ...")
    
    masks = mask_generator.generate(image_rgb)
    print(f"... found {len(masks)} total masks.")

    if not masks:
        print("No masks found for this image, skipping.")
        continue

    top_n = 5
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)[:top_n]
    print(f"... using top {top_n} masks with the largest area.")

    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']  
        bbox = mask_data['bbox']

        x, y, w, h = bbox

        segmented_item_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        image_crop = image_rgb[y:y+h, x:x+w]
        mask_crop = mask[y:y+h, x:x+w]

        segmented_item_rgba[:, :, :3] = image_crop
        segmented_item_rgba[:, :, 3] = mask_crop * 255

    
        item_image = Image.fromarray(segmented_item_rgba)
        
    
        save_path = os.path.join(output_folder, f"item_{mask_counter:04d}.png")
        item_image.save(save_path)
        
        mask_counter += 1

    print(f"Saved {len(masks)} segmented items from this image.")

print("\n" + "="*40)
print(f"All images processed successfully!")
print(f"Total masks saved: {mask_counter}")
print("="*40)