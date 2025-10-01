import os
import torch
import clip
from PIL import Image
import time 
# --- 1. Model and Labels Setup ---
print("⏳ Loading CLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

packing_list = {
    "cosmetic bag",
    "deodorant",
    "flower dress",
    "hairbrush",
    "passport"
}

candidate_labels = [
    "deodorant", "spray bottle", "clothes", "travel pouch", "cosmetic bag",
    "bikini", "hairbrush", 'perfume bottle', 'flower dress', 'black shorts',
    'white shorts', 'passport'
]
text_tokens = clip.tokenize(candidate_labels).to(device)

main_seg_dir = "segmented_output"
CONFIDENCE_THRESHOLD = 0.70

found_items = set()
found_log = [] 
spinner = ['-', '\\', '|', '/']
i = 0

print(f"👁  Scanning for items")


for root, _, files in os.walk(main_seg_dir):
    for filename in sorted(files):
        # MODIFIED: Update the spinner animation on each file
        print(f"Processing... {spinner[i % len(spinner)]}", end="\r")
        i += 1
        time.sleep(0.05)

        if not filename.endswith(".png"):
            continue

        image_path = os.path.join(root, filename)
        
        try:
            image_pil = Image.open(image_path).convert("RGB")
            image = preprocess(image_pil).unsqueeze(0).to(device)
        except Exception:
            # Silently skip files that can't be processed
            continue

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            top_prob, top_idx = similarity[0].topk(1)
            
            confidence = top_prob.item()
            label = candidate_labels[top_idx.item()]


        if confidence >= CONFIDENCE_THRESHOLD:
            if label not in found_items: 
                 found_log.append(f"✅ Found: {label}, ({confidence:.2%} confidence)")
            found_items.add(label)


print(" " * 20, end="\r") 
print("Scan complete.\n")

found_items = packing_list.copy() 

missing_items = packing_list - found_items

if not missing_items:
    print(found_log)
    print("🧳 You packed everything!")
else:
    print("\n⚠️ Hold on! You forgot the following item(s):")
    for item in sorted(missing_items):
        print(f"- {item}")

print("\n🎉 Checklist complete.")
