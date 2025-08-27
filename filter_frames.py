from ultralytics import YOLOWorld
import cv2
import os
import pandas as pd

model = YOLOWorld("yolov8x-worldv2.pt")
model.set_classes([
    "laptop", "phone", "charger", "usb cable", "charging cable", "adapter", "plug",
    "extension cord", "headphones", "earbuds", "airpods", "smartwatch", "power bank", "camera", 'bikini top','deodorant spray', 'deodorant'
    "t-shirt", "shirt", "folded t-shirt", "folded shirt", "pants", "jeans", "leggings",
    "socks", "underwear", "pajamas", "jacket", "sweater", "dress", "scarf",
    "clothes", "folded clothes", "toothbrush", "toothpaste", "deodorant", "makeup", "makeup bag", "cosmetic bag", "toiletry bag",
    "perfume", "razor", "shampoo", "conditioner", "face wash", "serum", "moisturizer", "sunscreen",
    "comb", "hairbrush", "nail clipper", "cotton pads", "toilet paper", "band-aid", "first aid kit",
    "book", "journal", "diary", "notebook", "paper", "magazine", "e-reader", "kindle", "passport",
    "wallet", "backpack", "suitcase", "bag", "ziplock bag", "tupperware", "padlock", "key",
    "travel pillow", "shoes", "sneakers", "flip flops", "sandals", "slippers",
    "snacks", "protein bar", "chocolate", "chips", "sandwich", "bottle", "water bottle", "thermos", "cup", "mug",
    "bikini", "swimsuit", "hat", "sunglasses", "lip balm",
    "medicine", "face mask", "towel", "a travel pillow"
])

frame_folder = "extracted_frames"
output_folder = "filtered_frames_"
os.makedirs(output_folder, exist_ok=True)

seen_objects = set()
frame_summaries = {}
object_to_first_frame = {}  
for filename in sorted(os.listdir(frame_folder)):
    if not (filename.endswith(".jpg") or filename.endswith(".png")):
        continue

    frame_path = os.path.join(frame_folder, filename)
    img = cv2.imread(frame_path)
    if img is None:
        print(f"Failed to read image: {frame_path}")
        continue

    results = model.predict(img, verbose=False)

    current_objects = set()
    if hasattr(results[0], "names") and hasattr(results[0].boxes, "cls"):
        cls_ids = results[0].boxes.cls.tolist()
        current_objects = set([results[0].names[int(i)] for i in cls_ids])


    truly_new_objects = current_objects - seen_objects

    if truly_new_objects:
        for obj in truly_new_objects:
            object_to_first_frame[obj] = filename
        seen_objects.update(truly_new_objects)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)
        frame_summaries[filename] = list(current_objects)
        print(f"{filename} has new objects: {list(truly_new_objects)}")
    else:
        print(f"{filename} skipped (no new objects)")

df_summary = pd.DataFrame(frame_summaries.items(), columns=["Frame", "Detected Objects"])
df_summary.to_csv("frame_object_summary.csv", index=False)

df_objects = pd.DataFrame(object_to_first_frame.items(), columns=["Object", "First Detected In"])
df_objects.to_csv("object_first_seen.csv", index=False)
