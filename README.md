***youtube link!!***

https://www.youtube.com/watch?v=XxIaPTiDUZA

This project is a computer vision AI project that helps you pack smarter. It looks inside your luggage and checks whether everything on your list is actually there.

### **Here’s the pipeline in simple terms:**

* **YOLO** → spots new objects in video frames, so the model doesn’t have to process the whole packing video. Instead, it detects the frames where a new item has been packed.

* **SAM (Segment Anything)** → the more you pack, the more complicated the image gets, making it harder to identify objects. SAM cleans up the background so that the items become much easier to identify.

* **CLIP** → connects what it sees to words. So when SAM gives a cut-out of a toothbrush, CLIP can say: “yeah, that’s a toothbrush.”

* **List checker** → once everything is recognized, the system checks it against your packing list (e.g. toothbrush, socks, charger) and tells you what’s already packed and what’s still missing.

So basically: 
***YOLO finds it, SAM cleans it, CLIP names it, and the system makes sure you didn’t forget it.***

### **Script Flow**

The repo is broken into scripts, each handling one part of the process:

***extract_frames.py  →  filter_frames.py  →  sam_masks.py  →  identify_objects_CLIP.py***
