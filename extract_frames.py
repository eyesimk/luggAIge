import cv2
import os

video_path = "videos/packing_video.mp4"
output_folder = "extracted_frames"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("Failed to open video.")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps) 

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Saved {saved_count} high-quality frames to '{output_folder}'")
