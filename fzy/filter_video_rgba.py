import cv2
import os
from tqdm import tqdm
from PIL import Image
import json
from multiprocessing import Pool, cpu_count

input_file = "/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/dataset/Video/videollava_sft/videochatgpt_llavaimage_tune_opencv_image_first_token.json"
video_base = "/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/dataset/Video/videollava_sft"
output_file = "/home1/hxl/disk2/Backup/EAGLE/qbs/Eagle_LanguageBind/dataset/Video/videollava_sft/videochatgpt_llavaimage_tune_opencv_image_first_token_rgba.json"


with open(input_file, 'r') as f:
    data = json.load(f)

def valid_video(video_path):
    max_frames = 8
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return False
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    frame_count = 0
    fps_interval = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / (max_frames-1) ))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % fps_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            # print pil shape
            # print(f"Frame {frame_count}: shape {pil_image.size}")
            # test whether the image is RGBA
            if pil_image.mode != 'RGBA':
                print(f"Frame {frame_count} is not RGBA, converting...")
    if total_frames < 8:
        return False
    return True


if __name__ == "__main__":
    filtered_data = []
    for item in tqdm(data):
        video_path = item['image']
        video_path = os.path.join(video_base, video_path)
        if valid_video(video_path):
            filtered_data.append(item)
        else:
            print(f"Invalid video: {video_path}")
    print(f"Filtered {len(data) - len(filtered_data)} invalid videos.")
    # with open(output_file, 'w') as f:
    #     json.dump(filtered_data, f, indent=4)
    
