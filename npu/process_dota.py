import json
import random
random.seed(42)
import jsonlines
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from datasets import load_dataset
import torchvision.transforms as transforms
import cv2
from pathlib import Path

DATA_BASE = Path("xxx")  # Replace with the actual base path
DOTA_METADATA_DIR = DATA_BASE / "DoTA_annotations"
DOTA_DIR = DATA_BASE / "DoTA"
DOTA_CLEAN_DIR = DATA_BASE / "DoTA_CLEAN"
#   "0RJPQ_97dcs_001437": {
#     "video_start": 1437,
#     "video_end": 1547,
#     "anomaly_start": 37,
#     "anomaly_end": 49,
#     "anomaly_class": "ego: pedestrian",
#     "num_frames": 111,
#     "subset": "train"
#   },

sample_format_lambda = lambda question, answer, image_path, id, path_prefix="videos": {
    'id': str(id),
    'conversations': [
        {'from': 'human', 'value': question},
        {'from': 'gpt', 'value': answer}
    ],
    'image': image_path, # /home1/hxl/disk/EAGLE/qbs/Eagle_LanguageBind/dataset/Video/added/ActivityNetCaps/v1-3/train_val/v_ehGHCYKzyZ8.mp4
    # 'image': f"{path_prefix}/{image_path.split('/')[-1]}" # v_ehGHCYKzyZ8.mp4
}


class DotaDataset(Dataset):
    def __init__(self, metadata_file):
        self.metadata = self.load_data(metadata_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        self.data = list()
        for video_id, video_info in data.items():
            info = deepcopy(video_info)
            info['video_path'] = DOTA_DIR / video_id / "video.mp4"
            info['video_id'] = video_id
            if not info['video_path'].exists():
                print(f"Video path {info['video_path']} does not exist, skipping.")
                continue
            self.data.append(info)
        return self.data
    
    
    def to_qa(self):
        # construct question-answer pairs from the metadata, question about the anomaly type in the video
        qa_pairs = []
        for video_info in self.data:
            video_id = video_info['video_id']
            question = f"<image>\nWhat is the anomaly type in the video?"
            answer = video_info['anomaly_class']
            image_path = str(video_info['video_path'])
            qa_pairs.append(sample_format_lambda(question, answer, image_path, video_id))
        with open(DOTA_CLEAN_DIR / "dota_qa_pairs.json", 'w') as f:
            json.dump(qa_pairs, f, indent=4)
        return qa_pairs
    
    def to_loc(self):
        # question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{data["question"]}" in this video, the event starts at'
        loc_pairs = []
        for video_info in self.data:
            video_id = video_info['video_id']
            duration = video_info['num_frames'] / video_info['fps']
            question1 = f'The video\'s duration is {duration}s. Please predict the start time of the event "{video_info["anomaly_class"]}" in this video, the event starts at'
            loc_pairs.append(sample_format_lambda(question1, "", "", video_id))
        with open(DOTA_CLEAN_DIR / "dota_loc_pairs.json", 'w') as f:
            json.dump(loc_pairs, f, indent=4)
        return loc_pairs



    def get_video_frames(self, video_path, start_frame, end_frame):
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        for frame_num in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_num} from {video_path}")
                continue
            frames.append(frame)
        cap.release()
        return frames
            

if __name__ == "__main__":
    train_data = DOTA_METADATA_DIR / "metadata_train.json"
    test_data = DOTA_METADATA_DIR / "metadata_test.json"
    
    