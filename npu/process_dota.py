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
        pass
    
    def to_loc(self):
        pass
    
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
    
    