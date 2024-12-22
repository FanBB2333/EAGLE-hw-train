import json
import jsonlines
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
PROJECT_BASE = Path(__file__).resolve().parents[1]
DATASET_BASE = PROJECT_BASE / 'dataset'

class VideoDS(Dataset):
    def __init__(self, name):
        self.name = name
        self.data = list()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ActivityNet(VideoDS):
    def __init__(self, db_path = DATASET_BASE / 'ActivityNetCaps'):
        name = 'ActivityNet'
        super().__init__(name)
        self.db_path = db_path
        self.train = json.load(open(db_path / 'train.json'))
        self.val1 = json.load(open(db_path / 'val_1.json'))
        self.val2 = json.load(open(db_path / 'val_2.json'))
        print(f'ActivityNet {name} loaded')

class Breakfast(VideoDS):
    def __init__(self, db_path = DATASET_BASE / 'breakfast'):
        name = 'Breakfast'
        super().__init__(name)
        self.db_path = db_path
        self.anno_path = self.db_path / "segmentation_coarse"
        self.video_path = self.db_path / "BreakfastII_15fps_qvga_sync"

class Charades(VideoDS):
    def __init__(self, db_path = DATASET_BASE / 'Charades'):
        name = 'Charades'
        super().__init__(name)
        self.db_path = db_path
        self.anno_path = self.db_path
        self.video_path = self.db_path / "Charades_v1_480"

class QVHighlights(VideoDS):
    def __init__(self, db_path = DATASET_BASE / 'QVHighlights'):
        name = 'QVHighlights'
        super().__init__(name)
        self.db_path = db_path
        annotations = self.db_path / 'annotations'
        self.test = self.load_jsonl(annotations / 'highlight_test_release.jsonl')
        self.train = self.load_jsonl(annotations / 'highlight_train_release.jsonl')
        self.val = self.load_jsonl(annotations / 'highlight_val_release.jsonl')
        
    def load_jsonl(self, path):
        data = list()
        with jsonlines.open(path) as reader:
            for obj in reader:
                data.append(obj)
        return data

     

if __name__ == "__main__":
    pass