import json
import jsonlines
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from tqdm import tqdm
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
        # self.video_path = self.db_path / "all_test"
        self.video_path = self.db_path / "v1-2/val"
        print(f"[{name}] length of val1: {len(self.val1)}, val2: {len(self.val2)}")
        self.load_data()
    
    def load_data(self):
        # load from the val1 and val2 data
        splits = [self.val1, self.val2, self.train]
        split = splits[2]
        self.data = list()
        for k, v in split.items():
            # k: id
            # v: {'duration': 55.15, 'timestamps': [[0.28, 55.15], [13.79, 54.32]], 'sentences': ['A weight lifting tutorial is given.', '  The coach helps the guy in red with the proper body placement and lifting technique.']}
            video_file = self.video_path / f"{k}.mp4"
            if not os.path.exists(video_file):
                print(f"File {video_file} does not exist")
                continue
            for idx in range(len(v['sentences'])):
                sentence = v['sentences'][idx]
                timestamps = v['timestamps'][idx]
                self.data.append({
                    'video_file': video_file,
                    'sentence': sentence,
                    'timestamps': timestamps
                })

        print(f"[{self.name}] length of data: {len(self.data)}")
        

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

     
def test():
    import json
    path = "/home6/fzy/repos/EAGLE/dataset/ActivityNetCaps/val_1.json"
    val1 = json.load(open(path))
    print(list(val1.values())[0])

if __name__ == "__main__":
    pass