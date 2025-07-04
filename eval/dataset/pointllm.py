import torch
from torch.utils.data import Dataset

import json
import os

from eval.utils import VideoDataInput

class PointLLMDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.video_path = './dataset/PointLLM/video'
        self.json_data = json.load(open('dataset/PointLLM/try.json'))

    def __len__(self):
        return len(self.json_data)
    
    def __getitem__(self, index) -> VideoDataInput:
        data_dict = self.json_data[index]
        video_name = data_dict['object_id']
        question = data_dict['conversations'][0]['value']
        answer = data_dict['conversations'][1]['value']
        return VideoDataInput(
            data_path=os.path.join(self.video_path, video_name + '.mp4'),
            question=question,
            answer=answer
        )
    def collate_fn(self, input):
        return input
    