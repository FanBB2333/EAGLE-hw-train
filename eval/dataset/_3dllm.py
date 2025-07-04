import torch
from torch.utils.data import Dataset

import json
import os

from eval.utils import VideoDataInput

class ThreeDLLMDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.video_path = './dataset/3D-LLM/3dllm_final_scene_data_v2/point_clouds_pcd_videos'
        self.json_data = json.load(open('dataset/3D-LLM/100_part2_scene.json'))

    def __len__(self):
        return len(self.json_data)
    
    def __getitem__(self, index) -> VideoDataInput:
        data_dict = self.json_data[index]
        video_name = str(data_dict['scene_id'])
        question = data_dict['question']
        answer = data_dict['answers'][0]
        return VideoDataInput(
            data_path=os.path.join(self.video_path, video_name + '.mp4'),
            question=question,
            answer=answer,
            question_id=video_name
        )
    def collate_fn(self, input):
        return input
    