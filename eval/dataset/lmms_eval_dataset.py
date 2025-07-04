import torch
from torch.utils.data import Dataset

import os

# from eval.utils import VideoDataInput

import json
from dataclasses import dataclass
from typing import Dict, Sequence, Any

import pandas

import torch
from torch.utils.data import Dataset

from eagle.model.multimodal_encoder.audio_models.languagebind_audio import LanguageBindAudio
from eagle.model.multimodal_encoder.audio_models.processing_audio import LanguageBindAudioProcessor
from eagle.model.multimodal_encoder.audio_models.tokenization_audio import LanguageBindAudioTokenizer
from eagle.model.multimodal_encoder.audio_models.configuration_audio import LanguageBindAudioConfig

from eagle import conversation as conversation_lib
from eagle.constants import IGNORE_INDEX
from eagle.conversation import conv_templates
from eagle.datasets.utils import get_input_ids_len, make_label

from lmms_eval.tasks import TaskManager, get_task_dict


class LMMsEvalDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        task_name: str,
        split_name: str,
        transform: Any = None
    ):
        super().__init__()
        task_manager = TaskManager("INFO", model_name="eagle")
        task_dict = get_task_dict([task_name], task_manager)
        self.dataset = task_dict[task_name].dataset[split_name]
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index: int):
        item = self.dataset[index]
        if self.transform:
            item = self.transform(self.data_path, item)
        return item
    
    def collate_fn(self, input_):
        return input_

def clotho_aqa_tranform(data_path, item):
    audio_id = item["audio"]["path"]
    audio_file = os.path.join(data_path, audio_id)
    question = item["question"].strip() + "\nAnswer the question using a single word only."
    gt = item["answer"].strip().lower()
    return audio_id, audio_file, question, gt