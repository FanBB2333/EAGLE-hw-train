import torch
from torch.utils.data import Dataset

import os
import json

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


class ConversationDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        ann_path: str,
    ):
        super().__init__()
        self.anns = json.load(open(ann_path))
        self.data_path = data_path

    def __len__(self):
        return len(self.anns)
        
    def __getitem__(self, index: int):
        item = self.anns[index]
        image_id = item["image"]
        image_path = os.path.join(self.data_path, image_id)
        question = item["conversations"][0]["value"].strip()
        gt = item["conversations"][1]["value"]
        if isinstance(gt, list):
            gt = [ans.strip() for ans in gt]
        else:
            gt = gt.strip()
        return image_id, image_path, question, gt
    
    def collate_fn(self, input_):
        return input_
