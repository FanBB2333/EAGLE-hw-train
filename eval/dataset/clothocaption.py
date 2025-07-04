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


class ClothoCapsDataset(Dataset):
    def __init__(
        self,
    ):
        super().__init__()
        audio_dir = "OneLLM/datasets/Eval/audio/clothov2/evaluation/"
        self.audio_anns = json.load(open("OneLLM/datasets/Eval/audio/clothov2/eval_clothocap_ann.json"))
        self.audio_ids = [x['id'] for x in self.audio_anns['images']]
        self.audio_names = [x['file_name'] for x in self.audio_anns['images']]
        self.audio_files = [os.path.join(audio_dir, x) for x in self.audio_names]

    def __len__(self):
        return len(self.audio_files)
        
    def __getitem__(self, index: int):
        audio_file = self.audio_files[index]
        return audio_file, self.audio_names[index], self.audio_ids[index]
    
    def collate_fn(self, input_):
        return input_
