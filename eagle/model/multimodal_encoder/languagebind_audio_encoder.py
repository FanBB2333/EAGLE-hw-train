# USELESS
import torch
import torch.nn as nn

from .audio_models.languagebind_audio import LanguageBindAudio, LanguageBindAudioConfig

class LanguageBindAudioTower(nn.module):
    def __init__(
        self,
        audio_tower,
        **kwargs
    ):
        super().__init__()
        self.is_loaded = False
        self.load_audio_tower(audio_tower, **kwargs)

    def load_audio_tower(self, audio_tower, **kwargs):
        self.audio_tower = LanguageBindAudio.from_pretrained(audio_tower, **kwargs)
        self.is_loaded = True

    def load_model(self):
        assert self.is_loaded, "All the audio encoders should be loaded during initialization!"

