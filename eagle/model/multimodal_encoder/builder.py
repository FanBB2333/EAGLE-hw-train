# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is modified from https://github.com/haotian-liu/LLaVA/

import os
from .clip_encoder import CLIPVisionTower, LanguageBindAudioTower, LanguageBindVideoTower, Qwen2AudioTower
from .clip_encoder import Qwen2VLTower
from .clip_encoder import PointBertTower
# from .languagebind_audio_encoder import LanguageBindAudioTower
# from .languagebind_video_encoder import LanguageBindVideoTower
from .multi_backbone_channel_concatenation_encoder import MultiBackboneChannelConcatenationVisionTower
from .convnext_encoder import ConvNextVisionTower
from .hr_clip_encoder import HRCLIPVisionTower
from .vision_models.eva_vit import EVAVITVisionTower
from .sam_encoder import SAMVisionTower
from .pix2struct_encoder import Pix2StructLargeVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    print(f"vision_tower: {vision_tower}")

    if "clip" in vision_tower.lower():
        is_absolute_path_exists = os.path.exists(vision_tower)
        if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
            vision_tower_cfg.freeze_vision = False
            # vision_tower_cfg.input_image_size = 336
            clip_vision_tower = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs) 
            clip_vision_tower.load_model()
            return clip_vision_tower
    elif "eva02" in vision_tower.lower():
        vision_tower_cfg.input_image_size = 1024
        vision_tower_cfg.freeze_vision = False
        vision_tower_cfg.vision_tower_pretrained_from = './model/Vision_Encoder/eva02_L_coco_det_sys_o365.pth'
        return EVAVITVisionTower("eva02-l-16", args=vision_tower_cfg, **kwargs) 
    elif "sam" in vision_tower.lower():
        vision_tower_cfg.freeze_vision = False
        vision_tower_cfg.input_image_size = 1024
        vision_tower_cfg.add_pixel_shuffle = True
        return SAMVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif "pix2struct" in vision_tower.lower():
        vision_tower_cfg.input_image_size = 1024
        vision_tower_cfg.freeze_vision = False
        vision_tower_cfg.do_resize = True
        vision_tower_cfg.de_normalize = True
        return Pix2StructLargeVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    elif ";" in vision_tower:
        return MultiBackboneChannelConcatenationVisionTower(vision_tower, args=vision_tower_cfg)

    elif "qwen2-vl" in vision_tower.lower():
        vision_tower_cfg.freeze_vision = False
        return Qwen2VLTower(vision_tower, args=vision_tower_cfg, **kwargs)


    raise ValueError(f'Unknown vision tower: {vision_tower}')

# BEGIN
# Add audio tower
# def build_audio_tower(audio_tower_cfg, **kwargs):
#     audio_tower = getattr(audio_tower_cfg, 'mm_audio_tower', getattr(audio_tower_cfg, 'audio_tower', None))

#     if 'LanguageBind_Audio_FT' in audio_tower:
#         is_absolute_path_exists = os.path.exists(audio_tower)
        
#         if is_absolute_path_exists:
#             from .audio_models.languagebind_audio import LanguageBindAudio
#             return LanguageBindAudio.from_pretrained(audio_tower, **kwargs).vision_model       
#         raise ValueError(f'Unknown vision tower: {audio_tower}')

#     else:
#         raise NotImplementedError
# # END

# BEGIN qbs
# Add audio tower
def build_audio_tower(vision_tower_cfg, **kwargs):
    audio_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if 'LanguageBind_Audio_FT' in audio_tower:
        is_absolute_path_exists = os.path.exists(audio_tower)
        
        if is_absolute_path_exists:
            return LanguageBindAudioTower(audio_tower, args=vision_tower_cfg, **kwargs)     
        raise ValueError(f'Unknown vision tower: {audio_tower}')

    elif 'Qwen2-Audio' in audio_tower:
        is_absolute_path_exists = os.path.exists(audio_tower)
        
        if is_absolute_path_exists:
            return Qwen2AudioTower(audio_tower, args=vision_tower_cfg, **kwargs)     
        raise ValueError(f'Unknown vision tower: {audio_tower}')

    else:
        raise NotImplementedError

# Add video tower
def build_video_tower(vision_tower_cfg, **kwargs):
    video_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if 'LanguageBind_Video_FT' in video_tower:
        is_absolute_path_exists = os.path.exists(video_tower)
        
        if is_absolute_path_exists:
            return LanguageBindVideoTower(video_tower, args=vision_tower_cfg, **kwargs)     
        raise ValueError(f'Unknown vision tower: {video_tower}')

    else:
        raise NotImplementedError
# END qbs

def build_3d_tower(vision_tower_cfg, **kwargs):
    _3d_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if 'PointBert' in _3d_tower:
        is_absolute_path_exists = os.path.exists(_3d_tower)
        
        if is_absolute_path_exists:
            return PointBertTower(_3d_tower, args=vision_tower_cfg, **kwargs)
        raise ValueError(f'Unknown vision tower: {_3d_tower}')

    else:
        raise NotImplementedError