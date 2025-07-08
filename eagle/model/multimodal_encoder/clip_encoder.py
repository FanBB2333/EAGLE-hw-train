# This file is modified from https://github.com/haotian-liu/LLaVA/
from typing import Optional
import librosa
# import objaverse
import numpy as np
from packaging import version
from PIL import Image
import os
import transformers
import torch
import torch.nn as nn
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLModel,
    # Qwen2VisionTransformerPretrainedModel,
    # Qwen2_5_VLForConditionalGeneration,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, Qwen2AudioConfig
from eagle.model.multimodal_encoder.languagebind import LanguageBindAudio, LanguageBindAudioTokenizer, LanguageBindAudioProcessor
from eagle.model.multimodal_encoder.languagebind import LanguageBindVideo, LanguageBindVideoTokenizer, LanguageBindVideoProcessor
from eagle.model.multimodal_encoder.pointbert.point_encoder import PointTransformer
from eagle.model.multimodal_encoder.pointbert.config import PointTransformerConfig

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

        # BEGIN hxl
        # Try for load pretrained
        self.load_model()
        # END hxl

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.vision_model.encoder.gradient_checkpointing =True
        # self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class LanguageBindVideoTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        # self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        # self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        LanguageBindVideo_model = LanguageBindVideo.from_pretrained(self.vision_tower_name, device_map=device_map)
        tokenizer = LanguageBindVideoTokenizer.from_pretrained(self.vision_tower_name)
        self.image_processor = LanguageBindVideoProcessor(LanguageBindVideo_model, tokenizer)
        # self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower = LanguageBindVideo_model.vision_model
        # self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_features.reshape(-1, 8, image_features.size(-2), image_features.size(-1))
        if self.select_feature == 'patch':
            image_features = image_features[:, :, 1:, :]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        image_features = image_features.reshape(image_features.size(0), -1, image_features.size(-1))
        return image_features

    # @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, return_dict=True)
            # image_forward_outs = image_forward_outs.hidden_states
            # image_features = image_forward_outs.to(images.dtype)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

class LanguageBindAudioTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        LanguageBindAudio_model = LanguageBindAudio.from_pretrained(self.vision_tower_name, device_map=device_map)
        tokenizer = LanguageBindAudioTokenizer.from_pretrained(self.vision_tower_name)
        self.image_processor = LanguageBindAudioProcessor(LanguageBindAudio_model, tokenizer)
        self.vision_tower = LanguageBindAudio_model.vision_model
        # self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        # print(image_features.shape)
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, return_dict=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class Qwen2VLVideoTower(nn.Module):
    def __init__(self, vision_tower, args, modality='video', delay_load=False):
        super().__init__()
        self.modality = modality

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.load_model()
        print(f"Vision tower {self.vision_tower_name} is loaded.")

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        target_version = version.parse("4.52.0")
        current_version = version.parse(transformers.__version__)
        if current_version >= target_version:
            model = Qwen2VLModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        elif current_version >= version.parse("4.51.3"):
            # older transformers(4.51.3)
            model = Qwen2VLForConditionalGeneration.from_pretrained(self.vision_tower_name, device_map=device_map)
        # self.image_processor = Qwen2VLImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor = AutoProcessor.from_pretrained(
            self.vision_tower_name, 
            fixed_patch_size=(14, 14)
        )
        self.processor = AutoProcessor.from_pretrained(self.vision_tower_name)
        self.visual = model.visual
        # self.vision_tower.requires_grad_(False)
        self.is_loaded = True


    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        # raise ValueError("Stop")
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        # import pdb
        # pdb.set_trace()
        # print(f"pixel_values_videos shape: {pixel_values_videos.shape}")
        # print(f"pixel_values_videos_dim:{pixel_values_videos.dim()}")
        # print(f"video_grid_thw shape: {video_grid_thw.shape if video_grid_thw is not None else None}, needed shape: {torch.tensor([[4, 24, 42]]).shape}")
        # video_grid_thw shape: torch.Size([2, 1, 3])
        # video_grid_thw = None
        if video_grid_thw is None:
            video_grid_thw = torch.tensor([[4, 24, 42]] * pixel_values_videos.shape[0])
        if pixel_values_videos.dim() == 3:
            frame_embed_list = []
            b, n, d = pixel_values_videos.shape
            for i in range(0, b):
                frame_embed_list.append(self.visual(pixel_values_videos[i], grid_thw=video_grid_thw[i]))
            video_embeds = torch.stack(frame_embed_list, dim=0)
            
        return video_embeds


    def forward(self, images: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None):
        with torch.no_grad():
            if self.modality == 'video':
                image_features = self.get_video_features(images, video_grid_thw=video_grid_thw)
            else:
                raise ValueError(f"Unsupported modality: {self.modality}")

        return image_features


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.visual.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class FixedTokenQwen2VLImageProcessor(Qwen2VLImageProcessor):
    def __init__(self, *args, fixed_patch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_patch_size = fixed_patch_size  # 例如 (14, 14) 表示固定14x14个patches

    def preprocess(self, images, **kwargs):
        if self.fixed_patch_size:
            # 强制调整所有图像到固定尺寸
            if not isinstance(images, list):
                images = [images]

            # 计算目标尺寸以产生固定数量的patches
            target_height = self.fixed_patch_size[0] * self.patch_size
            target_width = self.fixed_patch_size[1] * self.patch_size

            processed_images = []
            for image in images:
                if isinstance(image, str):
                    image = Image.open(image)
                # 强制resize到目标尺寸
                image = image.resize((target_width, target_height))
                processed_images.append(image)

            return super().preprocess(processed_images, **kwargs)
        else:
            return super().preprocess(images, **kwargs)


class Qwen2VLTower(nn.Module):
    def __init__(self, vision_tower, args, modality='image', delay_load=False):
        super().__init__()
        self.modality = modality

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.load_model()
        print(f"Vision tower {self.vision_tower_name} is loaded.")

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        target_version = version.parse("4.52.0")
        current_version = version.parse(transformers.__version__)
        if current_version >= target_version:
            model = Qwen2VLModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        elif current_version >= version.parse("4.51.3"):
            # older transformers(4.51.3)
            model = Qwen2VLForConditionalGeneration.from_pretrained(self.vision_tower_name, device_map=device_map)
        # self.image_processor = Qwen2VLImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor = FixedTokenQwen2VLImageProcessor.from_pretrained(
            self.vision_tower_name, 
            fixed_patch_size=(14, 14)
        )
        self.processor = AutoProcessor.from_pretrained(self.vision_tower_name)
        self.visual = model.visual
        # self.vision_tower.requires_grad_(False)
        self.is_loaded = True


    def get_video_features(
        self, pixel_values_videos: torch.FloatTensor, video_grid_thw: Optional[torch.LongTensor] = None
    ):
        pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
        # print(f"pixel_values_videos shape: {pixel_values_videos.shape}") 
        # import pdb
        # pdb.set_trace()
        if video_grid_thw is None:
            print(pixel_values_videos.dim())
            if pixel_values_videos.dim() == 5:
                # [B, T, C, H, W]
                b, t, c, h, w = pixel_values_videos.shape
                patch_size = getattr(self.visual.config, "patch_size", 14)
                th = h // patch_size
                tw = w // patch_size
                video_grid_thw = torch.tensor([[t, th, tw]] * b, device=pixel_values_videos.device)
            elif pixel_values_videos.dim() == 3:
                # [B, N, D]，N为patch数
                b, n, d = pixel_values_videos.shape
                video_grid_thw = torch.tensor([[n, 1, 1]] * b, device=pixel_values_videos.device)
            elif pixel_values_videos.dim() == 2:
                # [B, N]，N为patch数
                b, n = pixel_values_videos.shape
                video_grid_thw = torch.tensor([[n, 1, 1]] * b, device=pixel_values_videos.device)
            else:
                raise ValueError(f"Unsupported pixel_values_videos shape: {pixel_values_videos.shape}")
        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
        return video_embeds

    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        # pixel_values = pixel_values.type(self.visual.dtype)
        # print(f"pixel_values shape: {pixel_values.shape}")
        if pixel_values.dim() == 3:
            image_embed_list = []
            for i in range(pixel_values.shape[0]):
                image_embed_list.append(
                    self.visual(pixel_values[i], grid_thw=image_grid_thw[i] if image_grid_thw is not None else None)
                )
            image_embeds = torch.stack(image_embed_list, dim=0)
        elif pixel_values.dim() == 2:
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        # add 1 size to the batch dimension if the input is a single image
        if image_embeds.dim() == 2:
            image_embeds = image_embeds.unsqueeze(0)
        return image_embeds

    def forward(self, images: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None):
        with torch.no_grad():
            if self.modality == 'image':
                image_features = self.get_image_features(images, image_grid_thw=image_grid_thw)
            elif self.modality == 'video':
                image_features = self.get_video_features(images, video_grid_thw=image_grid_thw)
            else:
                raise ValueError(f"Unsupported modality: {self.modality}")

        # 可选：进一步选择 patch/cls 等特征（视具体模型输出格式而定）
        # if self.select_feature == 'patch':
        #     image_features = image_features[:, :, 1:, :]  # 假设第1个token是cls
        # elif self.select_feature == 'cls_patch':
        #     pass  # 全部保留
        # else:
        #     raise ValueError(f"Unexpected select feature {self.select_feature}")

        return image_features


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.visual.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class Qwen2AudioTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = Qwen2AudioConfig.from_pretrained(self.vision_tower_name).audio_config

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        Qwen2Audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(self.vision_tower_name, device_map=device_map)
        self._image_processor = AutoProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = Qwen2Audio_model.audio_tower
        # self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        # print(image_features.shape)
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, return_dict=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    def image_processor(self, image, *args, **kwargs):
        if isinstance(image, str):
            audio, sr = librosa.load(image, sr=self._image_processor.feature_extractor.sampling_rate)
        return {"pixel_values": self._image_processor(text="", audios=audio, sampling_rate=sr, *args, **kwargs)["input_features"]}

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class PointBertTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        # self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = PointTransformerConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        config = PointTransformerConfig.from_pretrained(self.vision_tower_name)
        if getattr(config, "use_color", False):
            config.point_dims = 6
        self.vision_tower = PointTransformer(config)
        self.vision_tower.load_checkpoint(os.path.join(self.vision_tower_name, "model.pt"))

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # image_features = image_forward_outs.hidden_states[self.select_layer]
        image_features = image_forward_outs
        # print(image_features.shape)
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    def image_processor(self, image, *args, **kwargs):
        if isinstance(image, str):
            if image.endswith('.npy'):
                pc = np.load(image)
                assert pc.shape == (8192, 6), f"Expected point cloud shape (8192, 6), got {pc.shape}"
                image = torch.tensor(pc, dtype=self.dtype)
        
        image = self.pc_norm(image)
        return {"pixel_values": image}
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = torch.mean(xyz, dim=0)
        xyz = xyz - centroid
        m = torch.max(torch.sqrt(torch.sum(xyz ** 2, dim=1)))
        xyz = xyz / m

        pc = torch.cat((xyz, other_feature), dim=1)
        return pc

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.trans_dim

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2