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

#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Gemma3Config, Gemma2Model, Gemma3ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..eagle_arch import EagleMetaModel, EagleMetaForCausalLM


class EagleConfig(Gemma3Config):
    model_type = "eagle_llama"


class EagleLlamaModel(EagleMetaModel, Gemma2Model):
    config_class = EagleConfig

    def __init__(self, config: Gemma3Config):
        super(EagleLlamaModel, self).__init__(config)


class EagleLlamaForCausalLM(Gemma3ForCausalLM, EagleMetaForCausalLM):
    config_class = EagleConfig

    def __init__(self, config):
        super(Gemma3ForCausalLM, self).__init__(config)
        self.model = EagleLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # BEGIN 
        self.modal = 'image'
        # END
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    # BEGIN
    def set_modal(self, modal: str):
        assert modal in ['image', 'audio', 'video', 'point']
        self.modal = modal

    # END

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs # for llama3, upgrade the transformers and will receive an additional argument cache_position
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            # BEGIN 
            if self.modal == 'image':
            # END
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    mlp_balance_loss,
                    mlp_router_z_loss
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    self.modal,
                    image_sizes,
                )
            # BEGIN
            # elif self.modal == 'audio':
            #     (
            #         input_ids,
            #         position_ids,
            #         attention_mask,
            #         past_key_values,
            #         inputs_embeds,
            #         labels,
            #     ) = self.prepare_inputs_labels_audio(
            #         input_ids=input_ids,
            #         position_ids=position_ids,
            #         attention_mask=attention_mask,
            #         past_key_values=past_key_values,
            #         labels=labels,
            #         audios=pixel_values
            #     )
            elif self.modal == 'audio':
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    mlp_balance_loss,
                    mlp_router_z_loss
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    self.modal,
                    image_sizes,
                )
            # BEGIN qbs
            # elif self.modal == 'video':
            #     (
            #         input_ids,
            #         position_ids,
            #         attention_mask,
            #         past_key_values,
            #         inputs_embeds,
            #         labels,
            #     ) = self.prepare_inputs_labels_video(
            #         input_ids=input_ids,
            #         position_ids=position_ids,
            #         attention_mask=attention_mask,
            #         past_key_values=past_key_values,
            #         labels=labels,
            #         videos=pixel_values
            #     )
            elif self.modal == 'video':
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    mlp_balance_loss,
                    mlp_router_z_loss
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    # pixel_values,
                    self.modal,
                    image_sizes,
                )
            # END qbs

        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # Adapted from CuMo, add moe loss
        # if self.config.training:
        if self.config.mlp_smoe:
            loss = out['loss']
            if self.config.local_rank == 0:
                print('language loss: ', loss.item())
            if self.config.mlp_smoe:
                mlp_balance_loss = mlp_balance_loss.sum(dim=-1).mean()
                mlp_balance_loss = self.config.balance_loss_coef * mlp_balance_loss
                loss += mlp_balance_loss
                mlp_router_z_loss = mlp_router_z_loss.sum(dim=-1).mean()
                mlp_router_z_loss = self.config.router_z_loss_coef * mlp_router_z_loss
                loss += mlp_router_z_loss

            if self.config.local_rank == 0:
                if self.config.mlp_smoe:
                    print('mlp balance loss: ', mlp_balance_loss.item(), 'mlp router z loss: ', mlp_router_z_loss.item())

            out['loss'] = loss

        return out

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modality: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _,
                _ # Because of the adding of moe
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                modality=modality,
                image_sizes=image_sizes
            )
            # print("Image not none")
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            # print("Image none")

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs
    
    # BEGIN hxl

    @torch.no_grad()
    def audio_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        audios: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if audios is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_audio(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                audios=audios
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
    # END hxl

AutoConfig.register("eagle_llama", EagleConfig)
AutoModelForCausalLM.register(EagleConfig, EagleLlamaForCausalLM)
