from collections import OrderedDict
from typing import Sequence

import torch
from torch import nn
from mmengine.model import BaseModel
from mmdet.registry import MODELS

import open_clip


@MODELS.register_module()
class CLIPTextModel(BaseModel):
    def __init__(self,
                 name='ViT-B-16',
                 pretrained='openai',
                 pad_to_max=False,
                 use_sub_sentence_represent=False,
                 special_tokens_list=None,
                 add_pooling_layer=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(name)

        self.language_backbone = nn.Sequential(OrderedDict([
            ('body', CLIPEncoder(self.model))
        ]))
        self.language_dim = self.model.text_projection.shape[1]

        self.pad_to_max = pad_to_max
        self.use_sub_sentence_represent = use_sub_sentence_represent

        if self.use_sub_sentence_represent:
            assert special_tokens_list is not None, \
                'special_tokens_list must be provided if use_sub_sentence_represent is True'
            self.special_tokens = self.tokenizer(special_tokens_list, context_length=self.model.context_length)['input_ids']

    def forward(self, captions: Sequence[str], **kwargs) -> dict:
        device = next(self.language_backbone.parameters()).device
        tokenized = self.tokenizer(captions, context_length=self.model.context_length)
        input_ids = tokenized.to(device)

        output = self.language_backbone({'input_ids': input_ids})
        return output


class CLIPEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.context_length = model.context_length

    def forward(self, x: dict) -> dict:
        input_ids = x['input_ids']  # Shape: [B, T]
        device = input_ids.device

        x = self.token_embedding(input_ids)  # [B, T, D]
        x = x + self.positional_embedding[:x.size(1), :]
        x = x.permute(1, 0, 2)  # [T, B, D]

        x = self.transformer(x)  # [T, B, D]
        x = x.permute(1, 0, 2)  # [B, T, D]
        x = self.ln_final(x)

        eos_feature = x[:, -1, :]
        text_features = eos_feature @ self.text_projection
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return {
            'embedded': x,
            'text_features': text_features,
            'masks': torch.ones_like(input_ids, dtype=torch.bool, device=device),
            'hidden': x
        }
