# # Copyright (c) OpenMMLab. All rights reserved.
# from collections import OrderedDict
# from typing import Sequence
# from types import SimpleNamespace

# import torch
# from mmengine.model import BaseModel
# from torch import nn

# try:
#     import open_clip
# except ImportError:
#     open_clip = None

# from mmdet.registry import MODELS


# @MODELS.register_module()
# class CLIPTextModel(BaseModel):
#     """CLIP model for language embedding using OpenCLIP.

#     Args:
#         name (str): CLIP model name, e.g. 'ViT-B-32' or 'ViT-B-32-quickgelu'.
#         pretrained (str): Pretrained weights name, e.g. 'openai'.
#         max_tokens (int): Max token length (CLIP usually uses 77).
#         use_sub_sentence_represent (bool): Enable sub-sentence representation.
#         special_tokens_list (list): List of special tokens for sentence splits.
#         num_layers_of_embedded (int): How many hidden layers to average.
#     """

#     def __init__(self,
#                  name: str = 'ViT-B-32-quickgelu',  # Use -quickgelu to avoid warning
#                  pretrained: str = 'openai',
#                  max_tokens: int = 77,
#                  use_sub_sentence_represent: bool = False,
#                  special_tokens_list: list = None,
#                  num_layers_of_embedded: int = 1,
#                  **kwargs):
#         super().__init__(**kwargs)

#         if open_clip is None:
#             raise ImportError(
#                 'open_clip not found. Install with: pip install open_clip_torch')

#         self.max_tokens = max_tokens
#         self.use_sub_sentence_represent = use_sub_sentence_represent

#         # Load model and tokenizer
#         self.model, _, _ = open_clip.create_model_and_transforms(
#             model_name=name, pretrained=pretrained)
#         self.tokenizer = open_clip.get_tokenizer(name)

#         # Freeze model parameters by default
#         for p in self.model.parameters():
#             p.requires_grad = False

#         self.language_dim = self.model.text_projection.shape[1]
#         self.num_layers_of_embedded = num_layers_of_embedded

#         # Avoid recursion error: use dummy namespaces for attributes accessed externally
#         self.language_backbone = SimpleNamespace(
#             body=SimpleNamespace(language_dim=self.language_dim)
#         )

#         if self.use_sub_sentence_represent:
#             assert special_tokens_list is not None, \
#                 'special_tokens_list must be set if use_sub_sentence_represent is True'
#             self.special_tokens = self.tokenizer(
#                 special_tokens_list, context_length=self.model.context_length)

#     def forward(self, captions: Sequence[str], **kwargs) -> dict:
#         """Forward pass to compute CLIP text embeddings."""
#         device = next(self.model.parameters()).device
#         tokenized = self.tokenizer(
#             captions, context_length=self.model.context_length).to(device)

#         outputs = self.model.encode_text(tokenized)  # [B, D]

#         results = {
#             'embedded': outputs,                # [B, D]
#             'masks': torch.ones_like(outputs),  # Dummy mask for compatibility
#             'hidden': outputs.unsqueeze(1)      # [B, 1, D] for consistency
#         }

#         if self.use_sub_sentence_represent:
#             results['text_token_mask'] = torch.ones_like(tokenized, dtype=torch.bool)

#         return results
 ######################################################
from collections import OrderedDict
from typing import Sequence
from types import SimpleNamespace

import torch
from mmengine.model import BaseModel
from torch import nn

try:
    import open_clip
except ImportError:
    open_clip = None

from mmdet.registry import MODELS


@MODELS.register_module()
class CLIPTextModel(BaseModel):
    """CLIP model for language embedding using OpenCLIP.

    Args:
        name (str): CLIP model name, e.g. 'ViT-B-32' or 'ViT-B-32-quickgelu'.
        pretrained (str): Pretrained weights name, e.g. 'openai'.
        max_tokens (int): Max token length (CLIP usually uses 77).
        use_sub_sentence_represent (bool): Enable sub-sentence representation.
        special_tokens_list (list): List of special tokens for sentence splits.
        num_layers_of_embedded (int): How many hidden layers to average.
        pad_to_max (bool): Whether to pad tokenized input to max length.
        add_pooling_layer (bool): Placeholder for API compatibility.
    """

    def __init__(self,
                 name: str = 'ViT-B-32-quickgelu',
                 pretrained: str = 'openai',
                 max_tokens: int = 77,
                 use_sub_sentence_represent: bool = True,
                 special_tokens_list: list = ['[CLS]', '[SEP]', '.', '?'],
                 num_layers_of_embedded: int = 1,
                 pad_to_max: bool = False,
                 add_pooling_layer: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        if open_clip is None:
            raise ImportError(
                'open_clip not found. Install with: pip install open_clip_torch')

        self.max_tokens = max_tokens
        self.use_sub_sentence_represent = use_sub_sentence_represent
        self.pad_to_max = pad_to_max
        self.add_pooling_layer = add_pooling_layer  # for future use / compatibility

        # Load model and tokenizer
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name=name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(name)

        # Freeze model parameters by default
        self.set_requires_grad(False)

        self.language_dim = self.model.text_projection.shape[1]
        self.num_layers_of_embedded = num_layers_of_embedded

        self.language_backbone = SimpleNamespace(
            body=SimpleNamespace(language_dim=self.language_dim)
        )

        if self.use_sub_sentence_represent:
            assert special_tokens_list is not None, \
                'special_tokens_list must be set if use_sub_sentence_represent is True'
            self.special_tokens = self.tokenizer(
                special_tokens_list, context_length=self.model.context_length)

    def forward(self, captions: Sequence[str], **kwargs) -> dict:
        """Forward pass to compute CLIP text embeddings."""
        device = next(self.model.parameters()).device
        tokenized = self.tokenizer(
            captions, context_length=self.model.context_length).to(device)

        outputs = self.model.encode_text(tokenized)  # [B, D]

        results = {
            'embedded': outputs,                # [B, D]
            'masks': torch.ones_like(outputs),  # Dummy mask for compatibility
            'hidden': outputs.unsqueeze(1)      # [B, 1, D]
        }

        if self.use_sub_sentence_represent:
            results['text_token_mask'] = torch.ones_like(tokenized, dtype=torch.bool)

        return results

    def set_requires_grad(self, requires_grad: bool = True,
                          freeze_projection: bool = False):
        """Set requires_grad for model parameters.

        Args:
            requires_grad (bool): Whether to enable gradients.
            freeze_projection (bool): Whether to keep the final projection layer frozen.
        """
        for name, param in self.model.named_parameters():
            if freeze_projection and 'text_projection' in name:
                param.requires_grad = False
            else:
                param.requires_grad = requires_grad
