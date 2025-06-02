from mmengine.model import BaseModel
from torch import nn
import torch
import open_clip

from mmdet.registry import MODELS


@MODELS.register_module()
class CLIPTextModel(BaseModel):
    def __init__(self,
                 name='ViT-B-16',
                 pretrained='openai',
                 pad_to_max=False,
                 use_sub_sentence_represent=True,
                 special_tokens_list=None,
                 add_pooling_layer=False,
                 **kwargs):
        """
        Args:
            name (str): CLIP model name (e.g. 'ViT-B-16')
            pretrained (str or None): pretrained weights (e.g. 'openai')
            pad_to_max (bool): Whether to pad sequences to max length.
            use_sub_sentence_represent (bool): Whether to use sub-sentence representation.
            special_tokens_list (list[str]): List of tokens to consider as special tokens.
            add_pooling_layer (bool): Whether to add a pooling layer at the end.
        """
        super().__init__(**kwargs)

        self.name = name
        self.pretrained = pretrained
        self.pad_to_max = pad_to_max
        self.use_sub_sentence_represent = use_sub_sentence_represent
        self.special_tokens_list = special_tokens_list or ['[CLS]', '[SEP]', '.', '?']
        self.add_pooling_layer = add_pooling_layer

        # Load CLIP model + tokenizer
        self.model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)

        # Tokenizer and core CLIP components
        self.tokenizer = open_clip.get_tokenizer(name)
        self.text_encoder = self.model.transformer
        self.token_embedding = self.model.token_embedding
        self.positional_embedding = self.model.positional_embedding
        self.ln_final = self.model.ln_final
        self.text_projection = self.model.text_projection
        self.context_length = self.model.context_length

        if self.add_pooling_layer:
            self.pooling_layer = nn.AdaptiveAvgPool1d(1)

    def forward(self, texts):
        """
        Args:
            texts (List[str]): list of strings

        Returns:
            dict: {'text_features': Tensor[B, D]} normalized embeddings
        """
        device = next(self.parameters()).device

        # Tokenize input texts
        tokenized = self.tokenizer(texts, context_length=self.context_length).to(device)

        x = self.token_embedding(tokenized)  # [B, T, D]
        x = x + self.positional_embedding  # add positional embedding
        x = x.permute(1, 0, 2)  # [T, B, D] for transformer

        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)  # [B, T, D]

        x = self.ln_final(x)

        if self.use_sub_sentence_represent:
            if self.add_pooling_layer:
                x_pooled = self.pooling_layer(x.transpose(1, 2)).squeeze(-1)
            else:
                # Mean pooling
                x_pooled = x.mean(dim=1)
            text_features = x_pooled @ self.text_projection
        else:
            # Default: use the last token
            eos_feature = x[:, -1, :]
            text_features = eos_feature @ self.text_projection

        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return {'text_features': text_features}
