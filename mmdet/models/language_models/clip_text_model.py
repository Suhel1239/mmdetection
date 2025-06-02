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
                 **kwargs):
        """
        Args:
            model_name (str): CLIP model name (e.g. 'ViT-B-16')
            pretrained (str or None): pretrained weights (e.g. 'openai')
        """
        super().__init__(**kwargs)

        # Load CLIP model + tokenizer
        self.model, _, _ = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)

        # Get tokenizer for the model
        self.tokenizer = open_clip.get_tokenizer(name)

        # Use only text encoder from CLIP
        self.text_encoder = self.model.transformer
        self.token_embedding = self.model.token_embedding
        self.positional_embedding = self.model.positional_embedding
        self.ln_final = self.model.ln_final
        self.text_projection = self.model.text_projection

        self.context_length = self.model.context_length

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
        x = x.permute(1, 0, 2)  # NLD -> LND for transformer: [T, B, D]

        x = self.text_encoder(x)  # forward through transformer
        x = x.permute(1, 0, 2)  # back to [B, T, D]

        x = self.ln_final(x)  # layer norm

        # Take features from [EOS] token (last token)
        eos_feature = x[:, -1, :]

        # Project to text feature space
        text_features = eos_feature @ self.text_projection

        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return {'text_features': text_features}
