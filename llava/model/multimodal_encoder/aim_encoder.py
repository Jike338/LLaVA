from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from huggingface_hub import PyTorchModelHubMixin

import torch
from torch import nn

from .aim_utils import mixins, layers

import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

class AIMVisionTower(nn.Module):
    def __init__(self, delay_load=False, use_clip_processor=True):
        super().__init__()

        self.is_loaded = False
        if use_clip_processor:
            self.vision_tower_name = 'openai/clip-vit-large-patch14-336'
        if not delay_load:
            self.load_model()

    def load_model(self):
        self.aim_bkb = torch.hub.load("apple/ml-aim", "aim_1B")
        self.aim_bkb.requires_grad_(False)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = nn.Sequential(
            self.aim_bkb.preprocessor,
            self.aim_bkb.trunk
        )
        # self.vision_tower = self.vision_tower.to(torch.bfloat16)
        self._dtype = next(self.vision_tower.parameters()).dtype
        self._hidden_size = 2048
        self.is_loaded = True
    
    @torch.no_grad()
    def forward(self, images):
        # image size torch.Size([32, 3, 336, 336]) resize to 224x224
        self.vision_tower = self.vision_tower.to(self.dtype)
        expected_size = (224, 224)  # Or the expected size your AIM model was trained on
        images = F.interpolate(images, size=expected_size, mode='bilinear', align_corners=False)
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Please call load_model() first.")
        image_features, _agg_feat = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
        return image_features.to(torch.bfloat16)
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def hidden_size(self):
        return self._hidden_size