
# yolox/exp/yolox_finetune_exp.py

import torch
import torch.nn as nn
from loguru import logger
import argparse

from .yolox_base import Exp

class FinetuneExp(Exp):
    def __init__(self) -> None:
        super().__init__()
        self.pretrain_weights: str = "./YOLOX_outputs/pretrain_yolox_L_custom/pretrained_yolox_L_backbone.pth"
        self.basic_lr_per_img: float = 0.001 / 64.0
        self.max_epoch: int = 80
        self.depth: float = 1.00
        self.width: float = 1.00
        self.exp_name: str = "yolox_L_finetune_from_pretrain"
        self.freeze_epochs: int = 5

    def get_model(self) -> nn.Module:
        model = super().get_model()
        if self.pretrain_weights:
            logger.info(f"Loading pre-trained backbone weights from {self.pretrain_weights}")
            try:
                ckpt = torch.load(self.pretrain_weights, map_location="cpu")
                model.backbone.backbone.load_state_dict(ckpt, strict=False)
                logger.success("Pre-trained backbone weights loaded successfully!")
            except Exception as e:
                logger.warning(f"Could not load pre-trained weights: {e}")
        return model

    def get_trainer(self, args: argparse.Namespace): # -> FinetuneTrainer
        from yolox.core.finetune_trainer import FinetuneTrainer
        trainer = FinetuneTrainer(self, args)
        return trainer