# yolox/core/finetune_trainer.py

from loguru import logger
import argparse
import torch.nn as nn

from .trainer import Trainer
from yolox.exp import Exp

class FinetuneTrainer(Trainer):
    def __init__(self, exp: Exp, args: argparse.Namespace) -> None:
        super().__init__(exp, args)

    def before_train(self) -> None:
        self.model = self.exp.get_model()
        self.freeze_backbone(self.model)
        super().before_train()

    def before_epoch(self) -> None:
        super().before_epoch()
        if self.epoch + 1 == self.exp.freeze_epochs:
            logger.info("--- Unfreezing backbone and resetting optimizer ---")
            self.unfreeze_backbone(self.model)
            self.optimizer = self.exp.get_optimizer(self.args.batch_size)
            logger.info("Optimizer reset with all layers now trainable.")

    def freeze_backbone(self, model: nn.Module) -> None:
        logger.info(f"Freezing backbone for the first {self.exp.freeze_epochs} epochs.")
        for name, param in model.backbone.backbone.named_parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, model: nn.Module) -> None:
        model_to_unfreeze = model.module if self.is_distributed else model
        for name, param in model_to_unfreeze.backbone.backbone.named_parameters():
            param.requires_grad = True