
# yolox/core/pretrain_trainer.py

import os
from loguru import logger

import torch
import torch.nn as nn
from typing import Dict, Optional, Any

# Import the standard Trainer to inherit from it
from .trainer import Trainer
from yolox.utils import (
    is_parallel,
    save_checkpoint,
    WandbLogger,
)


class PretrainTrainer(Trainer):
    """
    A specialized trainer for the unsupervised autoencoder pre-training task.
    It inherits from the standard Trainer and overrides methods for:
    - Loss calculation (MSE reconstruction loss).
    - The core training iteration logic.
    - Checkpointing (to remove evaluation metrics).
    - Final model saving (to save only the backbone).
    """

    def __init__(self, exp: "Exp", args: "argparse.Namespace") -> None:
        super().__init__(exp, args)
        # Pre-training does not have an AP metric.
        if hasattr(self, "best_ap"):
            del self.best_ap
        # Set the loss function for autoencoder reconstruction
        self.criterion = nn.MSELoss()

    def before_train(self) -> None:
        # This method is largely copied from the parent Trainer, but with the
        # evaluation-related parts removed to avoid errors.
        logger.info(f"args: {self.args}")
        logger.info(f"exp value:\n{self.exp}")

        # Model and optimizer setup is the same
        super().before_train()

        # Overwrite the evaluator setup from the parent class
        self.evaluator = None

        # Modify loggers setup to not use evaluator
        if self.rank == 0 and self.args.logger == "wandb":
            self.wandb_logger = WandbLogger.initialize_wandb_logger(
                self.args, self.exp, self.train_loader.dataset
            )

    def train_one_iter(self) -> None:
        # This is the core logic for one iteration of autoencoder training.
        iter_start_time = torch.cuda.Event(enable_timing=True)
        iter_end_time = torch.cuda.Event(enable_timing=True)
        iter_start_time.record()

        # Unpack data, ignoring targets
        inps, _ = self.prefetcher.next()
        inps = inps.to(self.data_type)
        inps, _ = self.exp.preprocess(inps, None, self.input_size)

        # Unsupervised forward and loss calculation
        with torch.cuda.amp.autocast(enabled=self.amp_training):
            reconstructed = self.model(inps)
            loss = self.criterion(reconstructed, inps)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time.record()
        torch.cuda.synchronize()
        self.meter.update(
            iter_time=iter_end_time.elapsed_time(iter_start_time) / 1000.0,
            lr=lr,
            total_loss=loss,  # Log the reconstruction loss
        )

    def after_epoch(self) -> None:
        # Override to prevent evaluation and implement pre-training checkpointing.
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            if self.save_history_ckpt:
                self.save_ckpt(ckpt_name=f"epoch_{self.epoch + 1}")

    def save_ckpt(
        self, ckpt_name: str, update_best_ckpt: bool = False, ap: Optional[float] = None
    ) -> None:
        # Override to remove evaluation metrics (AP) from the checkpoint state.
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                False,  # update_best_ckpt is always False for pre-training
                self.file_name,
                ckpt_name,
            )

    def after_train(self) -> None:
        logger.info("Pre-training of experiment is done.")
        self.save_backbone()

    def save_best_backbone(self) -> None:
        """
        Saves the backbone from the best-performing model.
        If 'best_ckpt.pth' is not found, it falls back to saving the backbone
        from the final epoch's model (EMA if enabled, otherwise the standard model).
        """
        if self.rank == 0:
            best_ckpt_path = os.path.join(self.file_name, "best_ckpt.pth")
            
            source_model_state_dict = None
            source_name = ""

            # **Priority 1: Try to load the best checkpoint**
            if os.path.exists(best_ckpt_path):
                try:
                    logger.info(f"Loading best checkpoint from '{best_ckpt_path}' to extract backbone.")
                    ckpt = torch.load(best_ckpt_path, map_location="cpu")
                    source_model_state_dict = ckpt.get("model", {})
                    source_name = "best checkpoint"
                except Exception as e:
                    logger.warning(f"Failed to load best checkpoint: {e}. Falling back to final model.")
            
            # **Priority 2: Fallback to the final epoch's model**
            if source_model_state_dict is None:
                logger.warning("Could not find 'best_ckpt.pth' or it failed to load.")
                # Use EMA model if available, as it's generally more stable
                if self.use_model_ema:
                    logger.info("Falling back to the final epoch's EMA model.")
                    source_model = self.ema_model.ema
                    source_name = "final EMA model"
                else:
                    logger.info("Falling back to the final epoch's standard model.")
                    source_model = self.model
                    source_name = "final standard model"
                
                # Handle DDP wrapper if training was distributed
                if is_parallel(source_model):
                    source_model = source_model.module
                source_model_state_dict = source_model.state_dict()

            # **Extract and save the backbone weights from the chosen source**
            prefix = "backbone.backbone."
            backbone_state_dict = {
                k[len(prefix):]: v
                for k, v in source_model_state_dict.items()
                if k.startswith(prefix)
            }

            if not backbone_state_dict:
                logger.warning(f"No backbone weights found in the {source_name}. Cannot save backbone.")
                return

            save_path = os.path.join(self.file_name, "best_backbone.pth")
            torch.save(backbone_state_dict, save_path)
            logger.success(f"Successfully extracted backbone from {source_name} and saved to '{save_path}'")






