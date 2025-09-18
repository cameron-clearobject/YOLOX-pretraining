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
    load_ckpt,
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
        self.best_val_loss = float("inf")
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
        self.evaluator = self.exp.get_evaluator(
            self.args.batch_size, self.is_distributed
        )

    def before_epoch(self) -> None:
        logger.info(f"---> start pre-train epoch {self.epoch + 1}")

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
        self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            self.evaluate_and_save_model()

    def after_train(self) -> None:
        logger.info(
            f"Pre-training of experiment is done. Best validation loss was {self.best_val_loss:.4f}"
        )
        self.save_best_backbone()

    def resume_train(self, model: nn.Module) -> nn.Module:
        """
        Overrides the resume logic for the pre-training task.
        Handles three scenarios:
        1. --resume: Resumes a previous pre-training run (loads model, optimizer, epoch).
        2. --ckpt yolox*.pth: Initializes the encoder from a pre-trained YOLOX model.
        3. --ckpt other.pth: Loads weights from a previous autoencoder run.
        """
        ckpt_file = self.args.ckpt

        # --- CASE 1: Resume a previous pre-training session ---
        if self.args.resume:
            logger.info("Resuming pre-training session...")
            if ckpt_file is None:
                ckpt_file = os.path.join(self.file_name, "latest_ckpt.pth")

            if os.path.isfile(ckpt_file):
                ckpt = torch.load(ckpt_file, map_location=self.device)
                model.load_state_dict(ckpt["model"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

                start_epoch = (
                    self.args.start_epoch - 1
                    if self.args.start_epoch is not None
                    else ckpt["start_epoch"]
                )
                self.start_epoch = start_epoch
                logger.info(
                    f"Loaded pre-training checkpoint '{ckpt_file}' (resuming at epoch {self.start_epoch})"
                )
            else:
                logger.warning(
                    f"Resume checkpoint not found at {ckpt_file}. Starting from scratch."
                )
                self.start_epoch = 0

        # --- CASE 2 & 3: Load weights from a checkpoint without resuming state ---
        elif ckpt_file is not None:
            # --- CASE 2: Initialize encoder from a standard YOLOX model ---
            if os.path.basename(ckpt_file).startswith("yolox"):
                logger.info(
                    f"Initializing encoder from pre-trained YOLOX model: {ckpt_file}"
                )

                yolox_ckpt = torch.load(ckpt_file, map_location=self.device)
                yolox_state_dict = yolox_ckpt.get("model", yolox_ckpt)

                # Extract weights with the prefix 'backbone.backbone.'
                prefix = "backbone.backbone."
                encoder_state_dict = {
                    k[len(prefix) :]: v
                    for k, v in yolox_state_dict.items()
                    if k.startswith(prefix)
                }

                if not encoder_state_dict:
                    logger.error(
                        "No backbone weights found in the provided YOLOX checkpoint. Starting from scratch."
                    )
                else:
                    model.encoder.load_state_dict(encoder_state_dict, strict=False)
                    logger.success(
                        "Successfully loaded backbone weights into the autoencoder's encoder. Decoder is fresh."
                    )

                self.start_epoch = 0

            # --- CASE 3: Load weights from a previous autoencoder checkpoint ---
            else:
                logger.info(
                    f"Loading autoencoder weights for transfer learning from: {ckpt_file}"
                )
                ckpt = torch.load(ckpt_file, map_location=self.device)
                model_state = ckpt.get("model", ckpt)
                model = load_ckpt(model, model_state)
                self.start_epoch = 0

        # --- Default case: No resume, no checkpoint ---
        else:
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self) -> None:
        if self.evaluator is None:
            return

        # Run evaluation to get the average validation loss
        val_loss = self.exp.eval(
            self.ema_model.ema if self.use_model_ema else self.model,
            self.evaluator,
            self.is_distributed,
            half=self.amp_training,
        )

        # Check if the current model is the best
        update_best_ckpt = val_loss < self.best_val_loss
        self.best_val_loss = min(self.best_val_loss, val_loss)

        if self.rank == 0:
            logger.info(
                f"Val Loss: {val_loss:.4f}, Best Val Loss: {self.best_val_loss:.4f}"
            )
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("pretrain/val_loss", val_loss, self.epoch + 1)

        self.save_ckpt("last_epoch", update_best_ckpt, val_loss=val_loss)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", val_loss=val_loss)

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
                    logger.info(
                        f"Loading best checkpoint from '{best_ckpt_path}' to extract backbone."
                    )
                    ckpt = torch.load(best_ckpt_path, map_location="cpu")
                    source_model_state_dict = ckpt.get("model", {})
                    source_name = "best checkpoint"
                except Exception as e:
                    logger.warning(
                        f"Failed to load best checkpoint: {e}. Falling back to final model."
                    )

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
                k[len(prefix) :]: v
                for k, v in source_model_state_dict.items()
                if k.startswith(prefix)
            }

            if not backbone_state_dict:
                logger.warning(
                    f"No backbone weights found in the {source_name}. Cannot save backbone."
                )
                return

            save_path = os.path.join(self.file_name, "best_backbone.pth")
            torch.save(backbone_state_dict, save_path)
            logger.success(
                f"Successfully extracted backbone from {source_name} and saved to '{save_path}'"
            )

    def save_ckpt(
        self,
        ckpt_name: str,
        update_best_ckpt: bool = False,
        val_loss: Optional[float] = None,
    ) -> None:
        # Override to save validation loss instead of AP
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,  # This will save "best_ckpt.pth" if True
                self.file_name,
                ckpt_name,
            )
