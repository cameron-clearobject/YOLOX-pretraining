# yolox/exp/pretrain_exp.py

import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader as torchDataLoader,
    BatchSampler as torchBatchSampler,
    Dataset,
    SequentialSampler,
)
from torchvision.datasets import ImageFolder
import random
from loguru import logger
from typing import List, Tuple, Optional, Any
import argparse

from .yolox_base import Exp
from yolox.models import Autoencoder
from yolox.data import (
    PretrainTransform,
    InfiniteSampler,
    DataLoader,
    ValTransform,
    PretrainValTransform,
)
from yolox.utils import wait_for_the_master
from yolox.evaluators import ReconstructionEvaluator


def pretrain_collate_fn(batch):
    images, labels = default_collate(batch)
    return images, labels, None, None


class PretrainExp(Exp):

    def __init__(self):
        super().__init__()
        # --- Model Config ---
        self.depth = 0.33
        self.width = 0.50
        self.act = "silu"

        # --- Dataloader Config ---
        self.input_size = (256, 256)
        self.test_size = self.input_size
        self.multiscale_range = 5
        self.data_dir = "data/unlabeled"
        self.val_data_dir = "data/unlabeled_val"
        self.data_num_workers = 4

        # --- Transform Config ---
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.5, 1.5)
        self.shear = 2.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        # --- Training Config ---
        self.max_epoch = 100
        self.print_interval = 10
        self.eval_interval = 10
        self.ema = True
        # This attribute was missing, causing the error
        self.save_history_ckpt = True

        self.exp_name = "backbone_pretrain"
        self.output_weights_path = "pretrained_backbone.pth"

    def get_model(self):
        self.model = Autoencoder(dep_mul=self.depth, wid_mul=self.width, act=self.act)
        return self.model

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img: str = None
    ):
        if cache_img is not None and cache_img != "ram":
            logger.warning(
                "ImageFolder dataset does not support disk caching. Caching is disabled."
            )
            cache_img = None

        # if cache is True, we will create self.dataset before launch
        if self.dataset is None:
            self.dataset = self.get_dataset(
                cache=cache_img is not None, cache_type=cache_img
            )

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = torchBatchSampler(
            sampler=sampler, batch_size=batch_size, drop_last=False
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["collate_fn"] = pretrain_collate_fn

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        return train_loader

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Updated to accept cache arguments for compatibility with the training script.
        """
        if cache:
            logger.info(
                "Caching for ImageFolder is not implemented. Images will be loaded from disk directly."
            )

        transform = PretrainTransform(
            input_size=self.input_size,
            degrees=self.degrees,
            translate=self.translate,
            scales=self.mosaic_scale,
            shear=self.shear,
            flip_prob=self.flip_prob,
            hsv_prob=self.hsv_prob,
        )
        return ImageFolder(self.data_dir, transform=transform)

    def get_trainer(self, args):
        from yolox.core.pretrain_trainer import PretrainTrainer

        trainer = PretrainTrainer(self, args)
        return trainer

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, "random_size"):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            torch.distributed.barrier()
            torch.distributed.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
        return inputs, targets

    def get_eval_dataset(self, **kwargs) -> Optional[Dataset]:
        if not self.val_data_dir:
            return None
        transform = PretrainValTransform(input_size=self.test_size)
        return ImageFolder(self.val_data_dir, transform=transform)

    def get_eval_loader(
        self, batch_size: int, is_distributed: bool, **kwargs
    ) -> Optional[DataLoader]:
        val_dataset = self.get_eval_dataset(**kwargs)
        if val_dataset is None:
            return None

        sampler = SequentialSampler(val_dataset)
        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
            "batch_size": batch_size,
            "collate_fn": pretrain_collate_fn,
        }
        return torchDataLoader(val_dataset, **dataloader_kwargs)

    def get_evaluator(
        self,
        batch_size: int,
        is_distributed: bool,
        testdev: bool = False,
        legacy: bool = False,
    ):
        val_loader = self.get_eval_loader(batch_size, is_distributed)
        if val_loader is None:
            return None
        return ReconstructionEvaluator(dataloader=val_loader)

    def eval(
        self,
        model: nn.Module,
        evaluator: ReconstructionEvaluator,
        is_distributed: bool,
        half: bool = False,
        return_outputs: bool = False,
    ) -> float:
        """
        Runs the evaluation loop for the autoencoder, calculating reconstruction loss.
        """
        if evaluator is None:
            return float("inf")

        val_loader = evaluator.dataloader
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for inps, _ in val_loader:
                inps = inps.cuda()
                if half:
                    inps = inps.half()

                reconstructed = model(inps)
                loss = criterion(reconstructed, inps)
                total_loss += loss.item()
                num_batches += 1

        model.train()  # Set model back to training mode

        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        return avg_loss
