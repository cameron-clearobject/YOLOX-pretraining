
# yolox/exp/pretrain_exp.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as torchDataLoader, BatchSampler as torchBatchSampler
from torch.utils.data.dataloader import default_collate
from torchvision.datasets import ImageFolder
import random

# YOLOX imports
from .yolox_base import Exp
from yolox.models import Autoencoder
from yolox.data import PretrainTransform, InfiniteSampler, DataLoader

# Custom collate function to make ImageFolder compatible with DataPrefetcher
def pretrain_collate_fn(batch):
    """
    Collates a batch from ImageFolder and pads the output to four items.
    This prevents an unpacking error in the DataPrefetcher, which expects
    (input, target, img_info, img_id).
    """
    # The default collate function stacks images and labels into tensors
    images, labels = default_collate(batch)
    # Return a 4-tuple to match the prefetcher's expected output format
    return images, labels, None, None


class PretrainExp(Exp):
    def __init__(self):
        super().__init__()
        # --- Model Config ---
        self.depth = 0.33
        self.width = 0.50
        
        # --- Dataloader Config ---
        self.input_size = (256, 256)
        self.multiscale_range = 5
        self.data_dir = "data/unlabeled"

        # --- Transform Config ---
        self.degrees = 10.0
        self.translate = 0.1
        self.mosaic_scale = (0.5, 1.5)
        self.shear = 2.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        
        # --- Training Config ---
        self.max_epoch = 100
        self.exp_name = "backbone_pretrain"
        self.output_weights_path = "pretrained_backbone.pth"
        self.ema = True
        
    def get_model(self):
        self.model = Autoencoder(dep_mul=self.depth, wid_mul=self.width, act=self.act)
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        """
        Updated to be fully compatible with the YOLOX data loading pipeline.
        """
        dataset = self.get_dataset()

        # Use InfiniteSampler for endless stream of training data, as in standard YOLOX training
        sampler = InfiniteSampler(len(dataset), seed=self.seed if self.seed else 0)

        # Use a standard torch BatchSampler since mosaic is not used for pre-training
        batch_sampler = torchBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False
        )
        
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        # Use our custom collate function to format the batch correctly for the prefetcher
        dataloader_kwargs["collate_fn"] = pretrain_collate_fn

        # Use the custom YOLOX DataLoader
        train_loader = DataLoader(dataset, **dataloader_kwargs)
        
        return train_loader

    def get_dataset(self):
        """
        This helper function is called by get_data_loader.
        """
        transform = PretrainTransform(
            input_size=self.input_size, degrees=self.degrees, translate=self.translate,
            scales=self.mosaic_scale, shear=self.shear, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
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
            if not hasattr(self, 'random_size'):
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
