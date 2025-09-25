
#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import os
import torch.nn as nn

from yolox.exp import Exp as MyExp
from yolox.data import (
    COCOConditionalDataset,
    ConditionalTrainTransform,
    ConditionalValTransform,
)


class ModelWrapper(nn.Module):
    """
    A wrapper to adapt a standard YOLOX model for 6-channel input.
    It adds a single convolutional layer to project 6 channels down to the
    3 channels expected by the original model backbone.
    """
    def __init__(self, original_model, in_channels=6):
        super().__init__()
        # ASSUMPTION: A 1x1 convolution is sufficient to learn a mapping from
        # the 6-channel space to the 3-channel space the backbone expects.
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.original_model = original_model

        # alias original_model.backbone and original_model.head
        self.backbone = original_model.backbone
        self.head = original_model.head

    def forward(self, x, targets=None):
        x = self.input_layer(x)
        # Pass through to the original model. The YOLOX model handles training
        # and inference logic internally based on the 'targets' argument.
        if self.training:
            return self.original_model(x, targets)
        else:
            return self.original_model(x, None)


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 80
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.input_channels = 6

        # --- Add path to your 'before' images dataset ---
        self.before_data_dir = None  # IMPORTANT: Set this path

    def get_model(self):
        # Get the standard 3-channel model from the base experiment
        super().get_model()
        
        # Wrap the standard model to handle 6-channel input
        original_model = self.model
        self.model = ModelWrapper(original_model, in_channels=6)

        return self.model

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        assert self.before_data_dir is not None, "You must set `self.before_data_dir` in the experiment file."
        return COCOConditionalDataset(
            data_dir=self.data_dir,
            before_data_dir=self.before_data_dir,
            json_file=self.train_ann,
            name="train2017",
            img_size=self.input_size,
            preproc=ConditionalTrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img: str = None):
        # This overrides the base get_data_loader to ensure the MosaicDetection
        # wrapper uses our new ConditionalTrainTransform.
        
        from yolox.data import (
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            YoloBatchSampler,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        if self.dataset is None:
            with wait_for_the_master():
                self.dataset = self.get_dataset(cache=cache_img is not None, cache_type=cache_img)

        self.dataset = MosaicDetection(
            dataset=self.dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=ConditionalTrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )
        
        # The rest of this function is identical to the base implementation
        if is_distributed:
            batch_size = batch_size // self.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader
    
    def get_eval_dataset(self, **kwargs):
        assert self.before_data_dir is not None, "You must set `self.before_data_dir` in the experiment file."
        legacy = kwargs.get("legacy", False)
        testdev = kwargs.get("testdev", False)
        
        return COCOConditionalDataset(
            data_dir=self.data_dir,
            before_data_dir=self.before_data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="val2017" if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ConditionalValTransform(legacy=legacy),
        )

    def get_trainer(self, args):
        from yolox.core import StackedTrainer
        trainer = StackedTrainer(self, args)
        return trainer
