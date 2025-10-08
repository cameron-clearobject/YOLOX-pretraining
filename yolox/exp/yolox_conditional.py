
#!/usr/bin/env python3
# Copyright (c) Megvii Inc. All rights reserved.

import os
import torch
import torch.nn as nn

from yolox.exp import Exp as MyExp
from yolox.data import (
    COCOConditionalDataset,
    ConditionalTrainTransform,
    ConditionalValTransform,
)


# class ModelWrapper(nn.Module):
#     """
#     A wrapper to adapt a standard YOLOX model for 6-channel input.
#     It adds a single convolutional layer to project 6 channels down to the
#     3 channels expected by the original model backbone.
#     """
#     def __init__(self, original_model, in_channels=6):
#         super().__init__()
#         # ASSUMPTION: A 1x1 convolution is sufficient to learn a mapping from
#         # the 6-channel space to the 3-channel space the backbone expects.
#         self.input_layer = nn.Sequential(
#             nn.Conv2d(in_channels, 3, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(3),
#             nn.LeakyReLU(0.1, inplace=True),
#         )
#         self.original_model = original_model

#         # alias original_model.backbone and original_model.head
#         self.backbone = original_model.backbone
#         self.head = original_model.head

#     def forward(self, x, targets=None):
#         x = self.input_layer(x)
#         # Pass through to the original model. The YOLOX model handles training
#         # and inference logic internally based on the 'targets' argument.
#         if self.training:
#             return self.original_model(x, targets)
#         else:
#             return self.original_model(x, None)


class ChannelAttentionModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        
        pooled_x = self.avg_pool(x)  # (B, C, 1, 1)
        weights = self.fc(pooled_x)   # (B, C, 1, 1)
        
        return weights


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)

        avg_out = torch.mean(x, dim=1, keepdim=True) # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B, 1, H, W)      

        concatenated = torch.cat([avg_out, max_out], dim=1) # (B, 2, H, W)
        
        mask = self.sigmoid(self.conv(concatenated)) # (B, 1, H, W)
        
        return mask


class AttentionFusion(nn.Module):
    """
    A module to fuse 'after' features using 'before' features as context.
    """
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttentionModule(channels, reduction)
        self.spatial_attn = SpatialAttentionModule(kernel_size)

    def forward(self, f_after, f_before):
        # f_after: (B, C, H, W)
        # f_before: (B, C, H, W)

        channel_weights = self.channel_attn(f_before)      # (B, C, 1, 1)
        spatial_mask = self.spatial_attn(f_before)        # (B, 1, H, W)

        f_after_refined = f_after * channel_weights       # (B, C, H, W)
        f_after_refined = f_after_refined * spatial_mask  # (B, C, H, W)
        
        return f_after_refined


class ModelWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        self.backbone = self.original_model.backbone
        self.head = self.original_model.head

        # --- Create fusion modules for each FPN level ---
        self.fusion_p3 = AttentionFusion(channels=256)
        self.fusion_p4 = AttentionFusion(channels=512)
        self.fusion_p5 = AttentionFusion(channels=1024)

    def forward(self, x, targets=None):
        c = x.shape[1] // 2
        x_before = x[:, :c, :, :]
        x_after = x[:, c:, :, :]

        fpn_outs_before = self.backbone(x_before)
        fpn_outs_after = self.backbone(x_after)

        # Unpack features for clarity
        p3_before, p4_before, p5_before = fpn_outs_before
        p3_after, p4_after, p5_after = fpn_outs_after

        # Apply attention-based fusion with a residual connection
        fused_p3 = p3_after + self.fusion_p3(p3_after, p3_before)
        fused_p4 = p4_after + self.fusion_p4(p4_after, p4_before)
        fused_p5 = p5_after + self.fusion_p5(p5_after, p5_before)

        fpn_outs_fused = (fused_p3, fused_p4, fused_p5)

        if self.training:
            return self.head(fpn_outs_fused, targets, x_after)
        else:
            return self.head(fpn_outs_fused)

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
