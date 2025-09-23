
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import random
import cv2
import numpy as np

from .coco import COCODataset
from ..data_augment import TrainTransform, ValTransform, _mirror, augment_hsv
from yolox.utils import xyxy2cxcywh


def preproc_conditional(img, input_size, swap=(2, 0, 1)):
    """
    A modified version of preproc that handles 6-channel images.
    """
    if len(img.shape) == 3:
        # Padded image should have the same number of channels as input image
        num_channels = img.shape[2]
        padded_img = np.ones((input_size[0], input_size[1], num_channels), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)

    if len(resized_img.shape) == 2:
        resized_img = np.expand_dims(resized_img, axis=-1)

    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class ConditionalTrainTransform(TrainTransform):
    """
    TrainTransform for 6-channel conditional images.
    ASSUMPTION: HSV and other color-space augmentations are applied only to the 'after' image.
    Geometric augmentations like flip are applied to both 'before' and 'after' images.
    """
    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc_conditional(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        boxes_o = xyxy2cxcywh(boxes_o)

        # Split into before and after images for augmentation
        before_img = image[:, :, :3].copy()
        after_img = image[:, :, 3:].copy()

        if random.random() < self.hsv_prob:
            augment_hsv(after_img)

        after_img_t, boxes = _mirror(after_img, boxes, self.flip_prob)
        
        # Apply the same mirror transform to the 'before' image if it happened
        if after_img.shape[1] != after_img_t.shape[1]:
            before_img_t = before_img[:, ::-1].copy()
        else:
            before_img_t = before_img

        image_t = np.concatenate((before_img_t, after_img_t), axis=2)

        image_t, r_ = preproc_conditional(image_t, input_dim)
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc_conditional(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[: self.max_labels]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ConditionalValTransform(ValTransform):
    """
    ValTransform for 6-channel conditional images.
    """
    def __call__(self, img, res, input_size):
        img, _ = preproc_conditional(img, input_size, self.swap)
        if self.legacy:
            # Legacy normalization is not handled here as it assumes 3 channels.
            # The base experiment doesn't use it by default.
            pass
        return img, np.zeros((1, 5))


class COCOConditionalDataset(COCODataset):
    """
    COCO dataset that loads a 'before' image, stacks it with the 'after' (original)
    image, and returns a 6-channel input.
    """
    def __init__(
        self,
        before_data_dir,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        cache_type="ram",
    ):
        """
        Args:
            before_data_dir (str): Directory for 'before' images. It should mirror the
                                   structure of the main COCO data directory.
        """
        self.before_data_dir = before_data_dir
        super().__init__(data_dir, json_file, name, img_size, preproc, cache, cache_type)

    def pull_item(self, index):
        # Load 'after' image and its annotations using parent method
        # Note: super().pull_item returns a RESIZED image.
        after_img, label, origin_image_size, img_id = super().pull_item(index)

        # Load and resize the corresponding 'before' image
        file_name = self.annotations[index][3]
        
        # ASSUMPTION: 'before' images have the same filename and are located in a parallel directory structure.
        before_img_file = os.path.join(self.before_data_dir, self.name, file_name)
        before_img = cv2.imread(before_img_file)
        assert before_img is not None, f"'before' image not found at {before_img_file}"

        # We need to apply the exact same resizing to the 'before' image.
        # The resizing factor 'r' depends on the *original* image size.
        h, w = origin_image_size
        r = min(self.img_size[0] / h, self.img_size[1] / w)
        resized_before_img = cv2.resize(
            before_img,
            (int(w * r), int(h * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        # Ensure shapes match before stacking.
        assert after_img.shape == resized_before_img.shape, \
            f"Shape mismatch for {file_name}: after:{after_img.shape}, before:{resized_before_img.shape}"

        # Stack images along the channel axis to create a 6-channel image
        stacked_img = np.concatenate((resized_before_img, after_img), axis=2)
        
        return stacked_img, label, origin_image_size, img_id
