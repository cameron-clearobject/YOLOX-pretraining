#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import CondExp

class Exp(CondExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/coco128"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.before_data_dir = "./datasets/coco128-before"

        self.num_classes = 71

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1
