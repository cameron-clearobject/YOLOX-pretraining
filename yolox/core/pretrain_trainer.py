
# yolox/core/pretrain_trainer.py

import datetime
import os
import time
from loguru import logger

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer, ModelEMA, WandbLogger,
    get_local_rank, get_model_info, get_rank, get_world_size,
    gpu_mem_usage, is_parallel, load_ckpt, mem_usage,
    save_checkpoint, setup_logger, synchronize
)

class PretrainTrainer:
    def __init__(self, exp, args):
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = f"cuda:{self.local_rank}"
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="pretrain_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()

        # Unpack data, ignoring targets
        inps, _ = self.prefetcher.next()
        inps = inps.to(self.data_type)
        inps, _ = self.exp.preprocess(inps, None, self.input_size)
        data_end_time = time.time()

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

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            total_loss=loss, # Log the reconstruction loss
        )

    def before_train(self):
        logger.info(f"args: {self.args}")
        logger.info(f"exp value:\n{self.exp}")

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(f"Model Summary: {get_model_info(model, self.exp.input_size)}")
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)
        self.criterion = nn.MSELoss()

        # resume training
        model = self.resume_train(model)

        # data related init
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        self.prefetcher = DataPrefetcher(self.train_loader)
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        # Setup loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                self.wandb_logger = WandbLogger.initialize_wandb_logger(
                    self.args, self.exp, self.train_loader.dataset
                )
            else:
                raise ValueError("logger must be either 'tensorboard' or 'wandb'")

        logger.info("Pre-training start...")

    def after_train(self):
        logger.info("Pre-training of experiment is done.")
        self.save_backbone()

    def before_epoch(self):
        # This hook is for supervised training (mosaic/L1 loss), so we just log the epoch
        logger.info(f"---> start pre-train epoch {self.epoch + 1}")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        # Use eval_interval to control how often history checkpoints are saved
        if (self.epoch + 1) % self.exp.eval_interval == 0:
            if self.save_history_ckpt:
                self.save_ckpt(f"epoch_{self.epoch + 1}")

    def before_iter(self):
        pass

    def after_iter(self):
        if (self.iter + 1) % self.exp.print_interval == 0:
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}"

            progress_str = f"epoch: {self.epoch + 1}/{self.max_epoch}, iter: {self.iter + 1}/{self.max_iter}"
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join([f"{k}: {v.latest:.4f}" for k, v in loss_meter.items()])
            time_str = f"iter_time: {self.meter['iter_time'].avg:.3f}s, data_time: {self.meter['data_time'].avg:.3f}s"
            mem_str = f"gpu mem: {gpu_mem_usage():.0f}Mb"

            logger.info(f"{progress_str}, {mem_str}, {time_str}, {loss_str}, lr: {self.meter['lr'].latest:.6f}, size: {self.input_size[0]}, {eta_str}")

            if self.rank == 0 and self.args.logger == "tensorboard":
                self.tblogger.add_scalar("pretrain/lr", self.meter["lr"].latest, self.progress_in_iter)
                for k, v in loss_meter.items():
                    self.tblogger.add_scalar(f"pretrain/{k}", v.latest, self.progress_in_iter)
            self.meter.clear_meters()

        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            ckpt_file = self.args.ckpt or os.path.join(self.file_name, "latest_ckpt.pth")
            
            if os.path.isfile(ckpt_file):
                ckpt = torch.load(ckpt_file, map_location=self.device)
                model.load_state_dict(ckpt["model"])
                self.optimizer.load_state_dict(ckpt["optimizer"])
                start_epoch = ckpt.get("start_epoch", 0)
                self.start_epoch = start_epoch
                logger.info(f"loaded checkpoint '{ckpt_file}' (epoch {self.start_epoch})")
            else:
                logger.warning(f"No checkpoint found at {ckpt_file}, starting from scratch.")
                self.start_epoch = 0
        else:
            self.start_epoch = 0
        return model

    def save_ckpt(self, ckpt_name):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(ckpt_state, False, self.file_name, ckpt_name)

    def save_backbone(self):
        if self.rank == 0:
            save_path = os.path.join(self.file_name, self.exp.output_weights_path)
            model_to_save = self.ema_model.ema if self.use_model_ema else self.model
            if is_parallel(model_to_save):
                model_to_save = model_to_save.module
            
            torch.save(model_to_save.encoder.state_dict(), save_path)
            logger.success(f"Final pre-trained backbone weights saved to {save_path}")