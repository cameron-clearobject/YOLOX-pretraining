
# yolox/evaluators/pretrain_evaluator.py

from loguru import logger
import torch
import torch.nn as nn
from typing import Optional

from yolox.data import DataPrefetcher
from yolox.utils import adjust_status, synchronize

class ReconstructionEvaluator:
    """
    A simple evaluator for the unsupervised autoencoder pre-training task.
    Its purpose is to calculate the mean reconstruction loss on a validation dataset.
    """
    def __init__(self, dataloader: "DataLoader", img_size: tuple):
        self.dataloader = dataloader
        self.img_size = img_size

    def evaluate(
        self,
        model: nn.Module,
        distributed: bool = False,
        half: bool = False,
    ) -> float:
        """
        Runs the evaluation loop for the autoencoder.

        Args:
            model (nn.Module): The autoencoder model to evaluate.
            distributed (bool): Whether training is distributed or not.
            half (bool): Whether to use half-precision (FP16) or not.

        Returns:
            float: The average reconstruction loss on the validation set.
        """
        eval_model = model
        with adjust_status(eval_model, training=False):
            prefetcher = DataPrefetcher(self.dataloader)
            criterion = nn.MSELoss()
            total_loss = 0.0
            num_batches = 0

            logger.info("Running pre-training evaluation...")
            
            inps, _ = prefetcher.next()
            while inps is not None:
                if half:
                    inps = inps.half()

                with torch.no_grad():
                    reconstructed = eval_model(inps)
                    loss = criterion(reconstructed, inps)

                if distributed:
                    loss = torch.distributed.all_reduce(loss) / torch.distributed.get_world_size()

                total_loss += loss.item()
                num_batches += 1
                inps, _ = prefetcher.next()

        synchronize() # Wait for all processes to finish evaluation
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss