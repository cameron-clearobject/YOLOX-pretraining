
from torch.utils.data import DataLoader

class ReconstructionEvaluator:
    """A simple evaluator for the pre-training task."""
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
