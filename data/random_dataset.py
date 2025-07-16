import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset



class RandomDataset(Dataset):

    def __init__(self):
        """
        ...
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        # ...

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass