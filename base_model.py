
import abc
import os

import torch
from torch import nn

class BaseModel(nn.Module, abc.ABC):
    def __init__(self, pydantic_args):
        super().__init__()
        # ...

    @abc.abstractmethod
    def forward(self, x):
        pass

    def saving(self, name:str,  root_path="./checkpoints"):
        if not (name.endswith(".pt") or name.endswith(".pth")):
            name += ".pt"
        path = os.path.join(root_path, name)
        os.makedirs(root_path, exist_ok=True)
        torch.save(self.state_dict(), path)

    def loading(self, name:str,  root_path="./checkpoints", map_location=None):
        if not (name.endswith(".pt") or name.endswith(".pth")):
            name += ".pt"
        path = os.path.join(root_path, name)
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
        self.eval()

