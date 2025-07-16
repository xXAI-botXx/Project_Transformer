from random_dataset import RandomDataset
from physgen_dataset import PhysGenDataset

class MixDataset(Dataset):

    def __init__(self):
        """
        ...
        """
        # load random dataset
        self.random_dataset = RandomDataset()
        self.len_random_dataset = len(self.random_dataset)

        # load physgen dataset
        self.physgen_dataset = Physgen(variation="sound_reflection", mode="train", input_type="osm", output_type="standard")

    def __len__(self):
        return len(self.random_dataset) + len(self.physgen_dataset)

    def __getitem__(self, idx):
        if idx > self.len_random_dataset:
            updated_idx = idx - self.len_random_dataset
            return self.physgen_dataset[updated_idx]
        else:
            return self.random_dataset[idx] 
            





