import numpy as np
from torch.utils.data import Dataset

# ToDo: Remove dependence on torch and have the dataset implemented as an Iterable
# ToDo: Remove torch from the pip_requirements.txt


class AbstractDataset(Dataset):
    """
    A dataset implements 2 functions
        - __len__  (returns the number of samples in our dataset)
        - __getitem__ (returns a sample from the dataset at the given index idx)
    """

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.random_state = np.random.RandomState(self.params.get("seed", 123))

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def _load_data(self):
        raise NotImplementedError()
