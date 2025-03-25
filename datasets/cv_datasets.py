import torch
from torch.utils.data import Dataset


class Statistics_CV:
    mean_fmnist = 72.94
    std_fmnist = 90.02


class CV_Dataset(Dataset):
    def __init__(self, data, labels, name="fmnist"):
        self._data = data.float()
        self._label = labels.type(torch.LongTensor)
        self._num_labels = len(self._label.unique())

        # normalizing data
        self._mean = getattr(Statistics_CV, f"mean_{name}")
        self._std = getattr(Statistics_CV, f"std_{name}")
        self._data.sub_(self._mean)
        self._data.div_(self._std)

        # add channel dimension
        if len(self._data.shape) < 4:
            self._in_features = 1
            self._data.unsqueeze_(dim=1)
        else:
            self._in_features = 3

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return {"data": self._data[index], "label": self._label[index]}

    @property
    def num_labels(self):
        return self._num_labels

    @property
    def in_features(self):
        return self._in_features
