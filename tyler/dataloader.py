import os
from cv2 import transform

import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion"
]

class ChexpertBaseDataset(Dataset):
    def __init__(self, root_dir, df, transforms=None, classes=None, use_frontal=True, uncertainty_method="zero", smoothing_lower_bound=0, smoothing_upper_bound=1):
        if transforms:
            self.transforms = Compose(transforms)
        else:
            self.transforms = None
        df = df.copy()

        # load up the data
        if classes:
            labels = classes
        else:
            labels = df.iloc[:, 5:].columns.tolist()
        df[labels] = df[labels].fillna(0)

        self.labels = labels

        # preprocessing
        if use_frontal:
            df = df[df["Frontal/Lateral"] == "Frontal"]

        if uncertainty_method == "zero":
            df[labels] = df[labels].replace(-1, 0)
        elif uncertainty_method == "one":
            df[labels] = df[labels].replace(-1, 1)
        elif uncertainty_method == "smooth":
            for col in labels:
                df.loc[df[col] == -1, col] = np.random.uniform(smoothing_lower_bound, smoothing_upper_bound, size=df.loc[df[col] == -1, col].shape)
        
        self.image_paths = [os.path.join(root_dir, path) for path in df["Path"].tolist()]
        self.image_labels = df[labels].values.tolist()

        assert len(self.image_paths) == len(self.image_labels)

    def __len__(self):
        return len(self.image_paths)

    def _preprocess_image(self, image):
        image = self.transforms(image)
        return image

    def __getitem__(self, index):
        raise NotImplementedError

class ChexpertDensenetDataset(ChexpertBaseDataset):
    def __getitem__(self, index):
        path = self.image_paths[index]
        data = Image.open(path).convert("RGB")

        if self.transforms:
            data = self._preprocess_image(data)

        return torch.squeeze(data, 0), torch.Tensor(self.image_labels[index])

class ChexpertDensenetDataloader(DataLoader):
    def __init__(self, root_dir, df, transforms=None, classes=None, use_frontal=True, uncertainty_method="zeros", smoothing_lower_bound=0, smoothing_upper_bound=1,
                 batch_size=1, shuffle=False, num_workers=0):
        dataset = ChexpertDensenetDataset(root_dir, df, transforms, classes, use_frontal, uncertainty_method, smoothing_lower_bound, smoothing_upper_bound)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class ChexpertViTDataset(ChexpertBaseDataset):
    def __init__(self, root_dir, df, feature_extractor, include_labels=True, transforms=None, classes=None, use_frontal=True, uncertainty_method="zero", smoothing_lower_bound=0, smoothing_upper_bound=1):
        self.feature_extractor = feature_extractor
        self.include_labels = include_labels
        super().__init__(root_dir, df, transforms, classes, use_frontal, uncertainty_method, smoothing_lower_bound, smoothing_upper_bound)

    def __getitem__(self, index):
        path = self.image_paths[index]
        data = Image.open(path).convert("RGB")

        if self.transforms:
            data = self._preprocess_image(data)

        features = self.feature_extractor(data, return_tensors='pt')
        features['pixel_values'] = torch.squeeze(features['pixel_values'], 0)
        if self.include_labels:
            features['labels'] = torch.Tensor(self.image_labels[index])
        return features

class ChexpertViTDataloader(DataLoader):
    def __init__(self, root_dir, df, feature_extractor, use_frontal=True, uncertainty_method="zero", smoothing_lower_bound=0, smoothing_upper_bound=1,
                 batch_size=1, shuffle=False, num_workers=0):
        dataset = ChexpertViTDataset(root_dir, df, feature_extractor, use_frontal, uncertainty_method, smoothing_lower_bound, smoothing_upper_bound)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)