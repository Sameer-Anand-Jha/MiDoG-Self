import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset

class DG_Dataset(Dataset):
    def __init__(self, csv_file, domains, data_dir, shift_feature, transform=None):
        self.transform = transform
        self.domains = domains
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame[self.data_frame[shift_feature].isin(self.domains)]

        self.class_labels = self.data_frame['majority'].unique()
        self.class_to_idx = {label: idx for idx, label in enumerate(self.class_labels)}

        self.data_dir = data_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.data_frame.iloc[idx]
        img_path = os.path.join(self.data_dir, sample['image_id'])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        label = self.class_to_idx[sample['majority']]
        
        if self.transform:
            image = self.transform(image)

        return image, label
    

