# dataset.py
import torch
from torch.utils.data import Dataset
import cv2
import json
import torchvision.transforms as T

class CyanobacteriaDataset(Dataset):
    def __init__(self, annotations_file, transforms=None):
        with open(annotations_file) as f:
            self.annotations = json.load(f)
        self.transforms = transforms
        self.classes = {cls_name: idx for idx, cls_name in enumerate(sorted(set(a['class'] for a in self.annotations)))}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img = cv2.imread(ann['filename'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = torch.tensor([ann['bbox']], dtype=torch.float32)
        labels = torch.tensor([self.classes[ann['class']]], dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels}
        if self.transforms:
            img = self.transforms(img)
        return img, target
