import random

import torch
from torch.utils.data import Dataset

from src.data.dataset import FashionDataset


class TripletFashionDataset(Dataset):
    def __init__(self, base_dataset : FashionDataset):

        self.base_dataset = base_dataset
        self.labels = base_dataset.data['label'].tolist()

        self.label_to_indices = {}

        for idx, label in enumerate(self.labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        self.valid_labels = [ lbl for lbl, idxs in self.label_to_indices.items() if len(idxs) > 1 ]
        self.valid_labels_set = set(self.valid_labels)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index : int) -> tuple[torch.tensor, torch.tensor, torch.tensor]:

        anchor, label = self.base_dataset[index]
        if label not in self.valid_labels_set:
            return self.__getitem__(random.randint(0,len(self.base_dataset)-1))

        list_of_valid_indexes_positive = self.label_to_indices[label]

        pos_id = random.choice(list_of_valid_indexes_positive)
        while pos_id == index:
            pos_id = random.choice(list_of_valid_indexes_positive)

        neg_label = random.choice(self.valid_labels)
        while neg_label == label:
            neg_label = random.choice(self.valid_labels)

        list_of_valid_indexes_negative = self.label_to_indices[neg_label]
        neg_id = random.choice(list_of_valid_indexes_negative)

        negative_img, _ = self.base_dataset[neg_id]
        positive_img, _ = self.base_dataset[pos_id]

        return anchor, positive_img, negative_img
