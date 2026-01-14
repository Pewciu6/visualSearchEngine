import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class FashionDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform=None):
        """
        Args:
            csv_file (str): Path to styles.csv
            root_dir (str): Folder consisting of images
            transform (callable, optional): Transform to be applied on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform

        df = pd.read_csv(csv_file, on_bad_lines='skip')
        df['filename'] = df['id'].astype(str) + '.jpg'
        available_files = set(os.listdir(root_dir))

        initial_count = len(df)
        df = df[df['filename'].isin(available_files)]
        final_count = len(df)

        if initial_count != final_count:
            print(f"Deleted {initial_count - final_count} entries due to missing images.")

        self.data = df.reset_index(drop=True)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.data['articleType'].unique())}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        self.data['label'] = self.data['articleType'].map(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[any, int]:
        row = self.data.iloc[idx]
        img_name = os.path.join(self.root_dir, row['filename'])
        label = row['label']

        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, label_idx: int) -> str:
        return self.idx_to_class.get(label_idx, "Unknown")
