import json
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import FashionDataset
from src.models.net import EmbeddingNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EMBEDDING_SIZE= 128

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CSV_PATH = DATA_DIR / "styles.csv"
IMG_DIR = DATA_DIR / "images"
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "best_model.pth"
INDEX_PATH = ROOT_DIR / "index"

INDEX_PATH.mkdir(exist_ok=True)

def main():

    model = EmbeddingNet(embedding_size=EMBEDDING_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    TRANSFORMS = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = FashionDataset(csv_file=CSV_PATH, root_dir=IMG_DIR, transform=TRANSFORMS)

    def check_file_exists(row):
        path = IMG_DIR / f"{row['id']}.jpg"
        return path.exists()

    dataset.data = dataset.data[dataset.data.apply(check_file_exists, axis=1)]
    dataset.data = dataset.data.reset_index(drop=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_embeddings = []
    all_paths = []

    with torch.no_grad():
        for i, (images, _labels) in enumerate(tqdm(dataloader)):

            images = images.to(DEVICE)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())

            start_idx = i * BATCH_SIZE
            end_idx = start_idx + len(images)

            batch_ids = dataset.data.iloc[start_idx:end_idx]['id'].astype(str).tolist()

            batch_filenames = [f"{pid}.jpg" for pid in batch_ids]
            all_paths.extend(batch_filenames)

    vector_db = torch.cat(all_embeddings)
    torch.save(vector_db, INDEX_PATH / "vectors.pt")

    with open(INDEX_PATH / "filenames.json", "w") as f:
        json.dump(all_paths, f)

if __name__ == "__main__":
    main()
