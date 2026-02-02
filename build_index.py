import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import FashionDataset
from src.models.net import EmbeddingNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_SIZE = 128

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CSV_PATH = DATA_DIR / "styles.csv"
IMG_DIR = DATA_DIR / "images"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
INDEX_PATH = ROOT_DIR / "index"

INDEX_PATH.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Build Search Index")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet",
        choices=["resnet", "vit"],
        help="Architecture to use: 'resnet' or 'vit'",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (lower it for ViT if OOM occurs)",
    )
    args = parser.parse_args()

    print(f"Building index for: {args.model.upper()}")

    checkpoint_path = CHECKPOINT_DIR / f"best_model_{args.model}.pth"
    output_vectors_path = INDEX_PATH / f"vectors_{args.model}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Weights not found: {checkpoint_path}. Did you train this model?")

    model = EmbeddingNet(architecture=args.model, embedding_size=EMBEDDING_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    TRANSFORMS = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = FashionDataset(csv_file=CSV_PATH, root_dir=IMG_DIR, transform=TRANSFORMS)

    def check_file_exists(row):
        path = IMG_DIR / f"{row['id']}.jpg"
        return path.exists()

    dataset.data = dataset.data[dataset.data.apply(check_file_exists, axis=1)]
    dataset.data = dataset.data.reset_index(drop=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_embeddings = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(dataloader)):
            images = images.to(DEVICE)
            embeddings = model(images)

            all_embeddings.append(embeddings.cpu())
            all_labels.extend(labels.tolist())

            start_idx = i * args.batch_size
            end_idx = start_idx + len(images)

            batch_ids = dataset.data.iloc[start_idx:end_idx]["id"].astype(str).tolist()

            batch_filenames = [f"{pid}.jpg" for pid in batch_ids]
            all_paths.extend(batch_filenames)

            if i > 200:
                break

    vector_db = torch.cat(all_embeddings)
    torch.save(vector_db, output_vectors_path)
    print(f"Saved vectors to: {output_vectors_path}")

    with open(INDEX_PATH / "filenames.json", "w") as f:
        json.dump(all_paths, f)
    print("Saved filenames.json")

    with open(INDEX_PATH / "indexes.json", "w") as f:
        json.dump(all_labels, f)
    print("Saved labels to index")


if __name__ == "__main__":
    main()
