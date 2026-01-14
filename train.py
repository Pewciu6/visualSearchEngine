import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import FashionDataset
from src.data.triplet_dataset import TripletFashionDataset
from src.models.loss import TripletLoss
from src.models.net import EmbeddingNet

SEED = 42
BATCH_SIZE = 32
EMBEDDING_SIZE = 128
MARGIN = 1.0
LR = 1e-4
EPOCHS = 5
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CSV_PATH = DATA_DIR / "styles.csv"
IMG_DIR = DATA_DIR / "images"
SAVE_DIR = ROOT_DIR / "checkpoints"

SAVE_DIR.mkdir(exist_ok=True, parents=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print(f"Training on device: {DEVICE}")
    set_seed(SEED)

    transforms = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    base_dataset = FashionDataset(
        csv_file=str(CSV_PATH), root_dir=str(IMG_DIR), transform=transforms
    )
    triplet_dataset = TripletFashionDataset(base_dataset)

    dataloader = DataLoader(
        triplet_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    model = EmbeddingNet(embedding_size=EMBEDDING_SIZE, pretrained=True).to(DEVICE)
    criterion = TripletLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch")

        for anchor, positive, negative in progress_bar:
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)

            optimizer.zero_grad()

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Summary: Avg Loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_DIR / "best_model.pth")

        torch.save(model.state_dict(), SAVE_DIR / "last_model.pth")


if __name__ == "__main__":
    main()
