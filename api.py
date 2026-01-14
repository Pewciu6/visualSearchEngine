import io
import json
from collections.abc import Callable
from pathlib import Path

import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.models.net import EmbeddingNet

app = FastAPI(title="Visual Search Engine")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_SIZE = 128
ROOT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "best_model.pth"
INDEX_DIR = ROOT_DIR / "index"
IMG_DIR = ROOT_DIR / "data" / "images"

app.mount("/images", StaticFiles(directory=IMG_DIR), name="images")

model: Callable | None = None
transform: Callable | None = None
db_vectors: torch.Tensor | None = None
db_filenames: list[str] | None = None


@app.on_event("startup")
def load_model():
    global model, transform, db_vectors, db_filenames

    model = EmbeddingNet(embedding_size=EMBEDDING_SIZE, pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    if not (INDEX_DIR / "vectors.pt").exists():
        raise FileNotFoundError(
            "No index file found. Please create the index before starting the API."
        )
    else:
        db_vectors = torch.load(INDEX_DIR / "vectors.pt", map_location=DEVICE)
        with open(INDEX_DIR / "filenames.json") as f:
            db_filenames = json.load(f)

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@app.post("/predict")
async def predict(file: UploadFile = File):

    if model is None or transform is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(input_tensor)

    return {
        "filename": file.filename,
        "vector_sample": embedding.cpu().numpy()[0][:10].tolist(),
        "message": "Embedding generated successfully!",
    }


@app.post("/search")
async def similar(file: UploadFile = File, top_k: int = 5):

    if model is None or transform is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")

    if db_vectors is None or db_filenames is None:
        raise HTTPException(status_code=503, detail="Vector database not ready. Index missing.")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        query_vector = model(input_tensor)

    distances = torch.cdist(query_vector, db_vectors, p=2)
    values, indicies = torch.topk(distances, top_k, largest=False)

    results = []
    best_indices = indicies[0].cpu().numpy()
    best_distances = values[0].cpu().numpy()

    for i, idx in enumerate(best_indices):
        filename = db_filenames[idx]
        dist = float(best_distances[i])

        results.append(
            {
                "rank": i + 1,
                "filename": filename,
                "distance": round(dist, 4),
                "url": f"/images/{filename}",
                "image_url": f"http://127.0.0.1:8000/images/{filename}",
            }
        )

    return {"query_filename": file.filename, "results": results}


@app.get("/")
def root():
    return {"status": "System works", "model": "ResNet-Triplets"}
