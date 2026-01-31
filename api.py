import io
import json
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

MODELS = {}
VECTORS = {}
FILENAMES = {}
TRANSFORM = None


@app.on_event("startup")
def load_model():
    global TRANSFORM

    configs = {
        "resnet": {"path": "checkpoints/best_model_resnet.pth", "index": "index/vectors_resnet.pt"},
        "vit": {"path": "checkpoints/best_model_vit.pth", "index": "index/vectors_vit.pt"},
    }

    TRANSFORM = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    for name, cfg in configs.items():
        print(f"Loading model: {name}")
        net = EmbeddingNet(architecture=name, embedding_size=128, pretrained=False)

        if Path(cfg["path"]).exists():
            net.load_state_dict(torch.load(cfg["path"]), map_location=DEVICE)
            net.to(DEVICE)
            net.eval()
            MODELS[name] = net
        else:
            print(f"Warning: No weights found for {name} at {cfg['path']}")

        if Path(cfg["index"]).exists():
            VECTORS[name] = torch.load(cfg["index"], map_location=DEVICE)
            with open(INDEX_DIR / "filenames.json") as f:
                FILENAMES[name] = json.load(f)


@app.post("/search")
async def similar(file: UploadFile = File, top_k: int = 5, model_type: str = "resnet"):
    if TRANSFORM is None:
        raise HTTPException(status_code=500, detail="Image transforms not initialized.")

    if model_type not in MODELS:
        raise HTTPException(
            status_code=400, detail=f"Model {model_type} not available choose from: [resnet, vit]"
        )

    active_model = MODELS[model_type]
    active_vectors = VECTORS.get(model_type)
    active_filenames = FILENAMES.get(model_type)

    if active_vectors is None or active_filenames is None:
        raise HTTPException(status_code=503, detail="Vector database not ready. Index missing.")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    input_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        query_vector = active_model(input_tensor)

    distances = torch.cdist(query_vector, active_vectors, p=2)
    values, indicies = torch.topk(distances, top_k, largest=False)

    results = []
    best_indices = indicies[0].cpu().numpy()
    best_distances = values[0].cpu().numpy()

    for i, idx in enumerate(best_indices):
        filename = active_filenames[idx]
        dist = float(best_distances[i])

        results.append(
            {
                "rank": i + 1,
                "filename": filename,
                "distance": round(dist, 4),
                "image_url": f"http://127.0.0.1:8000/images/{filename}",
            }
        )

    return {"results": results, "model": model_type}


@app.get("/")
def root():
    return {"status": "System works", "model": "ResNet-Triplets"}
