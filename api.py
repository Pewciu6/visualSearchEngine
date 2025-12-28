import io

import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from src.models.net import EmbeddingNet

app = FastAPI(title='Visual Search Engine')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_SIZE = 128
CHECKPOINT_PATH = 'checkpoints/best_model.pth'

model = None
transform = None

@app.on_event('startup')
def load_model():
    global model, transform

    model = EmbeddingNet(embedding_size=EMBEDDING_SIZE, pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

@app.post('/predict')
async def predict(file : UploadFile = File):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(input_tensor)

    return {
        "filename": file.filename,
        "vector_sample": embedding.cpu().numpy()[0][:10].tolist(),
        "message": "Embedding generated successfully!"
    }

@app.get("/")
def root():
    return {"status": "System działa", "model": "ResNet-Triplets"}
