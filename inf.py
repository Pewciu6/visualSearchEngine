import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image

# Importy z Twojej biblioteki src
from src.models.net import EmbeddingNet
from src.data.dataset import FashionDataset

# --- KONFIGURACJA ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_SIZE = 128
BATCH_SIZE = 128        # Większy batch do inferencji (szybciej)

# Ścieżki
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
CSV_PATH = DATA_DIR / "styles.csv"
IMG_DIR = DATA_DIR / "images"
CHECKPOINT_PATH = ROOT_DIR / "checkpoints/best_model.pth"

def load_model(path: Path) -> EmbeddingNet:
    print(f"🧠 Ładowanie modelu z: {path}")
    model = EmbeddingNet(embedding_size=EMBEDDING_SIZE, pretrained=False).to(DEVICE)
    # Ładujemy wagi
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()  # Wyłącza Dropout i BatchNorm w tryb testowy
    return model

def create_embeddings(model, dataloader):
    """
    Tworzy 'Indeks' - przepuszcza wszystkie zdjęcia przez sieć i zapamiętuje wektory.
    """
    print("📇 Generowanie embeddingów dla całej bazy (Indexing)...")
    embeddings = []
    labels = []
    
    with torch.no_grad(): # Wyłączamy obliczanie gradientów (oszczędność RAM i czasu)
        for imgs, lbls in tqdm(dataloader, desc="Indexing"):
            imgs = imgs.to(DEVICE)
            emb_batch = model(imgs)
            embeddings.append(emb_batch.cpu()) # Zrzucamy na CPU, żeby nie zapchać GPU
            labels.append(lbls)
            
    # Sklejamy listę batchy w jeden duży tensor [N, 128]
    all_embeddings = torch.cat(embeddings)
    all_labels = torch.cat(labels)
    
    print(f"✅ Indeks gotowy. Kształt bazy: {all_embeddings.shape}")
    return all_embeddings, all_labels

def find_similar_images(query_idx, all_embeddings, k=5):
    """
    Znajduje 5 najbliższych sąsiadów w przestrzeni wektorowej.
    """
    # 1. Pobierz wektor zapytania
    query_vec = all_embeddings[query_idx].unsqueeze(0)  # Shape: [1, 128]
    
    # 2. Oblicz dystans Euklidesowy do WSZYSTKICH wektorów w bazie
    # cdist jest bardzo zoptymalizowane w PyTorch
    distances = torch.cdist(query_vec, all_embeddings, p=2)  # Shape: [1, N]
    
    # 3. Wybierz K najmniejszych dystansów
    # values = odległości, indices = numery zdjęć w bazie
    values, indices = torch.topk(distances, k=k+1, largest=False)
    
    # Odrzucamy pierwszy wynik (to samo zdjęcie, dystans 0) i bierzemy 5 kolejnych
    return values.squeeze()[1:], indices.squeeze()[1:]

def visualize_results(dataset, query_idx, result_indices, distances):
    """
    Rysuje wynik w matplotlib i zapisuje do pliku.
    """
    # Denormalizacja (żeby kolory wyglądały normalnie, a nie jak negatyw ImageNet)
    inv_normalize = T.Compose([
        T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
        T.ToPILImage()
    ])

    plt.figure(figsize=(15, 6))
    
    # Rysujemy Query (Zapytanie)
    query_img, query_label = dataset[query_idx]
    plt.subplot(1, 6, 1)
    plt.imshow(inv_normalize(query_img))
    plt.title(f"QUERY\n{dataset.get_class_name(query_label)}")
    plt.axis("off")
    
    # Rysujemy Wyniki
    for i, idx in enumerate(result_indices):
        idx = idx.item()
        dist = distances[i].item()
        
        img, lbl = dataset[idx]
        
        plt.subplot(1, 6, i + 2)
        plt.imshow(inv_normalize(img))
        
        # Nazwa klasy
        class_name = dataset.get_class_name(lbl)
        is_same_class = (lbl == query_label)
        color = "green" if is_same_class else "red"
        
        plt.title(f"Dist: {dist:.2f}\n{class_name}", color=color, fontsize=10)
        plt.axis("off")
        
    plt.tight_layout()
    output_file = f"result_{query_idx}.png"
    plt.savefig(output_file)
    print(f"🖼️ Wynik zapisano jako: {output_file}")
    plt.close()

def main():
    # 1. Setup Danych (Taki sam jak w train.py)
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Uwaga: Tutaj używamy zwykłego FashionDataset (nie Triplet), bo chcemy po prostu listę zdjęć
    dataset = FashionDataset(str(CSV_PATH), str(IMG_DIR), transform=transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 2. Ładowanie Modelu
    if not CHECKPOINT_PATH.exists():
        print("❌ Nie znaleziono modelu! Uruchom najpierw train.py")
        return
        
    model = load_model(CHECKPOINT_PATH)
    
    # 3. Indeksowanie (To potrwa chwilę - 44k zdjęć przez GPU)
    all_embeddings, all_labels = create_embeddings(model, dataloader)
    
    # 4. Pętla Testowa - Losujemy 3 przykłady
    print("\n🔍 Testowanie wyszukiwarki...")
    for _ in range(3):
        query_idx = random.randint(0, len(dataset)-1)
        
        dists, indices = find_similar_images(query_idx, all_embeddings, k=5)
        visualize_results(dataset, query_idx, indices, dists)

if __name__ == "__main__":
    main()