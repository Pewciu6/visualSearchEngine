# Visual Search Engine

A **Content-Based Image Retrieval (CBIR)** system powered by Deep Learning. This project allows users to upload an image of a fashion item (e.g., shoes, dress) and find visually similar products from a dataset.

It features a **Dual-Model Architecture**, allowing users to switch between **ResNet-18** (CNN) and **Vision Transformer (ViT)** to compare search results based on texture vs. semantic shape.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.22-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-grey)

---

## Key Features

* **Dual AI Models:** Choose between two state-of-the-art architectures:
    * **ResNet-18:** Excellent for capturing local textures and patterns.
    * **Vision Transformer (ViT-B/16):** Superior at understanding global context and semantic shape.
* **Interactive Frontend:** A user-friendly web interface built with **Streamlit** for uploading images, adjusting search parameters, and visualizing results in a grid.
* **Deep Metric Learning:** Models are trained with **Triplet Margin Loss** to create a 128-dimensional embedding space where visually similar items are clustered together.
* **Vector Search Engine:** Custom Nearest Neighbor search using PyTorch's optimized matrix operations (Euclidean distance).
* **Production-Ready API:** High-performance REST API built with **FastAPI**, serving the dual-model backend and handling image processing.

---

## System Architecture

1.  **Offline Training:** Two separate neural networks (ResNet & ViT) are trained on triplets (Anchor, Positive, Negative) to learn distinct embedding spaces.
2.  **Indexing:** The dataset is processed through *both* models. This results in two separate vector indices:
    * `vectors_resnet.pt`
    * `vectors_vit.pt`
3.  **Inference (Online):**
    * **Frontend:** User selects a model (e.g., "ViT") and uploads an image via Streamlit.
    * **API:** Receives the image and the model choice.
    * **Search:** The backend routes the image to the selected model, generates the embedding, and queries the corresponding vector index.
    * **Result:** Top-K matches are returned and displayed to the user.

---

## Tech Stack

* **Core:** Python 3.11
* **Deep Learning:** PyTorch, Torchvision (ResNet, ViT)
* **Frontend:** Streamlit
* **Backend:** FastAPI, Uvicorn
* **Data Processing:** Pandas, Pillow, NumPy

---

## Data Setup

Due to the size of the dataset (**Fashion Product Images**), the raw images are **not included** in this repository.

1.  Download the dataset (e.g., [Kaggle - Fashion Product Images Small](https://www.kaggle.com/paramaggarwal/fashion-product-images-small)).
2.  Extract it and organize your project folder structure as follows:

```text
visual-search-engine/
├── data/
│   ├── images/       # Put your 40k+ images here
│   └── styles.csv    # The metadata CSV
├── src/
│   ├── data/
│   └── models/
├── checkpoints/      # Model weights will be saved here
├── index/            # Vector database files (.pt, .json)
├── api.py
├── build_index.py
├── frontend.py
├── train.py
└── requirements.txt
```

## Local Installation

Clone the repository:
```text
git clone [https://github.com/your-username/visual-search-engine.git](https://github.com/your-username/visual-search-enginegit)
cd visual-search-engine
```

Install dependencies:
```text
pip install -r requirements.txt
```

Train the models:
```text
python train.py
```

Build the Index: This script processes all images and creates the vector database in the index/ folder.

```text
python build_index.py
```

Start the Server:

```text
uvicorn api:app --reload --port 8000
```

```text
streamlit run frontend.py
```
