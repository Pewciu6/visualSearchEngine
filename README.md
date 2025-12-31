# Visual Search Engine

A **Content-Based Image Retrieval (CBIR)** system powered by Deep Learning. This project allows users to upload an image of a fashion item (e.g., shoes, dress) and find visually similar products from a dataset, utilizing **Metric Learning** and **Vector Search**.

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![License](https://img.shields.io/badge/License-MIT-grey)

---

## Key Features
* **Deep Metric Learning:** Utilizes a **ResNet-18** backbone trained with **Triplet Margin Loss** to learn a 128-dimensional embedding space where similar items are clustered together.
* **Vector Search Engine:** Implements a custom Nearest Neighbor search using PyTorch's optimized matrix operations (Euclidean distance).
* **Production-Ready API:** A high-performance REST API built with **FastAPI**, supporting image uploads, static file serving, and returning ranked JSON results with metadata.
* **Dockerized Deployment:** Fully containerized application using a multi-stage Docker build, optimized for CPU inference (lightweight image).
* **Data Integrity:** Robust pipeline ensuring 100% synchronization between image files on disk and vector indices.

---

## System Architecture

1.  **Offline Training:** The neural network is trained on triplets (Anchor, Positive, Negative) to minimize the distance between similar items and maximize the distance between dissimilar ones.
2.  **Indexing:** The entire dataset (44k+ images) is passed through the trained network. The resulting 128-dimensional vectors are saved to a binary index (`vectors.pt`).
3.  **Inference (Online API):**
    * User uploads a query image via API.
    * Server computes the embedding for the query image.
    * Server calculates Euclidean distances to all vectors in the database.
    * Top-K nearest neighbors are returned as JSON results.

---

## Tech Stack

* **Core:** Python 3.9
* **Deep Learning:** PyTorch, Torchvision
* **Data Processing:** Pandas, Pillow, NumPy
* **Backend:** FastAPI, Uvicorn, Python-Multipart
* **DevOps:** Docker, Git

---

## Data Setup (Important!)

Due to the size of the dataset (**Fashion Product Images**), the raw images are **not included** in this repository.

To run the project, you need to:
1.  Download the dataset (e.g., from [Kaggle - Fashion Product Images Small](https://www.kaggle.com/paramaggarwal/fashion-product-images-small)).
2.  Extract it and organize your project folder structure as follows:

```text
visual-search-engine/
├── data/
│   ├── images/       
│   └── styles.csv    
├── src/
|   |-- data/
|   |   |-- dataset.py
|   |   |-- tripley_dataset.py
|   |
|   |-- models/
|       |-- loss.py
|       |-- net.py
├── checkpoints/      # Model weights (.pth)
├── index/            # Vector database files (.pt, .json)
├── Dockerfile
├── api.py
|── build_index.py
|-- train.py
|-- Dockerfile