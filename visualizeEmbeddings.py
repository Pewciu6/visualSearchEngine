import argparse
import json

import torch
from sklearn.manifold import TSNE

DEVICE = "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet")
    args = parser.parse_args()

    print(f"Loading vectors for {args.model}...")
    vectors = torch.load(f"index/vectors_{args.model}.pt", map_location="cpu")
    vectors_np = vectors.numpy()

    labels = None
    with open("index/indexes.json") as f:
       labels = json.load(f)

    print("Running t-sne...")
    tsne = TSNE(n_components=2, perplexity=80, random_state=42, init="pca", learning_rate="auto")
    vectors_2d = tsne.fit_transform(vectors_np)

    output_path = f"index/plot_coords_{args.model}.json"
    with open("index/filenames.json") as f:
        filenames = json.load(f)

    data = []
    for i, (x, y) in enumerate(vectors_2d):
        data.append({"label" : labels[i], "filename": filenames[i], "x": float(x), "y": float(y)})

    with open(output_path, "w") as f:
        json.dump(data, f)

    print(f"Saved visualization data to {output_path}")


if __name__ == "__main__":
    main()
