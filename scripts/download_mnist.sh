#!/usr/bin/env bash
set -euo pipefail

MNIST_DIR="data/mnist"
BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"

FILES=(
  "train-images-idx3-ubyte.gz"
  "train-labels-idx1-ubyte.gz"
  "t10k-images-idx3-ubyte.gz"
  "t10k-labels-idx1-ubyte.gz"
)

mkdir -p "$MNIST_DIR"

for file in "${FILES[@]}"; do
    if [[ ! -f "$MNIST_DIR/$file" && ! -f "$MNIST_DIR/${file%.gz}" ]]; then
        echo "Downloading $file..."
        curl -L "$BASE_URL/$file" -o "$MNIST_DIR/$file"
    else
        echo "$file already exists, skipping."
    fi
done

for file in "${FILES[@]}"; do
    if [[ -f "$MNIST_DIR/$file" ]]; then
        echo "Extracting $file..."
        gunzip -f "$MNIST_DIR/$file"
    fi
done

echo "MNIST ready in $MNIST_DIR"