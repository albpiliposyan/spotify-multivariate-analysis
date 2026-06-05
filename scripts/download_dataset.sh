#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/dataset"
ZIP_PATH="$DATA_DIR/spotify-tracks-dataset.zip"
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/maharshipandya/-spotify-tracks-dataset"

mkdir -p "$DATA_DIR"

echo "Downloading Spotify dataset..."
curl -L --fail -o "$ZIP_PATH" "$DATASET_URL"

echo "Extracting dataset into $DATA_DIR..."
unzip -o "$ZIP_PATH" -d "$DATA_DIR"
rm "$ZIP_PATH"

if [[ ! -f "$DATA_DIR/spotify_dataset.csv" ]]; then
  csv_file="$(find "$DATA_DIR" -maxdepth 1 -type f -name '*.csv' | head -n 1)"
  if [[ -z "$csv_file" ]]; then
    echo "No CSV file found after extraction." >&2
    exit 1
  fi
  mv "$csv_file" "$DATA_DIR/spotify_dataset.csv"
fi

echo "Dataset ready: $DATA_DIR/spotify_dataset.csv"
