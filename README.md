# Spotify Multivariate Analysis

Python code for generating the Spotify multivariate analysis tables and figures.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Dataset

```bash
bash scripts/download_dataset.sh
```

This downloads the Kaggle dataset from:

```text
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
```

The extracted CSV is saved as:

```text
dataset/spotify_dataset.csv
```

## Run Analysis

```bash
python spotify_analysis.py
```

Generated files are written to:

```text
output/figures/
output/tables/
output/analysis_summary.txt
```
