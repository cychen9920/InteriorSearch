# InteriorLens

Search for interior design inspiration images using a pretrained PyTorch CLIP model.

Type a query like:

- `cozy bedroom with plants`
- `minimalist white kitchen`
- `green tile bathroom`
- `living room with wood furniture`

This projects embeds images and text into the same vector space, then ranks images by cosine similarity.

## What's Included

- Local image dataset
- PyTorch/OpenCLIP image and text embeddings
- NumPy cosine-similarity search
- Streamlit search app

CPU-friendly (no GPU required).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Add Images

Put your interior-design images in:

```text
data/images/
```

This version uses whatever images are in that folder. Later, you can replace or
expand them with images from an open-source dataset.

## Build the Index

Build the image index from the local folder:

```bash
python scripts/build_index.py --image-dir data/images --output data/index/interior_clip.npz
```

Run the app:

```bash
streamlit run app.py
```

Whenever you add or remove images, rebuild the index.

`data/images/` and `data/index/` are ignored by git.

## How It Works

1. `scripts/build_index.py` loads each image and runs it through CLIP's image encoder.
2. The normalized image embeddings are saved to `data/index/interior_clip.npz`.
3. `app.py` encodes your text query with CLIP's text encoder.
4. The app ranks images by dot product, which is cosine similarity because both vectors are normalized.
