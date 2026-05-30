import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from interior_search.clip_encoder import ClipEncoder
from interior_search.image_files import find_images
from interior_search.search_index import SearchIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a CLIP image search index.")
    parser.add_argument("--image-dir", default="data/images", help="Folder of images.")
    parser.add_argument(
        "--output",
        default="data/index/interior_clip.npz",
        help="Where to write the compressed index.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = find_images(args.image_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {Path(args.image_dir).resolve()}")

    encoder = ClipEncoder(model_name=args.model_name, pretrained=args.pretrained)
    embedding_batches = []

    print(f"Encoding {len(image_paths)} images on {encoder.device}...")
    for start in tqdm(range(0, len(image_paths), args.batch_size)):
        batch_paths = image_paths[start : start + args.batch_size]
        embedding_batches.append(encoder.encode_images(batch_paths))

    embeddings = np.concatenate(embedding_batches, axis=0)
    index = SearchIndex([str(path) for path in image_paths], embeddings)
    index.save(args.output)

    print(f"Wrote index to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

