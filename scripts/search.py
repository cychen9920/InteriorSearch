import argparse
from pathlib import Path

from interior_search.clip_encoder import ClipEncoder
from interior_search.search_index import SearchIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search a CLIP image index.")
    parser.add_argument("query", help='Text query, e.g. "cozy bedroom with plants"')
    parser.add_argument("--index", default="data/index/interior_clip.npz")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = Path(args.index)
    if not index_path.exists():
        raise SystemExit(f"Index not found: {index_path}")

    encoder = ClipEncoder(model_name=args.model_name, pretrained=args.pretrained)
    index = SearchIndex.load(index_path)
    query_embedding = encoder.encode_text(args.query)

    for rank, result in enumerate(index.search(query_embedding, top_k=args.top_k), start=1):
        print(f"{rank:02d}. {result['score']:.3f}  {result['path']}")


if __name__ == "__main__":
    main()

