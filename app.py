from pathlib import Path

import streamlit as st
from PIL import Image

from interior_search.clip_encoder import ClipEncoder
from interior_search.search_index import SearchIndex


DEFAULT_INDEX_PATH = "data/index/interior_clip.npz"


@st.cache_resource
def load_encoder() -> ClipEncoder:
    return ClipEncoder()


@st.cache_resource
def load_index(index_path: str) -> SearchIndex:
    return SearchIndex.load(index_path)


def render_results(results: list[dict]) -> None:
    if not results:
        st.warning("No results found.")
        return

    columns = st.columns(4)
    for i, result in enumerate(results):
        path = Path(result["path"])
        with columns[i % len(columns)]:
            st.image(str(path), use_container_width=True)
            st.caption(f"{result['score']:.3f} · {path.name}")


st.set_page_config(page_title="InteriorLens", layout="wide")

st.title("InteriorLens")
st.caption("Semantic search for interior design inspiration")

index_path = st.sidebar.text_input("Index path", DEFAULT_INDEX_PATH)
top_k = st.sidebar.slider("Results", min_value=4, max_value=24, value=12, step=4)

if not Path(index_path).exists():
    st.info(
        "Build an index first with "
        "`python scripts/build_index.py --image-dir data/images`."
    )
    st.stop()

encoder = load_encoder()
index = load_index(index_path)

text_tab, image_tab = st.tabs(["Text search", "Image search"])

with text_tab:
    query = st.text_input("Search interiors", placeholder="cozy bedroom with plants")

    if query:
        with st.spinner("Searching..."):
            query_embedding = encoder.encode_text(query)
            results = index.search(query_embedding, top_k=top_k)

        render_results(results)

with image_tab:
    uploaded_file = st.file_uploader(
        "Upload an interior image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Query image", width=320)

        with st.spinner("Finding similar interiors..."):
            query_embedding = encoder.encode_pil_image(uploaded_image)
            results = index.search(query_embedding, top_k=top_k)

        render_results(results)
