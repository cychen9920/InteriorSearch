from pathlib import Path

import streamlit as st

from interior_search.clip_encoder import ClipEncoder
from interior_search.search_index import SearchIndex


DEFAULT_INDEX_PATH = "data/index/interior_clip.npz"


@st.cache_resource
def load_encoder() -> ClipEncoder:
    return ClipEncoder()


@st.cache_resource
def load_index(index_path: str) -> SearchIndex:
    return SearchIndex.load(index_path)


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

query = st.text_input("Search interiors", placeholder="cozy bedroom with plants")

if not query:
    st.stop()

with st.spinner("Searching..."):
    encoder = load_encoder()
    index = load_index(index_path)
    query_embedding = encoder.encode_text(query)
    results = index.search(query_embedding, top_k=top_k)

if not results:
    st.warning("No results found.")
    st.stop()

columns = st.columns(4)
for i, result in enumerate(results):
    path = Path(result["path"])
    with columns[i % len(columns)]:
        st.image(str(path), use_container_width=True)
        st.caption(f"{result['score']:.3f} · {path.name}")

