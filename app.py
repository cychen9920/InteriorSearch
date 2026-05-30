from pathlib import Path

import streamlit as st
from PIL import Image

from interior_search.clip_encoder import ClipEncoder
from interior_search.search_index import SearchIndex


DEFAULT_INDEX_PATH = "data/index/interior_clip.npz"
EXAMPLE_QUERIES = [
    "warm wood kitchen",
    "modern living room",
    "bright bedroom",
    "historic dining room",
]


@st.cache_resource
def load_encoder() -> ClipEncoder:
    return ClipEncoder()


@st.cache_resource
def load_index(index_path: str) -> SearchIndex:
    return SearchIndex.load(index_path)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --cream: #baccbc;
            --blush: #f8e8e8;
            --mint: #e9f4ef;
            --lavender: #eee9fb;
            --ink: #3f3a3a;
            --muted: #847a7a;
            --line: #eadfda;
        }

        .stApp {
            background: var(--cream);
            color: var(--ink);
        }

        html,
        body,
        .stApp,
        .stMarkdown,
        .stText,
        .stCaptionContainer,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp p,
        .stApp label,
        .stApp button,
        .stApp input,
        .stApp textarea {
            font-family: "Trebuchet MS", "Avenir Next", Avenir, system-ui, sans-serif !important;
        }

        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.45);
            border-right: 1px solid var(--line);
        }

        .block-container {
            padding-top: 2rem;
            max-width: 1180px;
        }

        .hero {
            text-align: center;
            padding: 1.25rem 0 1.6rem;
        }

        .hero h1 {
            color: var(--ink);
            font-size: 3.1rem;
            font-weight: 760;
            letter-spacing: 0;
            margin: 0;
        }

        .hero p {
            color: var(--muted);
            font-size: 1.05rem;
            margin: 0.45rem auto 0;
            max-width: 44rem;
        }

        div[data-testid="stTabs"] button {
            color: var(--muted);
            font-weight: 650;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--ink);
        }

        div[data-testid="stTextInput"] input {
            border-radius: 999px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, 0.86);
            padding-left: 1rem;
        }

        div[data-testid="stFileUploader"] section {
            background: rgba(255, 255, 255, 0.74);
            border: 1px dashed #d6c7c2;
            border-radius: 18px;
        }

        div.stButton > button {
            border-radius: 999px;
            border: 1px solid #e3d2cf;
            background: rgba(255, 255, 255, 0.7);
            color: var(--ink);
            padding: 0.35rem 0.9rem;
        }

        div.stButton > button:hover {
            border-color: #cfaaa7;
            color: var(--ink);
            background: #fff4f0;
        }

        .section-label {
            color: var(--muted);
            font-size: 0.88rem;
            font-weight: 700;
            margin: 0.8rem 0 0.35rem;
            text-transform: uppercase;
        }

        .score-note {
            color: var(--muted);
            font-size: 0.88rem;
            margin: 0.1rem 0 1rem;
        }

        .result-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(234, 223, 218, 0.95);
            border-radius: 18px;
            box-shadow: 0 14px 35px rgba(97, 75, 61, 0.08);
            margin-bottom: 1.1rem;
            overflow: hidden;
            padding: 0.55rem;
        }

        .result-card img {
            aspect-ratio: 4 / 3;
            border-radius: 14px;
            display: block;
            height: auto;
            object-fit: cover;
            width: 100%;
        }

        .result-meta {
            align-items: center;
            display: flex;
            gap: 0.5rem;
            justify-content: space-between;
            padding: 0.55rem 0.15rem 0.05rem;
        }

        .result-name {
            color: var(--ink);
            font-size: 0.86rem;
            font-weight: 650;
            line-height: 1.2;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .score-badge {
            background: var(--lavender);
            border: 1px solid rgba(207, 197, 231, 0.8);
            border-radius: 999px;
            color: #5d5570;
            flex: 0 0 auto;
            font-size: 0.76rem;
            font-weight: 700;
            padding: 0.17rem 0.48rem;
        }

        .query-preview {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_results(results: list[dict]) -> None:
    if not results:
        st.warning("No results found.")
        return

    st.markdown(
        '<div class="score-note">Scores show CLIP cosine similarity. Higher scores are stronger matches.</div>',
        unsafe_allow_html=True,
    )

    columns = st.columns(4)
    for i, result in enumerate(results):
        path = Path(result["path"])
        with columns[i % len(columns)]:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.image(str(path), use_container_width=True)
            st.markdown(
                f"""
                <div class="result-meta">
                    <div class="result-name" title="{path.name}">{i + 1}. {path.name}</div>
                    <div class="score-badge" title="CLIP cosine similarity score">{result['score']:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)


st.set_page_config(page_title="InteriorStyle", layout="wide")
apply_theme()

st.markdown(
    """
    <div class="hero">
        <h1>InteriorStyle</h1>
        <p>Search interiors by mood, room, material, color, or visual similarity.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

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
    if "text_query" not in st.session_state:
        st.session_state.text_query = ""

    st.markdown('<div class="section-label">Try a search</div>', unsafe_allow_html=True)
    example_columns = st.columns(len(EXAMPLE_QUERIES))
    for column, example in zip(example_columns, EXAMPLE_QUERIES):
        with column:
            if st.button(example, use_container_width=True):
                st.session_state.text_query = example

    query = st.text_input(
        "Search interiors",
        placeholder="cozy bedroom with plants",
        key="text_query",
        label_visibility="collapsed",
    )

    if query:
        with st.spinner("Searching..."):
            query_embedding = encoder.encode_text(query)
            results = index.search(query_embedding, top_k=top_k)

        render_results(results)

with image_tab:
    uploaded_file = st.file_uploader(
        "Choose an interior image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)
        left, right = st.columns([1, 2.2])
        with left:
            st.markdown('<div class="section-label">Query image</div>', unsafe_allow_html=True)
            st.markdown('<div class="query-preview">', unsafe_allow_html=True)
            st.image(uploaded_image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.spinner("Finding similar interiors..."):
            query_embedding = encoder.encode_pil_image(uploaded_image)
            results = index.search(query_embedding, top_k=top_k)

        with right:
            render_results(results)
