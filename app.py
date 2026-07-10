# ---------------------------------------------------------------------------
# requirements.txt
# ---------------------------------------------------------------------------
# streamlit>=1.32
# tensorflow>=2.15
# scikit-learn>=1.3
# numpy>=1.24
# Pillow>=10.0
# tqdm>=4.66
# ---------------------------------------------------------------------------
"""
app.py
======
Streamlit front-end for the Fashion Recommendation Engine.

Pipeline
--------
1. Load pre-computed image embeddings, file paths, and a fitted KNN model
   (all produced offline by `extract_features.py`).
2. Let the user upload a fashion image (JPG/PNG).
3. Extract a ResNet50 embedding for the uploaded image using the exact same
   preprocessing pipeline used during indexing.
4. Query the KNN model (cosine metric) for the closest matches.
5. Render the top-5 recommended items in a responsive column grid.

Run with:
    streamlit run app.py
"""

import os
import pickle

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# --------------------------------------------------------------------------- #
# Page configuration (must be the first Streamlit call)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Fashion Recommendation Engine",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
ARTIFACTS_DIR = "artifacts"
EMBEDDINGS_PATH = os.path.join(ARTIFACTS_DIR, "embeddings.pkl")
FILENAMES_PATH = os.path.join(ARTIFACTS_DIR, "filenames.pkl")
KNN_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "knn_model.pkl")

IMAGE_SIZE = (224, 224)
NUM_RECOMMENDATIONS = 5
ALLOWED_TYPES = ["jpg", "jpeg", "png"]


# --------------------------------------------------------------------------- #
# Cached resource loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading ResNet50 feature extractor...")
def load_feature_extractor():
    """
    Build the same ResNet50 (no top, global average pooling) feature
    extractor used during offline indexing. Cached as a resource so the
    (relatively heavy) model is loaded only once per session.
    """
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Model

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    return model


@st.cache_resource(show_spinner="Loading recommendation index...")
def load_index_artifacts():
    """
    Load the pre-computed embeddings, file paths, and fitted KNN model.

    Returns
    -------
    tuple(np.ndarray, list, sklearn.neighbors.NearestNeighbors) or None
        Returns None if any artifact is missing, so the caller can show a
        clear, actionable error message instead of crashing.
    """
    missing = [
        p for p in (EMBEDDINGS_PATH, FILENAMES_PATH, KNN_MODEL_PATH) if not os.path.isfile(p)
    ]
    if missing:
        return None, missing

    try:
        with open(EMBEDDINGS_PATH, "rb") as f:
            embeddings = pickle.load(f)
        with open(FILENAMES_PATH, "rb") as f:
            filenames = pickle.load(f)
        with open(KNN_MODEL_PATH, "rb") as f:
            knn_model = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, ValueError) as err:
        st.session_state["_artifact_load_error"] = str(err)
        return None, []

    return (embeddings, filenames, knn_model), []


# --------------------------------------------------------------------------- #
# Feature extraction for the uploaded image
# --------------------------------------------------------------------------- #
def extract_query_features(pil_img: Image.Image, model) -> np.ndarray:
    """
    Convert an uploaded PIL image into an L2-normalized ResNet50 embedding,
    using the identical preprocessing steps applied in extract_features.py.
    """
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image

    img = pil_img.convert("RGB").resize(IMAGE_SIZE)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    features = model.predict(arr, verbose=0)[0]
    norm = np.linalg.norm(features)
    if norm == 0:
        return features
    return features / norm


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(catalog_size: int | None):
    st.sidebar.title("👗 About")
    st.sidebar.markdown(
        """
        This app recommends visually similar fashion items using:

        - **ResNet50** (ImageNet-pretrained CNN) for feature extraction
        - **K-Nearest Neighbors** (cosine similarity) for retrieval

        **How to use**
        1. Upload a photo of a clothing item, shoe, or accessory.
        2. The app extracts a visual embedding of your image.
        3. The 5 closest matches from the catalog are displayed.
        """
    )
    st.sidebar.divider()
    if catalog_size:
        st.sidebar.metric("Indexed catalog items", f"{catalog_size:,}")
    st.sidebar.divider()
    st.sidebar.caption(
        "Tip: for best results, upload a clear, front-facing photo of a "
        "single item on a plain background."
    )


# --------------------------------------------------------------------------- #
# Main application
# --------------------------------------------------------------------------- #
def main():
    st.title("👗 Fashion Recommendation Engine")
    st.markdown(
        "Upload a fashion image and get **visually similar** recommendations, "
        "powered by a ResNet50 CNN and a K-Nearest Neighbors similarity search."
    )
    st.divider()

    # --- Load index artifacts (embeddings, filenames, KNN model) ---
    artifacts, missing_files = load_index_artifacts()

    render_sidebar(len(artifacts[1]) if artifacts else None)

    if artifacts is None:
        if missing_files:
            st.error(
                "⚠️ Could not find the required index files. Please run "
                "`extract_features.py` first to build the recommendation index.\n\n"
                "Missing file(s):\n" + "\n".join(f"- `{p}`" for p in missing_files)
            )
        else:
            load_err = st.session_state.get("_artifact_load_error", "Unknown error.")
            st.error(
                "⚠️ The index files were found but could not be loaded (they may be "
                f"corrupted). Please re-run `extract_features.py`.\n\nDetails: {load_err}"
            )
        st.stop()

    embeddings, filenames, knn_model = artifacts

    # --- Load the CNN feature extractor (cached) ---
    try:
        feature_model = load_feature_extractor()
    except Exception as err:  # noqa: BLE001 - surface any TF/model load issue to the user
        st.error(f"⚠️ Failed to load the ResNet50 model: {err}")
        st.stop()

    # --- Image upload ---
    uploaded_file = st.file_uploader(
        "Upload a fashion image (JPG or PNG)", type=ALLOWED_TYPES
    )

    if uploaded_file is None:
        st.info("👆 Upload an image above to get started.")
        return

    # --- Validate & display the uploaded image ---
    try:
        query_image = Image.open(uploaded_file)
        query_image.verify()  # raises if the file is corrupt / not a real image
        uploaded_file.seek(0)  # reset pointer after verify()
        query_image = Image.open(uploaded_file)
        query_image.load()
    except (UnidentifiedImageError, OSError, ValueError) as err:
        st.error(
            "⚠️ The uploaded file could not be read as a valid image. "
            f"Please try a different JPG or PNG file.\n\nDetails: {err}"
        )
        return

    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.subheader("Your Upload")
        st.image(query_image, use_container_width=True)

    # --- Feature extraction + KNN query ---
    with st.spinner("Analyzing style and finding similar items..."):
        try:
            query_vector = extract_query_features(query_image, feature_model)
            k = min(NUM_RECOMMENDATIONS + 1, len(filenames))  # +1 in case query == itself
            distances, indices = knn_model.kneighbors(
                query_vector.reshape(1, -1), n_neighbors=k
            )
        except Exception as err:  # noqa: BLE001
            st.error(f"⚠️ Something went wrong while generating recommendations: {err}")
            return

    distances, indices = distances[0], indices[0]

    # --- Display recommendations ---
    st.divider()
    st.subheader("✨ Recommended For You")

    results = list(zip(indices, distances))[:NUM_RECOMMENDATIONS]

    if not results:
        st.warning("No similar items were found in the catalog.")
        return

    cols = st.columns(len(results))
    for col, (idx, dist) in zip(cols, results):
        similarity_pct = max(0.0, (1 - dist)) * 100
        img_path = filenames[idx]
        with col:
            try:
                st.image(img_path, use_container_width=True)
            except Exception:
                st.warning(f"Could not load:\n{os.path.basename(img_path)}")
                continue
            st.caption(f"{os.path.basename(img_path)}")
            st.progress(min(1.0, similarity_pct / 100))
            st.caption(f"{similarity_pct:.1f}% similar")


if __name__ == "__main__":
    main()
