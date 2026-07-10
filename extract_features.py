"""
extract_features.py

Offline indexing script for the Fashion Recommendation Engine.

WHAT THIS SCRIPT DOES
----------------------
1. Walks a local folder of fashion product images.
2. Loads a pre-trained ResNet50 (ImageNet weights, top classification layer
   removed, Global Average Pooling added) to convert every image into a
   2048-dimensional embedding vector.
3. Fits a scikit-learn `NearestNeighbors` model (cosine metric) on all
   embeddings.
4. Persists three artifacts to disk with `pickle`, which `app.py` later
   loads:
       - embeddings.pkl   -> numpy array of shape (N, 2048)
       - filenames.pkl    -> list[str] of image paths, aligned with embeddings
       - knn_model.pkl    -> fitted sklearn NearestNeighbors instance

USAGE
-----
    python extract_features.py --dataset ./data/images --output ./artifacts

REQUIREMENTS
------------
See requirements.txt in the project root.
"""

import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# TensorFlow / Keras imports are wrapped so we can give a friendly error
# message if the library is not installed, rather than a raw traceback.
try:
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing import image as keras_image
except ImportError as exc:  # pragma: no cover
    print(
        "ERROR: TensorFlow is required for feature extraction.\n"
        "Install it with: pip install tensorflow\n"
        f"Original error: {exc}"
    )
    sys.exit(1)

# --------------------------------------------------------------------------- #
# Configuration constants
# --------------------------------------------------------------------------- #
IMAGE_SIZE = (224, 224)          # ResNet50 default input size
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
N_NEIGHBORS = 6                  # 5 recommendations + the query image itself
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("extract_features")


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_feature_extractor() -> Model:
    """
    Build a ResNet50-based feature extractor.

    The top (classification) layer of ResNet50 is removed. A
    GlobalAveragePooling2D layer condenses the final (7, 7, 2048)
    convolutional feature map into a flat (2048,) vector per image.

    Returns
    -------
    tf.keras.Model
        A model that maps a preprocessed image batch -> (batch, 2048) array.
    """
    logger.info("Loading ResNet50 backbone (ImageNet weights)...")
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
        pooling=None,
    )
    base_model.trainable = False  # Freeze weights; we only need inference.

    x = GlobalAveragePooling2D()(base_model.output)
    feature_extractor = Model(inputs=base_model.input, outputs=x, name="resnet50_feature_extractor")
    logger.info("Feature extractor ready. Output dimension: %d", feature_extractor.output_shape[-1])
    return feature_extractor


# --------------------------------------------------------------------------- #
# Feature extraction helpers
# --------------------------------------------------------------------------- #
def preprocess_pil_image(pil_img: Image.Image) -> np.ndarray:
    """
    Convert a PIL image into a ResNet50-ready numpy array.

    Steps: RGB conversion -> resize -> array conversion -> batch dimension
    -> ResNet50 preprocessing (BGR conversion + ImageNet mean subtraction).
    """
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize(IMAGE_SIZE)
    img_array = keras_image.img_to_array(pil_img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def extract_feature_vector(pil_img: Image.Image, model: Model) -> np.ndarray:
    """
    Run a single image through the feature extractor and L2-normalize
    the resulting embedding. L2 normalization makes cosine similarity
    equivalent to a simple dot product, which is numerically stable and
    is standard practice for image-retrieval embeddings.
    """
    preprocessed = preprocess_pil_image(pil_img)
    raw_features = model.predict(preprocessed, verbose=0)[0]
    norm = np.linalg.norm(raw_features)
    if norm == 0:
        return raw_features
    return raw_features / norm


def collect_image_paths(dataset_dir: str) -> list:
    """Recursively collect all valid image file paths under dataset_dir."""
    image_paths = []
    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if fname.lower().endswith(VALID_EXTENSIONS):
                image_paths.append(os.path.join(root, fname))
    return sorted(image_paths)


# --------------------------------------------------------------------------- #
# Main indexing pipeline
# --------------------------------------------------------------------------- #
def build_index(dataset_dir: str, output_dir: str) -> None:
    if not os.path.isdir(dataset_dir):
        logger.error("Dataset directory not found: %s", dataset_dir)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    image_paths = collect_image_paths(dataset_dir)
    if not image_paths:
        logger.error(
            "No valid images (%s) found in %s", ", ".join(VALID_EXTENSIONS), dataset_dir
        )
        sys.exit(1)

    logger.info("Found %d candidate images.", len(image_paths))

    model = build_feature_extractor()

    features = []
    valid_paths = []
    skipped = 0

    start = time.time()
    for path in tqdm(image_paths, desc="Extracting features"):
        try:
            with Image.open(path) as img:
                vector = extract_feature_vector(img, model)
            features.append(vector)
            valid_paths.append(path)
        except (UnidentifiedImageError, OSError, ValueError) as err:
            # Gracefully skip corrupt / unreadable images instead of crashing
            # the whole indexing run.
            logger.warning("Skipping unreadable image '%s': %s", path, err)
            skipped += 1
            continue

    if not features:
        logger.error("No images could be processed successfully. Aborting.")
        sys.exit(1)

    elapsed = time.time() - start
    logger.info(
        "Extracted features for %d/%d images in %.1fs (%d skipped).",
        len(features), len(image_paths), elapsed, skipped,
    )

    features_array = np.array(features, dtype="float32")

    # Fit KNN with cosine similarity metric. n_neighbors is capped at the
    # dataset size so this also works on tiny demo datasets.
    n_neighbors = min(N_NEIGHBORS, len(features_array))
    logger.info("Fitting NearestNeighbors (metric=cosine, k=%d)...", n_neighbors)
    knn_model = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm="brute",   # exact search; required for the cosine metric
        metric="cosine",
    )
    knn_model.fit(features_array)

    # Persist artifacts
    features_path = os.path.join(output_dir, "embeddings.pkl")
    filenames_path = os.path.join(output_dir, "filenames.pkl")
    knn_path = os.path.join(output_dir, "knn_model.pkl")

    with open(features_path, "wb") as f:
        pickle.dump(features_array, f)
    with open(filenames_path, "wb") as f:
        pickle.dump(valid_paths, f)
    with open(knn_path, "wb") as f:
        pickle.dump(knn_model, f)

    logger.info("Saved:")
    logger.info("  - %s  (shape=%s)", features_path, features_array.shape)
    logger.info("  - %s  (%d paths)", filenames_path, len(valid_paths))
    logger.info("  - %s", knn_path)
    logger.info("Indexing complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ResNet50 features from a fashion image dataset and build a KNN index."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/images",
        help="Path to the folder containing fashion images (searched recursively).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./artifacts",
        help="Folder where embeddings.pkl, filenames.pkl and knn_model.pkl will be saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(args.dataset, args.output)
