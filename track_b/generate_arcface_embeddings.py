from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
from deepface import DeepFace


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "track_b"
    / "arcface_reference"
)

PAIR_PATH = (
    RESULT_ROOT
    / "lfw_view1_pairs_resolved.csv"
)

EMBEDDING_ROOT = (
    RESULT_ROOT
    / "embeddings"
)

CHUNK_ROOT = (
    EMBEDDING_ROOT
    / "chunks"
)

MANIFEST_PATH = (
    EMBEDDING_ROOT
    / "arcface_embedding_manifest.csv"
)

FINAL_EMBEDDINGS_PATH = (
    EMBEDDING_ROOT
    / "arcface_embeddings.npz"
)

METADATA_PATH = (
    EMBEDDING_ROOT
    / "arcface_embedding_metadata.json"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "skip"
ALIGN = False
NORMALIZATION = "ArcFace"
EMBEDDING_DIMENSION = 512


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate and cache ArcFace embeddings for "
            "all unique LFW View 1 images."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Images per DeepFace batch.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test image limit.",
    )

    return parser.parse_args()


# ============================================================
# LOAD UNIQUE VIEW 1 IMAGES
# ============================================================

def load_unique_images() -> pd.DataFrame:
    if not PAIR_PATH.exists():
        raise FileNotFoundError(
            f"Missing resolved pair table:\n{PAIR_PATH}\n"
            "Run validate_lfw_view1.py first."
        )

    pairs = pd.read_csv(
        PAIR_PATH,
        low_memory=False,
    )

    paths = pd.concat(
        [
            pairs["image1_path"],
            pairs["image2_path"],
        ],
        ignore_index=True,
    )

    paths = (
        paths
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    images = pd.DataFrame(
        {
            "embedding_index": np.arange(
                len(paths),
                dtype=int,
            ),
            "image_path": paths,
        }
    )

    images["image_exists"] = images[
        "image_path"
    ].map(
        lambda value: Path(value).exists()
    )

    missing = images[
        ~images["image_exists"]
    ]

    if not missing.empty:
        raise FileNotFoundError(
            "Required images are missing:\n"
            + "\n".join(
                missing["image_path"]
                .head(20)
                .tolist()
            )
        )

    return images


# ============================================================
# EMBEDDING HELPERS
# ============================================================

def l2_normalize(
    vector: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    norm = np.linalg.norm(
        vector
    )

    if norm == 0:
        raise ValueError(
            "ArcFace returned a zero-norm embedding."
        )

    return (
        vector / norm
    ).astype(
        np.float32
    )


def chunk_path(
    start_index: int,
    end_index: int,
) -> Path:
    return (
        CHUNK_ROOT
        / (
            f"arcface_"
            f"{start_index:05d}_"
            f"{end_index:05d}.npz"
        )
    )


def extract_single_embedding(
    result_item,
) -> np.ndarray:
    """
    DeepFace 0.0.101 can return either:

        {"embedding": [...]}

    or, when using a list of images:

        [{"embedding": [...]}]

    This helper accepts both.
    """

    if isinstance(
        result_item,
        dict,
    ):
        face_result = result_item

    elif isinstance(
        result_item,
        list,
    ):
        if len(result_item) != 1:
            raise RuntimeError(
                "Expected exactly one face result "
                "per LFW crop, but DeepFace returned "
                f"{len(result_item)} results."
            )

        face_result = result_item[0]

        if not isinstance(
            face_result,
            dict,
        ):
            raise TypeError(
                "Unexpected DeepFace nested result type: "
                f"{type(face_result)}"
            )

    else:
        raise TypeError(
            "Unexpected DeepFace result type: "
            f"{type(result_item)}"
        )

    if "embedding" not in face_result:
        raise KeyError(
            "DeepFace result does not contain "
            "'embedding'."
        )

    vector = np.asarray(
        face_result["embedding"],
        dtype=np.float32,
    )

    if vector.shape != (
        EMBEDDING_DIMENSION,
    ):
        raise ValueError(
            "Unexpected ArcFace embedding shape: "
            f"{vector.shape}"
        )

    return l2_normalize(
        vector
    )


def run_batch(
    batch_paths: list[str],
) -> np.ndarray:
    result = DeepFace.represent(
        img_path=batch_paths,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
        align=ALIGN,
        normalization=NORMALIZATION,
    )

    if not isinstance(
        result,
        list,
    ):
        raise TypeError(
            "DeepFace batch response was not a list. "
            f"Received: {type(result)}"
        )

    if len(result) != len(
        batch_paths
    ):
        raise RuntimeError(
            "DeepFace returned a different number "
            "of image results than requested. "
            f"Requested={len(batch_paths)}, "
            f"returned={len(result)}."
        )

    embeddings = [
        extract_single_embedding(
            item
        )
        for item in result
    ]

    return np.stack(
        embeddings,
        axis=0,
    )


# ============================================================
# GENERATE CHUNKS
# ============================================================

def generate_chunks(
    images: pd.DataFrame,
    batch_size: int,
) -> None:
    CHUNK_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = len(images)

    print(
        "\n"
        + "=" * 72
    )
    print(
        "ARCFACE EMBEDDING GENERATION"
    )
    print(
        "=" * 72
    )

    print(
        f"Images to process: {total:,}"
    )
    print(
        f"Batch size: {batch_size}"
    )
    print(
        f"Model: {MODEL_NAME}"
    )
    print(
        f"Detector: {DETECTOR_BACKEND}"
    )
    print(
        f"Alignment inside DeepFace: {ALIGN}"
    )
    print(
        f"Input normalization: {NORMALIZATION}"
    )

    for start in range(
        0,
        total,
        batch_size,
    ):
        end = min(
            start + batch_size,
            total,
        )

        output = chunk_path(
            start,
            end - 1,
        )

        if output.exists():
            print(
                f"[{end:>5}/{total}] "
                "cached — skipping"
            )
            continue

        batch = images.iloc[
            start:end
        ]

        batch_paths = (
            batch["image_path"]
            .astype(str)
            .tolist()
        )

        print(
            f"[{start + 1:>5}-{end:>5}"
            f"/{total}] embedding..."
        )

        embeddings = run_batch(
            batch_paths
        )

        np.savez_compressed(
            output,
            embeddings=embeddings,
            image_paths=np.asarray(
                batch_paths,
                dtype=str,
            ),
        )

        print(
            f"          saved {output.name}"
        )


# ============================================================
# CONSOLIDATE
# ============================================================

def consolidate_chunks(
    images: pd.DataFrame,
    batch_size: int,
) -> None:
    all_embeddings = []
    all_paths = []

    total = len(images)

    for start in range(
        0,
        total,
        batch_size,
    ):
        end = min(
            start + batch_size,
            total,
        )

        path = chunk_path(
            start,
            end - 1,
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing embedding chunk:\n{path}"
            )

        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            embeddings = data[
                "embeddings"
            ].astype(
                np.float32
            )

            image_paths = data[
                "image_paths"
            ].astype(
                str
            )

        all_embeddings.append(
            embeddings
        )

        all_paths.extend(
            image_paths.tolist()
        )

    matrix = np.concatenate(
        all_embeddings,
        axis=0,
    )

    expected_paths = (
        images["image_path"]
        .astype(str)
        .tolist()
    )

    if all_paths != expected_paths:
        raise RuntimeError(
            "Embedding path order does not match "
            "manifest order."
        )

    expected_shape = (
        len(images),
        EMBEDDING_DIMENSION,
    )

    if matrix.shape != expected_shape:
        raise RuntimeError(
            "Unexpected final embedding matrix shape. "
            f"Expected={expected_shape}, "
            f"received={matrix.shape}"
        )

    np.savez_compressed(
        FINAL_EMBEDDINGS_PATH,
        embeddings=matrix,
        image_paths=np.asarray(
            all_paths,
            dtype=str,
        ),
    )

    manifest = images.copy()

    manifest[
        "embedding_dimension"
    ] = EMBEDDING_DIMENSION

    manifest[
        "model_name"
    ] = MODEL_NAME

    manifest[
        "l2_normalized"
    ] = True

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    metadata = {
        "model_name": MODEL_NAME,
        "embedding_dimension": (
            EMBEDDING_DIMENSION
        ),
        "detector_backend": (
            DETECTOR_BACKEND
        ),
        "deepface_alignment": ALIGN,
        "input_normalization": (
            NORMALIZATION
        ),
        "embedding_l2_normalized": True,
        "n_unique_images": int(
            len(images)
        ),
        "pair_source": str(
            PAIR_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "protocol_scope": (
            "LFW View 1 development only"
        ),
        "view2_used": False,
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    print(
        "\n"
        + "=" * 72
    )
    print(
        "EMBEDDING CACHE COMPLETE"
    )
    print(
        "=" * 72
    )

    print(
        "Embedding matrix shape:",
        matrix.shape,
    )
    print(
        "Mean L2 norm:",
        f"{norms.mean():.6f}",
    )
    print(
        "Minimum L2 norm:",
        f"{norms.min():.6f}",
    )
    print(
        "Maximum L2 norm:",
        f"{norms.max():.6f}",
    )

    print(
        "\nSaved:"
    )
    print(
        FINAL_EMBEDDINGS_PATH
    )
    print(
        MANIFEST_PATH
    )
    print(
        METADATA_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1"
        )

    EMBEDDING_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = load_unique_images()

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError(
                "--limit must be >= 1"
            )

        images = (
            images
            .head(
                args.limit
            )
            .copy()
            .reset_index(
                drop=True
            )
        )

        images[
            "embedding_index"
        ] = np.arange(
            len(images),
            dtype=int,
        )

        print(
            "\nSMOKE TEST MODE"
        )
        print(
            f"Only the first "
            f"{len(images)} images "
            "will be processed."
        )

    generate_chunks(
        images=images,
        batch_size=args.batch_size,
    )

    consolidate_chunks(
        images=images,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
    