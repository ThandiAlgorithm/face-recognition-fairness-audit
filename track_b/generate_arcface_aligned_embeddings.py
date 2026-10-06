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
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAIR_PATH = (
    PROJECT_ROOT
    / "results"
    / "track_b"
    / "arcface_reference"
    / "lfw_view1_pairs_resolved.csv"
)

MASTER_PATH = (
    PROJECT_ROOT
    / "metadata"
    / "master_annotations.parquet"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "track_b"
    / "arcface_reference_aligned"
)

EMBEDDING_ROOT = (
    OUTPUT_ROOT
    / "embeddings"
)

CHUNK_ROOT = (
    EMBEDDING_ROOT
    / "chunks"
)

FINAL_EMBEDDINGS_PATH = (
    EMBEDDING_ROOT
    / "arcface_embeddings.npz"
)

MANIFEST_PATH = (
    EMBEDDING_ROOT
    / "arcface_embedding_manifest.csv"
)

METADATA_PATH = (
    EMBEDDING_ROOT
    / "arcface_embedding_metadata.json"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "ArcFace"

# Detection and alignment are deliberately skipped here because
# the input files are already the frozen project-aligned crops.
DETECTOR_BACKEND = "skip"
DEEPFACE_ALIGN = False
NORMALIZATION = "ArcFace"

EMBEDDING_DIMENSION = 512


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ArcFace embeddings from the frozen "
            "project-aligned LFW crops used by Track B."
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

    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Check pair-to-master mapping and aligned-file "
            "availability without running ArcFace."
        ),
    )

    return parser.parse_args()


# ============================================================
# PATH NORMALISATION
# ============================================================

def normalise_path_text(value) -> str:
    return (
        str(value)
        .strip()
        .replace("\\", "/")
    )


def canonical_lfw_path(value) -> str:
    """
    Map both absolute local paths and master-table relative paths
    to the same canonical LFW key.

    Examples:

      C:/.../data/LFW/lfw/lfw/Aaron/Aaron_0001.jpg

      LFW/lfw/lfw/Aaron/Aaron_0001.jpg

    become:

      lfw/lfw/lfw/aaron/aaron_0001.jpg
    """

    path = normalise_path_text(
        value
    )

    lower = path.lower()

    marker = "/data/lfw/"

    if marker in lower:
        index = lower.index(
            marker
        )

        path = path[
            index + len("/data/"):
        ]

    else:
        parts = path.split("/")

        for i, part in enumerate(
            parts
        ):
            if part.lower() == "lfw":
                path = "/".join(
                    parts[i:]
                )
                break

    return path.lower()


def resolve_project_path(
    value,
) -> Path:
    path = Path(
        str(value)
    )

    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    )


# ============================================================
# LOAD VIEW 1 IMAGE REQUIREMENTS
# ============================================================

def load_view1_images() -> pd.DataFrame:
    if not PAIR_PATH.exists():
        raise FileNotFoundError(
            f"Missing resolved View 1 pair file:\n{PAIR_PATH}\n"
            "Run validate_lfw_view1.py first."
        )

    pairs = pd.read_csv(
        PAIR_PATH,
        low_memory=False,
    )

    required = {
        "image1_path",
        "image2_path",
    }

    missing = (
        required
        - set(
            pairs.columns
        )
    )

    if missing:
        raise KeyError(
            "Pair file missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    raw_paths = pd.concat(
        [
            pairs["image1_path"],
            pairs["image2_path"],
        ],
        ignore_index=True,
    )

    raw_paths = (
        raw_paths
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .reset_index(
            drop=True
        )
    )

    images = pd.DataFrame(
        {
            "raw_image_path": raw_paths,
        }
    )

    images[
        "path_key"
    ] = images[
        "raw_image_path"
    ].map(
        canonical_lfw_path
    )

    return images


# ============================================================
# LOAD MASTER ALIGNMENT MAPPING
# ============================================================

def load_alignment_lookup() -> pd.DataFrame:
    if not MASTER_PATH.exists():
        raise FileNotFoundError(
            f"Missing master table:\n{MASTER_PATH}"
        )

    master = pd.read_parquet(
        MASTER_PATH
    )

    required = {
        "dataset",
        "original_path",
        "aligned_path",
    }

    missing = (
        required
        - set(
            master.columns
        )
    )

    if missing:
        raise KeyError(
            "Master table missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    lfw = master[
        master["dataset"]
        .astype(str)
        .str.lower()
        .eq("lfw")
    ].copy()

    lfw[
        "path_key"
    ] = lfw[
        "original_path"
    ].map(
        canonical_lfw_path
    )

    if lfw[
        "path_key"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate canonical LFW paths found "
            "in master table."
        )

    return lfw[
        [
            "path_key",
            "original_path",
            "aligned_path",
        ]
    ].copy()


# ============================================================
# BUILD OFFICIAL INPUT MANIFEST
# ============================================================

def build_manifest() -> pd.DataFrame:
    images = load_view1_images()

    lookup = load_alignment_lookup()

    manifest = images.merge(
        lookup,
        on="path_key",
        how="left",
        validate="one_to_one",
    )

    manifest[
        "mapped_to_master"
    ] = manifest[
        "original_path"
    ].notna()

    manifest[
        "aligned_path_available"
    ] = manifest[
        "aligned_path"
    ].notna()

    def resolve_aligned(
        value,
    ):
        if pd.isna(
            value
        ):
            return None

        return str(
            resolve_project_path(
                value
            )
        )

    manifest[
        "aligned_image_path"
    ] = manifest[
        "aligned_path"
    ].map(
        resolve_aligned
    )

    manifest[
        "aligned_file_exists"
    ] = manifest[
        "aligned_image_path"
    ].map(
        lambda value: (
            False
            if value is None
            else Path(
                value
            ).exists()
        )
    )

    manifest.insert(
        0,
        "embedding_index",
        np.arange(
            len(manifest),
            dtype=int,
        ),
    )

    return manifest


# ============================================================
# REPORT INPUT STATUS
# ============================================================

def print_input_status(
    manifest: pd.DataFrame,
) -> None:
    total = len(
        manifest
    )

    mapped = int(
        manifest[
            "mapped_to_master"
        ].sum()
    )

    has_aligned_path = int(
        manifest[
            "aligned_path_available"
        ].sum()
    )

    existing = int(
        manifest[
            "aligned_file_exists"
        ].sum()
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "TRACK B ALIGNED INPUT CHECK"
    )

    print(
        "=" * 72
    )

    print(
        f"Unique View 1 images required: {total:,}"
    )

    print(
        f"Mapped to LFW master table:     {mapped:,}/{total:,}"
    )

    print(
        f"Aligned paths recorded:         "
        f"{has_aligned_path:,}/{total:,}"
    )

    print(
        f"Aligned files existing locally: "
        f"{existing:,}/{total:,}"
    )

    unresolved = manifest[
        ~manifest[
            "mapped_to_master"
        ]
    ]

    if not unresolved.empty:
        print(
            "\nUnresolved master-table paths:"
        )

        print(
            unresolved[
                [
                    "raw_image_path",
                    "path_key",
                ]
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    missing_files = manifest[
        manifest[
            "aligned_path_available"
        ]
        & ~manifest[
            "aligned_file_exists"
        ]
    ]

    if not missing_files.empty:
        print(
            "\nExample aligned files currently missing:"
        )

        print(
            missing_files[
                [
                    "raw_image_path",
                    "aligned_path",
                ]
            ]
            .head(5)
            .to_string(
                index=False
            )
        )


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

    if norm <= 0:
        raise ValueError(
            "ArcFace returned a zero-norm embedding."
        )

    return (
        vector / norm
    ).astype(
        np.float32
    )


def extract_single_embedding(
    result_item,
) -> np.ndarray:
    """
    DeepFace 0.0.101 can return either:

        {"embedding": [...]}

    or:

        [{"embedding": [...]}]

    for each item in a batch.
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
        if len(
            result_item
        ) != 1:
            raise RuntimeError(
                "Expected exactly one embedding result "
                "for each already-aligned face crop, but "
                f"DeepFace returned {len(result_item)}."
            )

        face_result = result_item[0]

        if not isinstance(
            face_result,
            dict,
        ):
            raise TypeError(
                "Unexpected nested DeepFace result type: "
                f"{type(face_result)}"
            )

    else:
        raise TypeError(
            "Unexpected DeepFace result type: "
            f"{type(result_item)}"
        )

    if "embedding" not in face_result:
        raise KeyError(
            "DeepFace result does not contain 'embedding'."
        )

    embedding = np.asarray(
        face_result[
            "embedding"
        ],
        dtype=np.float32,
    )

    if embedding.shape != (
        EMBEDDING_DIMENSION,
    ):
        raise ValueError(
            "Unexpected ArcFace embedding shape: "
            f"{embedding.shape}"
        )

    return l2_normalize(
        embedding
    )


def run_batch(
    aligned_paths: list[str],
) -> np.ndarray:
    result = DeepFace.represent(
        img_path=aligned_paths,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
        align=DEEPFACE_ALIGN,
        normalization=NORMALIZATION,
    )

    if not isinstance(
        result,
        list,
    ):
        raise TypeError(
            "DeepFace batch response was not a list."
        )

    if len(
        result
    ) != len(
        aligned_paths
    ):
        raise RuntimeError(
            "DeepFace batch length mismatch. "
            f"Requested={len(aligned_paths)}, "
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
# CHUNK CACHE
# ============================================================

def chunk_path(
    start: int,
    end: int,
) -> Path:
    return (
        CHUNK_ROOT
        / (
            f"arcface_aligned_"
            f"{start:05d}_"
            f"{end:05d}.npz"
        )
    )


def generate_chunks(
    manifest: pd.DataFrame,
    batch_size: int,
) -> None:
    CHUNK_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = len(
        manifest
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ARCFACE — PROJECT-ALIGNED LFW"
    )

    print(
        "=" * 72
    )

    print(
        f"Images: {total:,}"
    )

    print(
        f"Batch size: {batch_size}"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        "Input: frozen project-aligned LFW crops"
    )

    print(
        f"DeepFace detector: {DETECTOR_BACKEND}"
    )

    print(
        f"DeepFace alignment: {DEEPFACE_ALIGN}"
    )

    print(
        f"Normalization: {NORMALIZATION}"
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

        batch = manifest.iloc[
            start:end
        ]

        paths = (
            batch[
                "aligned_image_path"
            ]
            .astype(str)
            .tolist()
        )

        print(
            f"[{start + 1:>5}-{end:>5}"
            f"/{total}] embedding..."
        )

        embeddings = run_batch(
            paths
        )

        np.savez_compressed(
            output,
            embeddings=embeddings,
            aligned_paths=np.asarray(
                paths,
                dtype=str,
            ),
        )

        print(
            f"          saved {output.name}"
        )


# ============================================================
# CONSOLIDATE
# ============================================================

def consolidate(
    manifest: pd.DataFrame,
    batch_size: int,
) -> None:
    total = len(
        manifest
    )

    matrices = []
    paths = []

    for start in range(
        0,
        total,
        batch_size,
    ):
        end = min(
            start + batch_size,
            total,
        )

        file = chunk_path(
            start,
            end - 1,
        )

        if not file.exists():
            raise FileNotFoundError(
                f"Missing chunk:\n{file}"
            )

        with np.load(
            file,
            allow_pickle=False,
        ) as data:
            matrices.append(
                data[
                    "embeddings"
                ].astype(
                    np.float32
                )
            )

            paths.extend(
                data[
                    "aligned_paths"
                ]
                .astype(str)
                .tolist()
            )

    matrix = np.concatenate(
        matrices,
        axis=0,
    )

    expected_shape = (
        total,
        EMBEDDING_DIMENSION,
    )

    if matrix.shape != expected_shape:
        raise RuntimeError(
            "Unexpected final embedding shape. "
            f"Expected={expected_shape}, "
            f"received={matrix.shape}"
        )

    expected_paths = (
        manifest[
            "aligned_image_path"
        ]
        .astype(str)
        .tolist()
    )

    if paths != expected_paths:
        raise RuntimeError(
            "Embedding path order does not match manifest."
        )

    norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Final ArcFace embeddings are not "
            "L2-normalized."
        )

    np.savez_compressed(
        FINAL_EMBEDDINGS_PATH,
        embeddings=matrix,
        aligned_paths=np.asarray(
            paths,
            dtype=str,
        ),
        raw_image_paths=np.asarray(
            manifest[
                "raw_image_path"
            ]
            .astype(str)
            .tolist(),
            dtype=str,
        ),
    )

    output_manifest = (
        manifest.copy()
    )

    output_manifest[
        "model_name"
    ] = MODEL_NAME

    output_manifest[
        "embedding_dimension"
    ] = EMBEDDING_DIMENSION

    output_manifest[
        "embedding_l2_normalized"
    ] = True

    output_manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    metadata = {
        "model_id": "R0_ArcFace",
        "role": "pretrained_reference_recognizer",
        "input_source": (
            "frozen project-aligned LFW crops"
        ),
        "master_table": str(
            MASTER_PATH.relative_to(
                PROJECT_ROOT
            )
        ),
        "model_name": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "deepface_detector_backend": (
            DETECTOR_BACKEND
        ),
        "deepface_alignment": (
            DEEPFACE_ALIGN
        ),
        "normalization": NORMALIZATION,
        "embedding_l2_normalized": True,
        "n_unique_images": total,
        "protocol_scope": (
            "LFW View 1 development protocol"
        ),
        "view2_used": False,
        "raw_input_reference_run": (
            "excluded from final reporting"
        ),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ALIGNED ARCFACE EMBEDDING CACHE COMPLETE"
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
            "--batch-size must be >= 1."
        )

    manifest = (
        build_manifest()
    )

    print_input_status(
        manifest
    )

    total = len(
        manifest
    )

    if total != 4992:
        raise RuntimeError(
            "Expected exactly 4,992 unique View 1 images, "
            f"but found {total:,}."
        )

    if not manifest[
        "mapped_to_master"
    ].all():
        raise RuntimeError(
            "Not all View 1 images map to the "
            "LFW master table."
        )

    if not manifest[
        "aligned_path_available"
    ].all():
        raise RuntimeError(
            "Some View 1 images do not have aligned_path "
            "values in the master table."
        )

    if args.check_only:
        print(
            "\nCHECK-ONLY MODE COMPLETE"
        )

        if manifest[
            "aligned_file_exists"
        ].all():
            print(
                "All required aligned files are available."
            )
        else:
            missing = int(
                (
                    ~manifest[
                        "aligned_file_exists"
                    ]
                ).sum()
            )

            print(
                f"{missing:,} aligned files are currently "
                "missing locally."
            )

            print(
                "This is expected while waiting for "
                "Tandy's aligned-image folder."
            )

        return

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError(
                "--limit must be >= 1."
            )

        manifest = (
            manifest
            .head(
                args.limit
            )
            .copy()
            .reset_index(
                drop=True
            )
        )

        manifest[
            "embedding_index"
        ] = np.arange(
            len(manifest),
            dtype=int,
        )

        print(
            "\nSMOKE TEST MODE"
        )

        print(
            f"Using first {len(manifest)} "
            "aligned images."
        )

    if not manifest[
        "aligned_file_exists"
    ].all():
        missing = int(
            (
                ~manifest[
                    "aligned_file_exists"
                ]
            ).sum()
        )

        raise FileNotFoundError(
            f"{missing:,} selected aligned LFW files "
            "are missing locally. Do not run ArcFace "
            "until the aligned-image folder is available."
        )

    EMBEDDING_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    generate_chunks(
        manifest=manifest,
        batch_size=args.batch_size,
    )

    consolidate(
        manifest=manifest,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
    