from pathlib import Path

import numpy as np
import pandas as pd


META = Path("metadata")
AUDIT = META / "detection_integrity_image.parquet"
DIAGNOSTICS = (
    META / "detection_alignment_diagnostics.parquet"
)

SCORES_OUT = META / "detection_scores.csv"
SUMMARY_OUT = META / "confidence_summary.csv"
HIST_OUT = META / "confidence_histogram.csv"

THRESHOLD = 0.90


def is_missing(value):
    return value is None or (
        isinstance(value, float) and np.isnan(value)
    )


def main():
    audit = pd.read_parquet(AUDIT)
    diagnostic = pd.read_parquet(DIAGNOSTICS)

    if len(audit) != 135039 or len(diagnostic) != 135039:
        raise RuntimeError(
            "Expected complete production extraction."
        )

    if (
        not audit["image_id"].is_unique
        or not diagnostic["image_id"].is_unique
    ):
        raise RuntimeError("Duplicate image IDs.")

    # Avoid duplicated fields when joining.
    diagnostic = diagnostic[
        [
            "image_id",
            "detections_all",
        ]
    ]

    df = audit.merge(
        diagnostic,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    raw = df[
        df["route"].eq("project_retinaface")
        & df["detection_status"].eq("face_detected")
    ].copy()

    rows = []

    for record in raw.itertuples(index=False):
        detections = record.detections_all

        if not isinstance(
            detections,
            (list, tuple, np.ndarray),
        ):
            raise RuntimeError(
                f"Missing detections_all: {record.image_id}"
            )

        if len(detections) != int(record.n_detections):
            raise RuntimeError(
                f"Detection-count mismatch: {record.image_id}"
            )

        for index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                raise RuntimeError(
                    f"Malformed detection: {record.image_id}"
                )

            score = float(detection["score"])

            if not np.isfinite(score) or not (
                THRESHOLD <= score <= 1.0
            ):
                raise RuntimeError(
                    "Unexpected retained detection score: "
                    f"{record.image_id}: {score}"
                )

            rows.append({
                "image_id": record.image_id,
                "dataset": record.dataset,
                "score_type": "all_retained",
                "detection_index": index,
                "confidence": score,
            })

        # Primary face is separately recorded.
        primary = record.det_confidence

        if not is_missing(primary):
            primary = float(primary)

            if not np.isfinite(primary) or not (
                THRESHOLD <= primary <= 1.0
            ):
                raise RuntimeError(
                    f"Unexpected primary score: {record.image_id}"
                )

            rows.append({
                "image_id": record.image_id,
                "dataset": record.dataset,
                "score_type": "primary",
                "detection_index": -1,
                "confidence": primary,
            })

    scores = pd.DataFrame(rows)

    if scores.empty:
        raise RuntimeError("No detection scores available.")

    summaries = []

    for (dataset, score_type), part in scores.groupby(
        ["dataset", "score_type"]
    ):
        x = part["confidence"]

        summaries.append({
            "dataset": dataset,
            "score_type": score_type,
            "n_scores": len(x),
            "min": x.min(),
            "p05": x.quantile(0.05),
            "median": x.median(),
            "mean": x.mean(),
            "p95": x.quantile(0.95),
            "max": x.max(),
            "std": x.std(),
        })

    summary = pd.DataFrame(summaries)

    # Fixed 0.01 bins from 0.90 to 1.00.
    # Include an exact score of 1.00 in the last bin.
    edges = np.linspace(0.90, 1.00, 11)
    hist_rows = []

    for (dataset, score_type), part in scores.groupby(
        ["dataset", "score_type"]
    ):
        values = part["confidence"].to_numpy()

        counts, _ = np.histogram(values, bins=edges)

        for i, count in enumerate(counts):
            hist_rows.append({
                "dataset": dataset,
                "score_type": score_type,
                "bin_start": edges[i],
                "bin_end": edges[i + 1],
                "count": int(count),
            })

    histogram = pd.DataFrame(hist_rows)

    # Each histogram must reconcile to the source scores.
    for (dataset, score_type), part in scores.groupby(
        ["dataset", "score_type"]
    ):
        observed = histogram.loc[
            histogram["dataset"].eq(dataset)
            & histogram["score_type"].eq(score_type),
            "count",
        ].sum()

        assert observed == len(part)

    # Check expected number of retained detection records.
    expected_all = int(raw["n_detections"].sum())
    actual_all = int(
        scores["score_type"].eq("all_retained").sum()
    )

    assert expected_all == actual_all

    SCORES_OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores.to_csv(SCORES_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    histogram.to_csv(HIST_OUT, index=False)

    print("\nCONFIDENCE SUMMARY")
    print(summary.to_string(index=False))

    print("\nRetained detection records:", actual_all)
    print("Saved:", SCORES_OUT)
    print("Saved:", SUMMARY_OUT)
    print("Saved:", HIST_OUT)


if __name__ == "__main__":
    main()
