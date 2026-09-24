from pathlib import Path
import hashlib

import pandas as pd


PAIRS = Path("metadata/lfw_protocol/pairs.txt")
IMAGE_ROOT = Path("data/LFW/lfw/lfw")
MASTER = Path("metadata/master_annotations.parquet")
OUTPUT = Path("metadata/lfw_view2_pairs.parquet")

EXPECTED_SHA256 = (
    "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"
)


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def canonical_path(person, number):
    filename = f"{person}_{int(number):04d}.jpg"

    return (
        f"LFW/lfw/lfw/"
        f"{person}/"
        f"{filename}"
    )


def disk_path(person, number):
    filename = f"{person}_{int(number):04d}.jpg"

    return (
        IMAGE_ROOT
        / person
        / filename
    )


def main():

    if not PAIRS.exists():
        raise RuntimeError(
            f"Missing protocol file: {PAIRS}"
        )

    digest = sha256_file(PAIRS)

    print("pairs.txt SHA256:")
    print(digest)

    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            "pairs.txt checksum does not match expected official file"
        )

    lines = [
        line.strip()
        for line in PAIRS.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    header = lines[0].split()

    if header != ["10", "300"]:
        raise RuntimeError(
            f"Unexpected pairs.txt header: {header}"
        )

    pair_lines = lines[1:]

    expected_pairs = 10 * 600

    if len(pair_lines) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} pairs, "
            f"found {len(pair_lines)}"
        )

    rows = []
    missing_disk = []

    for i, line in enumerate(pair_lines):

        fold = (i // 600) + 1
        position = i % 600

        parts = line.split()

        if len(parts) == 3:
            # Same person.
            person, n1, n2 = parts

            pa = canonical_path(person, n1)
            pb = canonical_path(person, n2)

            da = disk_path(person, n1)
            db = disk_path(person, n2)

            is_same = True

        elif len(parts) == 4:
            # Different people.
            person1, n1, person2, n2 = parts

            pa = canonical_path(person1, n1)
            pb = canonical_path(person2, n2)

            da = disk_path(person1, n1)
            db = disk_path(person2, n2)

            is_same = False

        else:
            raise RuntimeError(
                f"Malformed pair line {i + 2}: {line}"
            )

        if not da.exists():
            missing_disk.append(str(da))

        if not db.exists():
            missing_disk.append(str(db))

        rows.append(
            {
                "fold": fold,
                "pair_index": i,
                "fold_position": position,
                "is_same": is_same,
                "original_path_a": pa,
                "original_path_b": pb,
            }
        )

    pairs = pd.DataFrame(rows)

    print("\nPairs:", len(pairs))
    print("Same:", int(pairs["is_same"].sum()))
    print("Different:", int((~pairs["is_same"]).sum()))

    print("\nPer fold:")
    print(
        pairs.groupby(
            ["fold", "is_same"]
        )
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print(
        "\nMissing image files:",
        len(set(missing_disk)),
    )

    if missing_disk:
        print("\nFirst missing files:")
        for p in sorted(set(missing_disk))[:20]:
            print(p)

        raise RuntimeError(
            "Some LFW protocol images are missing"
        )

    # Check every protocol path against the canonical master

    master = pd.read_parquet(
        MASTER,
        columns=[
            "image_id",
            "dataset",
            "original_path",
        ],
    )

    lfw = master[
        master["dataset"]
        .astype(str)
        .str.lower()
        .eq("lfw")
    ].copy()

    if len(lfw) != 13233:
        raise RuntimeError(
            f"Expected 13233 LFW master rows, got {len(lfw)}"
        )

    path_to_id = dict(
        zip(
            lfw["original_path"],
            lfw["image_id"],
        )
    )

    all_protocol_paths = (
        set(pairs["original_path_a"])
        | set(pairs["original_path_b"])
    )

    missing_master = sorted(
        all_protocol_paths
        - set(path_to_id)
    )

    print(
        "Unique protocol images:",
        len(all_protocol_paths),
    )

    print(
        "Protocol images missing from master:",
        len(missing_master),
    )

    if missing_master:
        print("\nFirst missing master paths:")
        for p in missing_master[:20]:
            print(p)

        raise RuntimeError(
            "Protocol paths do not fully map to master table"
        )

    pairs["image_id_a"] = (
        pairs["original_path_a"]
        .map(path_to_id)
    )

    pairs["image_id_b"] = (
        pairs["original_path_b"]
        .map(path_to_id)
    )

    if pairs[
        ["image_id_a", "image_id_b"]
    ].isna().any().any():
        raise RuntimeError(
            "Null image IDs after protocol mapping"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs.to_parquet(
        OUTPUT,
        index=False,
    )

    print("\n" + "=" * 60)
    print("LFW VIEW 2 VALIDATION PASSED")
    print("=" * 60)

    print("Folds: 10")
    print("Pairs per fold: 600")
    print("Matched per fold: 300")
    print("Mismatched per fold: 300")
    print("Total pairs: 6000")
    print("Master LFW rows: 13233")
    print("Missing protocol images: 0")
    print("Missing master mappings: 0")
    print()
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
