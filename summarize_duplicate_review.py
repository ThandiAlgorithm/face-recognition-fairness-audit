from pathlib import Path

import pandas as pd


path = Path(
    "metadata/near_duplicate_review.csv"
)

if not path.exists():
    raise FileNotFoundError(
        path
    )

df = pd.read_csv(
    path
)

print(
    "Total candidate-pair rows:",
    len(df),
)

print()
print(
    "Overall review status:"
)

print(
    df[
        "review_status"
    ]
    .fillna("candidate")
    .value_counts(
        dropna=False
    )
    .to_string()
)

cross_mask = (
    df["cross_dataset"]
    .astype(str)
    .str.lower()
    .isin(
        [
            "true",
            "1",
            "yes",
        ]
    )
)

cross = df[
    cross_mask
]

within = df[
    ~cross_mask
]

print()
print(
    "Cross-dataset review:"
)

print(
    cross[
        "review_status"
    ]
    .fillna("candidate")
    .value_counts(
        dropna=False
    )
    .to_string()
)

print()
print(
    "Within-dataset review:"
)

print(
    within[
        "review_status"
    ]
    .fillna("candidate")
    .value_counts(
        dropna=False
    )
    .to_string()
)

remaining = (
    df["review_status"]
    .fillna("candidate")
    .eq("candidate")
    .sum()
)

reviewed = (
    len(df)
    - remaining
)

print()
print(
    "Reviewed pairs:",
    reviewed,
)

print(
    "Remaining candidate pairs:",
    remaining,
)

if len(df) > 0:

    percent = (
        reviewed
        / len(df)
        * 100
    )

    print(
        "Progress:",
        f"{percent:.2f}%",
    )
