from __future__ import annotations

from pathlib import Path

import pandas as pd


def image_path(image_root: Path, name: str, image_number: int) -> Path:
    return image_root / str(name) / f"{name}_{int(image_number):04d}.jpg"


def _data_columns(dataframe: pd.DataFrame) -> list[str]:
    return [c for c in dataframe.columns if c != "metadata_source"]


def normalize_match_table(dataframe: pd.DataFrame, source_name: str) -> pd.DataFrame:
    columns = _data_columns(dataframe)
    if len(columns) < 3:
        raise ValueError(f"{source_name} needs at least three columns.")

    rows = []
    for row_index, row in dataframe.iterrows():
        name = str(row[columns[0]])
        image_one = int(row[columns[1]])
        image_two = int(row[columns[2]])
        rows.append(
            {
                "source": source_name,
                "source_row": int(row_index),
                "pair_type": "genuine",
                "name_1": name,
                "image_num_1": image_one,
                "name_2": name,
                "image_num_2": image_two,
            }
        )
    return pd.DataFrame(rows)


def normalize_mismatch_table(dataframe: pd.DataFrame, source_name: str) -> pd.DataFrame:
    columns = _data_columns(dataframe)
    if len(columns) < 4:
        raise ValueError(f"{source_name} needs at least four columns.")

    rows = []
    for row_index, row in dataframe.iterrows():
        rows.append(
            {
                "source": source_name,
                "source_row": int(row_index),
                "pair_type": "impostor",
                "name_1": str(row[columns[0]]),
                "image_num_1": int(row[columns[1]]),
                "name_2": str(row[columns[2]]),
                "image_num_2": int(row[columns[3]]),
            }
        )
    return pd.DataFrame(rows)


def normalize_pairs_table(dataframe: pd.DataFrame, source_name: str = "pairs.csv") -> pd.DataFrame:
    columns = _data_columns(dataframe)
    if len(columns) < 4:
        raise ValueError(f"{source_name} needs four data columns.")

    rows = []
    for row_index, row in dataframe.iterrows():
        first_name = row[columns[0]]
        first_number = row[columns[1]]
        third_value = row[columns[2]]
        fourth_value = row[columns[3]]

        if pd.isna(fourth_value):
            pair_type = "genuine"
            second_name = first_name
            second_number = third_value
        else:
            pair_type = "impostor"
            second_name = third_value
            second_number = fourth_value

        rows.append(
            {
                "source": source_name,
                "source_row": int(row_index),
                "fold": int(row_index // 600) + 1,
                "pair_type": pair_type,
                "name_1": str(first_name),
                "image_num_1": int(first_number),
                "name_2": str(second_name),
                "image_num_2": int(second_number),
            }
        )
    return pd.DataFrame(rows)


def validate_pair_table(pair_table: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    output = pair_table.copy()

    output["path_1"] = output.apply(lambda r: str(image_path(image_root, r["name_1"], r["image_num_1"])), axis=1)
    output["path_2"] = output.apply(lambda r: str(image_path(image_root, r["name_2"], r["image_num_2"])), axis=1)

    output["path_1_exists"] = output["path_1"].apply(lambda p: Path(p).exists())
    output["path_2_exists"] = output["path_2"].apply(lambda p: Path(p).exists())

    genuine_valid = output["pair_type"].eq("genuine") & output["name_1"].eq(output["name_2"])
    impostor_valid = output["pair_type"].eq("impostor") & output["name_1"].ne(output["name_2"])
    output["identity_relation_valid"] = genuine_valid | impostor_valid

    output["pair_valid"] = output["path_1_exists"] & output["path_2_exists"] & output["identity_relation_valid"]

    return output


def validate_people_table(people: pd.DataFrame, image_root: Path) -> pd.DataFrame:
    columns = _data_columns(people)
    if len(columns) < 2:
        raise ValueError("people table needs name and image count columns.")

    name_column = columns[0]
    count_column = columns[1]
    rows = []

    for _, row in people.iterrows():
        if pd.isna(row[name_column]):
            continue

        name = str(row[name_column])
        declared = int(row[count_column])
        folder = image_root / name
        actual = len(list(folder.glob("*.jpg"))) if folder.exists() else 0

        rows.append(
            {
                "name": name,
                "declared_images": declared,
                "actual_images": actual,
                "folder_exists": folder.exists(),
                "count_matches": declared == actual,
            }
        )

    return pd.DataFrame(rows)


def protocol_summary(validated_pairs: pd.DataFrame) -> dict:
    return {
        "rows": int(len(validated_pairs)),
        "genuine_pairs": int(validated_pairs["pair_type"].eq("genuine").sum()),
        "impostor_pairs": int(validated_pairs["pair_type"].eq("impostor").sum()),
        "valid_pairs": int(validated_pairs["pair_valid"].sum()),
        "invalid_pairs": int((~validated_pairs["pair_valid"]).sum()),
    }
