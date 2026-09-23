from __future__ import annotations

import argparse
import csv
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps, ImageTk
import tkinter as tk


ALLOWED_STATUSES = {
    "candidate",
    "confirmed",
    "false_alarm",
    "uncertain",
}

DECISION_STATUSES = {
    "confirmed",
    "false_alarm",
    "uncertain",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default="metadata/near_duplicate_review.csv",
    )

    parser.add_argument(
        "--data-root",
        default="data",
    )

    parser.add_argument(
        "--mode",
        choices=["all", "cross", "within"],
        default="all",
    )

    parser.add_argument(
        "--method",
        choices=[
            "all",
            "crop",
            "phash",
            "phash+crop",
        ],
        default="all",
    )

    parser.add_argument(
        "--order",
        choices=["strong", "file"],
        default="strong",
    )

    parser.add_argument(
        "--min-crop-regions",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
    )

    return parser.parse_args()


def canonical_bool(value):
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def validate_review_frame(df):
    required = {
        "original_path_a",
        "original_path_b",
        "dataset_a",
        "dataset_b",
        "cross_dataset",
        "phash_distance",
        "crop_matching_regions",
        "crop_distance_sum",
        "candidate_method",
        "review_status",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    statuses = set(
        df["review_status"]
        .fillna("candidate")
        .astype(str)
    )

    bad = statuses - ALLOWED_STATUSES

    if bad:
        raise RuntimeError(
            f"Unknown review statuses: {sorted(bad)}"
        )


def make_backup(csv_path):
    backup_dir = (
        csv_path.parent
        / "review_backups"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup = backup_dir / (
        f"{csv_path.stem}_{stamp}.csv"
    )

    shutil.copy2(
        csv_path,
        backup,
    )

    return backup


def journal_path_for(csv_path):
    return csv_path.with_suffix(
        csv_path.suffix + ".journal.csv"
    )


def append_journal(
    journal_path,
    row_id,
    status,
):
    new_file = not journal_path.exists()

    with journal_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        if new_file:
            writer.writerow(
                [
                    "row_id",
                    "review_status",
                    "timestamp",
                ]
            )

        writer.writerow(
            [
                row_id,
                status,
                datetime.now().isoformat(
                    timespec="seconds"
                ),
            ]
        )

        f.flush()
        os.fsync(f.fileno())


def replay_journal(
    df,
    journal_path,
):
    if not journal_path.exists():
        return 0

    journal = pd.read_csv(
        journal_path
    )

    if journal.empty:
        return 0

    applied = 0

    for row in journal.itertuples(
        index=False
    ):
        row_id = int(row.row_id)
        status = str(
            row.review_status
        )

        if status not in DECISION_STATUSES:
            raise RuntimeError(
                f"Invalid journal status: "
                f"{status}"
            )

        if 0 <= row_id < len(df):
            df.at[
                row_id,
                "review_status",
            ] = status

            applied += 1

    return applied


def atomic_write_csv(
    df,
    csv_path,
):
    temp = csv_path.with_suffix(
        csv_path.suffix + ".tmp"
    )

    df.to_csv(
        temp,
        index=False,
    )

    os.replace(
        temp,
        csv_path,
    )


def resolve_image(
    data_root,
    original_path,
):
    path = Path(
        str(original_path)
    )

    if path.is_absolute():
        return path

    return data_root / path


def fit_image(
    path,
    max_size=(540, 540),
):
    with Image.open(path) as im:

        im = ImageOps.exif_transpose(
            im
        )

        im = im.convert("RGB")

        im.thumbnail(
            max_size,
            Image.Resampling.LANCZOS,
        )

        return im.copy()


def priority_frame(df):
    work = df.copy()

    method_rank = {
        "phash+crop": 0,
        "phash": 1,
        "crop": 2,
    }

    work["_method_rank"] = (
        work["candidate_method"]
        .map(method_rank)
        .fillna(9)
        .astype(int)
    )

    work["_phash"] = pd.to_numeric(
        work["phash_distance"],
        errors="coerce",
    ).fillna(999999)

    work["_crop_regions"] = (
        pd.to_numeric(
            work[
                "crop_matching_regions"
            ],
            errors="coerce",
        )
        .fillna(0)
    )

    work["_crop_sum"] = (
        pd.to_numeric(
            work[
                "crop_distance_sum"
            ],
            errors="coerce",
        )
        .fillna(999999)
    )

    return work.sort_values(
        by=[
            "_method_rank",
            "_phash",
            "_crop_regions",
            "_crop_sum",
            "_row_id",
        ],
        ascending=[
            True,
            True,
            False,
            True,
            True,
        ],
        kind="stable",
    )


def build_queue(
    df,
    mode,
    method,
    min_crop_regions,
    order,
):
    work = df.copy()

    work["_row_id"] = (
        work.index
    )

    # Only show rows still waiting
    # for human review.
    work = work[
        work["review_status"]
        .fillna("candidate")
        .astype(str)
        .eq("candidate")
    ].copy()

    cross = work[
        "cross_dataset"
    ].map(
        canonical_bool
    )

    if mode == "cross":
        work = work[
            cross
        ].copy()

    elif mode == "within":
        work = work[
            ~cross
        ].copy()

    if method != "all":
        work = work[
            work[
                "candidate_method"
            ].eq(method)
        ].copy()

    if min_crop_regions > 0:

        regions = pd.to_numeric(
            work[
                "crop_matching_regions"
            ],
            errors="coerce",
        ).fillna(0)

        work = work[
            regions
            >= min_crop_regions
        ].copy()

    if order == "strong":
        work = priority_frame(
            work
        )

    return work


class Reviewer:

    def __init__(
        self,
        root,
        full_df,
        queue,
        csv_path,
        journal_path,
        data_root,
        flush_every,
    ):
        self.root = root

        self.df = full_df

        self.queue = (
            queue
            .reset_index(drop=True)
        )

        self.csv_path = csv_path
        self.journal_path = (
            journal_path
        )

        self.data_root = data_root

        self.flush_every = max(
            1,
            int(flush_every),
        )

        self.pos = 0

        self.decisions_since_flush = 0

        self.photo_a = None
        self.photo_b = None

        root.title(
            "Near Duplicate Review"
        )

        root.geometry(
            "1250x820"
        )

        root.minsize(
            1000,
            700,
        )

        self.header = tk.Label(
            root,
            text="",
            anchor="w",
            justify="left",
            font=(
                "Segoe UI",
                11,
                "bold",
            ),
        )

        self.header.pack(
            fill="x",
            padx=10,
            pady=(8, 4),
        )

        image_frame = tk.Frame(
            root
        )

        image_frame.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5,
        )

        self.left_label = tk.Label(
            image_frame,
            text="",
            compound="top",
            anchor="n",
            justify="center",
        )

        self.left_label.pack(
            side="left",
            fill="both",
            expand=True,
            padx=5,
        )

        self.right_label = tk.Label(
            image_frame,
            text="",
            compound="top",
            anchor="n",
            justify="center",
        )

        self.right_label.pack(
            side="left",
            fill="both",
            expand=True,
            padx=5,
        )

        self.meta = tk.Label(
            root,
            text="",
            anchor="w",
            justify="left",
            font=(
                "Consolas",
                10,
            ),
        )

        self.meta.pack(
            fill="x",
            padx=10,
            pady=5,
        )

        controls = tk.Frame(
            root
        )

        controls.pack(
            fill="x",
            padx=10,
            pady=(5, 10),
        )

        tk.Button(
            controls,
            text="Confirmed [C]",
            width=18,
            command=lambda:
                self.decide(
                    "confirmed"
                ),
        ).pack(
            side="left",
            padx=4,
        )

        tk.Button(
            controls,
            text="False alarm [F]",
            width=18,
            command=lambda:
                self.decide(
                    "false_alarm"
                ),
        ).pack(
            side="left",
            padx=4,
        )

        tk.Button(
            controls,
            text="Uncertain [U]",
            width=18,
            command=lambda:
                self.decide(
                    "uncertain"
                ),
        ).pack(
            side="left",
            padx=4,
        )

        tk.Button(
            controls,
            text="Skip [S]",
            width=14,
            command=self.skip,
        ).pack(
            side="left",
            padx=4,
        )

        tk.Button(
            controls,
            text="Back [B]",
            width=14,
            command=self.back,
        ).pack(
            side="left",
            padx=4,
        )

        tk.Button(
            controls,
            text="Save + Quit [Q]",
            width=16,
            command=self.save_and_quit,
        ).pack(
            side="right",
            padx=4,
        )

        root.bind(
            "c",
            lambda _:
                self.decide(
                    "confirmed"
                ),
        )

        root.bind(
            "C",
            lambda _:
                self.decide(
                    "confirmed"
                ),
        )

        root.bind(
            "f",
            lambda _:
                self.decide(
                    "false_alarm"
                ),
        )

        root.bind(
            "F",
            lambda _:
                self.decide(
                    "false_alarm"
                ),
        )

        root.bind(
            "u",
            lambda _:
                self.decide(
                    "uncertain"
                ),
        )

        root.bind(
            "U",
            lambda _:
                self.decide(
                    "uncertain"
                ),
        )

        root.bind(
            "s",
            lambda _:
                self.skip(),
        )

        root.bind(
            "S",
            lambda _:
                self.skip(),
        )

        root.bind(
            "b",
            lambda _:
                self.back(),
        )

        root.bind(
            "B",
            lambda _:
                self.back(),
        )

        root.bind(
            "q",
            lambda _:
                self.save_and_quit(),
        )

        root.bind(
            "Q",
            lambda _:
                self.save_and_quit(),
        )

        root.protocol(
            "WM_DELETE_WINDOW",
            self.save_and_quit,
        )

        self.show_current()

    def flush(self):

        atomic_write_csv(
            self.df,
            self.csv_path,
        )

        self.decisions_since_flush = 0

    def current_row(self):

        if self.pos >= len(
            self.queue
        ):
            return None

        return self.queue.iloc[
            self.pos
        ]

    def show_current(self):

        row = self.current_row()

        if row is None:

            self.flush()

            self.header.config(
                text=(
                    "Review queue complete "
                    "for this session."
                )
            )

            self.left_label.config(
                image="",
                text="",
            )

            self.right_label.config(
                image="",
                text="",
            )

            self.meta.config(
                text=(
                    "CSV saved."
                )
            )

            return

        row_id = int(
            row["_row_id"]
        )

        current_status = str(
            self.df.at[
                row_id,
                "review_status",
            ]
        )

        path_a = resolve_image(
            self.data_root,
            row[
                "original_path_a"
            ],
        )

        path_b = resolve_image(
            self.data_root,
            row[
                "original_path_b"
            ],
        )

        self.header.config(
            text=(
                f"Pair "
                f"{self.pos + 1:,}"
                f" / "
                f"{len(self.queue):,}"
                f"     "
                f"CSV row {row_id:,}"
                f"     "
                f"status: "
                f"{current_status}"
            )
        )

        try:
            image_a = fit_image(
                path_a
            )

            self.photo_a = (
                ImageTk.PhotoImage(
                    image_a
                )
            )

            self.left_label.config(
                image=self.photo_a,
                text=(
                    f"{row['dataset_a']}\n"
                    f"{row['original_path_a']}"
                ),
            )

        except Exception as exc:

            self.photo_a = None

            self.left_label.config(
                image="",
                text=(
                    "IMAGE A ERROR\n"
                    f"{path_a}\n"
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        try:
            image_b = fit_image(
                path_b
            )

            self.photo_b = (
                ImageTk.PhotoImage(
                    image_b
                )
            )

            self.right_label.config(
                image=self.photo_b,
                text=(
                    f"{row['dataset_b']}\n"
                    f"{row['original_path_b']}"
                ),
            )

        except Exception as exc:

            self.photo_b = None

            self.right_label.config(
                image="",
                text=(
                    "IMAGE B ERROR\n"
                    f"{path_b}\n"
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

        self.meta.config(
            text=(
                f"cross_dataset      = "
                f"{row['cross_dataset']}\n"

                f"candidate_method   = "
                f"{row['candidate_method']}\n"

                f"phash_distance     = "
                f"{row['phash_distance']}\n"

                f"crop_regions       = "
                f"{row['crop_matching_regions']}\n"

                f"crop_distance_sum  = "
                f"{row['crop_distance_sum']}\n\n"

                "C = confirmed duplicate\n"
                "F = false alarm\n"
                "U = uncertain\n"
                "S = skip\n"
                "B = go back\n"
                "Q = save and quit"
            )
        )

    def decide(
        self,
        status,
    ):

        if status not in (
            DECISION_STATUSES
        ):
            raise ValueError(
                status
            )

        row = self.current_row()

        if row is None:
            return

        row_id = int(
            row["_row_id"]
        )

        self.df.at[
            row_id,
            "review_status",
        ] = status

        # Save decision immediately
        # to a journal.
        append_journal(
            self.journal_path,
            row_id,
            status,
        )

        self.pos += 1

        self.decisions_since_flush += 1

        if (
            self.decisions_since_flush
            >= self.flush_every
        ):
            self.flush()

        self.show_current()

    def skip(self):

        if (
            self.current_row()
            is None
        ):
            return

        self.pos += 1

        self.show_current()

    def back(self):

        if self.pos <= 0:
            return

        self.pos -= 1

        self.show_current()

    def save_and_quit(self):

        try:
            self.flush()

        finally:
            self.root.destroy()


def main():

    args = parse_args()

    csv_path = Path(
        args.csv
    )

    data_root = Path(
        args.data_root
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            csv_path
        )

    if not data_root.exists():
        raise FileNotFoundError(
            data_root
        )

    df = pd.read_csv(
        csv_path
    )

    validate_review_frame(
        df
    )

    journal_path = (
        journal_path_for(
            csv_path
        )
    )

    replayed = (
        replay_journal(
            df,
            journal_path,
        )
    )

    if replayed:
        atomic_write_csv(
            df,
            csv_path,
        )

    backup = make_backup(
        csv_path
    )

    queue = build_queue(
        df=df,
        mode=args.mode,
        method=args.method,
        min_crop_regions=(
            args.min_crop_regions
        ),
        order=args.order,
    )

    print(
        "Review CSV:",
        csv_path,
    )

    print(
        "Backup:",
        backup,
    )

    print(
        "Journal:",
        journal_path,
    )

    print(
        "Journal decisions replayed:",
        replayed,
    )

    print(
        "Pairs in this queue:",
        len(queue),
    )

    if queue.empty:

        print(
            "Nothing to review "
            "for this filter."
        )

        return

    root = tk.Tk()

    Reviewer(
        root=root,
        full_df=df,
        queue=queue,
        csv_path=csv_path,
        journal_path=journal_path,
        data_root=data_root,
        flush_every=args.flush_every,
    )

    root.mainloop()


if __name__ == "__main__":
    main()
