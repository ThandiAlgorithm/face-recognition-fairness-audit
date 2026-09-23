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
from tkinter import messagebox


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


# Arguments

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
        choices=[
            "within",
            "cross",
            "all",
        ],
        default="within",
    )

    parser.add_argument(
        "--page-size",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--flush-every",
        type=int,
        default=20,
    )

    return parser.parse_args()


# Small helpers

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
            f"Unexpected review statuses: "
            f"{sorted(bad)}"
        )


def resolve_image(
    data_root: Path,
    original_path: str,
):
    p = Path(str(original_path))

    if p.is_absolute():
        return p

    return data_root / p


def load_thumbnail(
    path: Path,
    size=(220, 220),
):
    with Image.open(path) as im:

        im = ImageOps.exif_transpose(
            im
        )

        im = im.convert("RGB")

        im.thumbnail(
            size,
            Image.Resampling.LANCZOS,
        )

        return im.copy()


# Backup and saving

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
        f"{csv_path.stem}"
        f"_grid_{stamp}.csv"
    )

    shutil.copy2(
        csv_path,
        backup,
    )

    return backup


def journal_path_for(csv_path):
    # Same journal used by the one-by-one reviewer.
    return csv_path.with_suffix(
        csv_path.suffix
        + ".journal.csv"
    )


def append_journal(
    journal_path,
    row_id,
    status,
):
    new_file = (
        not journal_path.exists()
    )

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
                int(row_id),
                status,
                datetime.now().isoformat(
                    timespec="seconds"
                ),
            ]
        )

        # Write every human decision immediately.
        f.flush()
        os.fsync(
            f.fileno()
        )


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

    # Later entries overwrite earlier entries for the same row.
    for row in journal.itertuples(
        index=False
    ):
        row_id = int(
            row.row_id
        )

        status = str(
            row.review_status
        )

        if status not in (
            ALLOWED_STATUSES
        ):
            continue

        if (
            0
            <= row_id
            < len(df)
        ):
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
        csv_path.suffix
        + ".tmp"
    )

    df.to_csv(
        temp,
        index=False,
    )

    os.replace(
        temp,
        csv_path,
    )


# Candidate prioritisation

def add_strength_columns(df):
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

    work["_regions"] = pd.to_numeric(
        work[
            "crop_matching_regions"
        ],
        errors="coerce",
    ).fillna(0)

    work["_crop_sum"] = pd.to_numeric(
        work[
            "crop_distance_sum"
        ],
        errors="coerce",
    ).fillna(999999)

    return work


def unresolved_for_mode(
    df,
    mode,
):
    work = df[
        df["review_status"]
        .fillna("candidate")
        .astype(str)
        .eq("candidate")
    ].copy()

    cross = (
        work["cross_dataset"]
        .map(canonical_bool)
    )

    if mode == "within":
        work = work[
            ~cross
        ].copy()

    elif mode == "cross":
        work = work[
            cross
        ].copy()

    return work


def find_best_anchor(
    unresolved,
):
    if unresolved.empty:
        return None

    paths = pd.concat(
        [
            unresolved[
                "original_path_a"
            ],
            unresolved[
                "original_path_b"
            ],
        ],
        ignore_index=True,
    )

    counts = (
        paths
        .value_counts()
    )

    if counts.empty:
        return None

    return str(
        counts.index[0]
    )


def get_anchor_candidates(
    unresolved,
    anchor,
):
    rows = unresolved[
        (
            unresolved[
                "original_path_a"
            ]
            == anchor
        )
        |
        (
            unresolved[
                "original_path_b"
            ]
            == anchor
        )
    ].copy()

    rows["_row_id"] = (
        rows.index
    )

    rows = add_strength_columns(
        rows
    )

    # Strongest-looking candidates first, but show all of them.
    rows = rows.sort_values(
        by=[
            "_method_rank",
            "_phash",
            "_regions",
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

    return rows


# GUI

class GridReviewer:

    def __init__(
        self,
        root,
        df,
        csv_path,
        data_root,
        journal_path,
        mode,
        page_size,
        flush_every,
    ):
        self.root = root
        self.df = df

        self.csv_path = (
            csv_path
        )

        self.data_root = (
            data_root
        )

        self.journal_path = (
            journal_path
        )

        self.mode = mode

        self.page_size = max(
            1,
            int(page_size),
        )

        self.flush_every = max(
            1,
            int(flush_every),
        )

        self.decisions_since_flush = 0

        self.current_anchor = None
        self.current_rows = None

        # Bumped every time a new grid starts building, so a batch
        # still scheduled for an old group can tell it is stale.
        self._grid_generation = 0

        # Keeps ImageTk references alive.
        self.photos = {}

        root.title(
            "Grouped Near-Duplicate Review"
        )

        root.geometry(
            "1600x950"
        )

        root.minsize(
            1200,
            750,
        )

        # Top information

        self.header = tk.Label(
            root,
            text="",
            anchor="w",
            justify="left",
            font=(
                "Segoe UI",
                12,
                "bold",
            ),
        )

        self.header.pack(
            fill="x",
            padx=10,
            pady=(8, 3),
        )

        self.rule_label = tk.Label(
            root,
            text=(
                "CONFIRMED = same underlying photograph.   "
                "Same person in a different photograph = FALSE ALARM."
            ),
            anchor="w",
            justify="left",
            font=(
                "Segoe UI",
                10,
                "bold",
            ),
        )

        self.rule_label.pack(
            fill="x",
            padx=10,
            pady=(0, 5),
        )

        # Main left and right layout

        main = tk.Frame(
            root
        )

        main.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5,
        )

        # Anchor side
        self.anchor_frame = tk.Frame(
            main,
            width=360,
        )

        self.anchor_frame.pack(
            side="left",
            fill="y",
            padx=(0, 10),
        )

        self.anchor_title = tk.Label(
            self.anchor_frame,
            text="ANCHOR IMAGE",
            font=(
                "Segoe UI",
                12,
                "bold",
            ),
        )

        self.anchor_title.pack(
            pady=(5, 5),
        )

        self.anchor_image = tk.Label(
            self.anchor_frame,
            text="",
            compound="top",
            justify="center",
        )

        self.anchor_image.pack(
            fill="both",
            expand=False,
        )

        self.anchor_info = tk.Label(
            self.anchor_frame,
            text="",
            justify="left",
            anchor="w",
            wraplength=350,
        )

        self.anchor_info.pack(
            fill="x",
            pady=8,
        )

        # Scrollable candidate area

        right = tk.Frame(
            main
        )

        right.pack(
            side="left",
            fill="both",
            expand=True,
        )

        self.canvas = tk.Canvas(
            right
        )

        scrollbar = tk.Scrollbar(
            right,
            orient="vertical",
            command=self.canvas.yview,
        )

        self.canvas.configure(
            yscrollcommand=(
                scrollbar.set
            )
        )

        scrollbar.pack(
            side="right",
            fill="y",
        )

        self.canvas.pack(
            side="left",
            fill="both",
            expand=True,
        )

        self.grid_frame = tk.Frame(
            self.canvas
        )

        self.canvas_window = (
            self.canvas.create_window(
                (0, 0),
                window=self.grid_frame,
                anchor="nw",
            )
        )

        self.grid_frame.bind(
            "<Configure>",
            self._update_scrollregion,
        )

        self.canvas.bind(
            "<Configure>",
            self._resize_canvas_window,
        )

        # Mouse wheel
        self.canvas.bind_all(
            "<MouseWheel>",
            self._mousewheel,
        )

        root.bind(
            "a",
            lambda _:
                self.mark_all_visible_false(),
        )

        root.bind(
            "A",
            lambda _:
                self.mark_all_visible_false(),
        )

        # Bottom controls

        bottom = tk.Frame(
            root
        )

        bottom.pack(
            fill="x",
            padx=10,
            pady=(4, 10),
        )

        self.progress_label = tk.Label(
            bottom,
            text="",
            anchor="w",
            justify="left",
        )

        self.progress_label.pack(
            side="left",
        )

        tk.Button(
            bottom,
            text="All visible = False [A]",
            width=22,
            command=self.mark_all_visible_false,
        ).pack(
            side="right",
            padx=4,
        )

        tk.Button(
            bottom,
            text="Next group / refresh",
            width=20,
            command=self.next_group,
        ).pack(
            side="right",
            padx=4,
        )

        tk.Button(
            bottom,
            text="Save",
            width=12,
            command=self.flush,
        ).pack(
            side="right",
            padx=4,
        )

        tk.Button(
            bottom,
            text="Save + Quit",
            width=14,
            command=self.save_and_quit,
        ).pack(
            side="right",
            padx=4,
        )

        root.protocol(
            "WM_DELETE_WINDOW",
            self.save_and_quit,
        )

        self.load_group()

    # Canvas

    def _update_scrollregion(
        self,
        _event=None,
    ):
        self.canvas.configure(
            scrollregion=(
                self.canvas.bbox(
                    "all"
                )
            )
        )

    def _resize_canvas_window(
        self,
        event,
    ):
        self.canvas.itemconfigure(
            self.canvas_window,
            width=event.width,
        )

    def _mousewheel(
        self,
        event,
    ):
        self.canvas.yview_scroll(
            int(
                -1
                * (event.delta / 120)
            ),
            "units",
        )

    # Save

    def flush(self):
        atomic_write_csv(
            self.df,
            self.csv_path,
        )

        self.decisions_since_flush = 0

    # Progress

    def update_progress(self):
        unresolved_all = (
            unresolved_for_mode(
                self.df,
                self.mode,
            )
        )

        overall_candidate = (
            self.df[
                "review_status"
            ]
            .fillna("candidate")
            .eq("candidate")
            .sum()
        )

        self.progress_label.config(
            text=(
                f"Unresolved in this mode: "
                f"{len(unresolved_all):,}     "
                f"All remaining candidates: "
                f"{overall_candidate:,}"
            )
        )

    # Load one anchor group

    def load_group(self):
        self.photos.clear()

        unresolved = (
            unresolved_for_mode(
                self.df,
                self.mode,
            )
        )

        self.update_progress()

        if unresolved.empty:

            self.header.config(
                text=(
                    f"{self.mode.upper()} "
                    f"REVIEW COMPLETE"
                )
            )

            self.anchor_image.config(
                image="",
                text="",
            )

            self.anchor_info.config(
                text=""
            )

            for child in (
                self.grid_frame
                .winfo_children()
            ):
                child.destroy()

            messagebox.showinfo(
                "Review complete",
                (
                    f"No candidate pairs remain "
                    f"for mode: {self.mode}"
                ),
            )

            return

        anchor = find_best_anchor(
            unresolved
        )

        self.current_anchor = (
            anchor
        )

        rows = (
            get_anchor_candidates(
                unresolved,
                anchor,
            )
        )

        self.current_rows = rows

        total_for_anchor = (
            (
                (
                    unresolved[
                        "original_path_a"
                    ]
                    == anchor
                )
                |
                (
                    unresolved[
                        "original_path_b"
                    ]
                    == anchor
                )
            )
            .sum()
        )

        self.header.config(
            text=(
                f"Mode: {self.mode}     "
                f"Anchor has "
                f"{total_for_anchor:,} "
                f"unresolved candidate pairs     "
                f"Showing ALL "
                f"{len(rows):,}"
            )
        )

        self.show_anchor(
            anchor,
            rows,
        )

        self.show_candidate_grid(
            anchor,
            rows,
        )

    # Anchor

    def show_anchor(
        self,
        anchor,
        rows,
    ):
        if len(rows) == 0:
            return

        first = rows.iloc[0]

        if (
            first[
                "original_path_a"
            ]
            == anchor
        ):
            dataset = first[
                "dataset_a"
            ]
        else:
            dataset = first[
                "dataset_b"
            ]

        image_path = resolve_image(
            self.data_root,
            anchor,
        )

        try:
            im = load_thumbnail(
                image_path,
                size=(340, 340),
            )

            photo = (
                ImageTk.PhotoImage(
                    im
                )
            )

            self.photos[
                "anchor"
            ] = photo

            self.anchor_image.config(
                image=photo,
                text="",
            )

        except Exception as exc:

            self.anchor_image.config(
                image="",
                text=(
                    "ANCHOR IMAGE ERROR\n"
                    f"{type(exc).__name__}\n"
                    f"{exc}"
                ),
            )

        self.anchor_info.config(
            text=(
                f"Dataset:\n"
                f"{dataset}\n\n"
                f"Path:\n"
                f"{anchor}"
            )
        )

    # Candidate grid

    def show_candidate_grid(
        self,
        anchor,
        rows,
    ):
        for child in (
            self.grid_frame
            .winfo_children()
        ):
            child.destroy()

        # Bump the generation so a batch still scheduled from the
        # previous group stops adding cards to this new grid.
        self._grid_generation += 1

        self._build_grid_batch(
            anchor,
            rows,
            start=0,
            generation=self._grid_generation,
        )

    def _build_grid_batch(
        self,
        anchor,
        rows,
        start,
        generation,
    ):
        # A newer group has since been loaded, stop here.
        if generation != self._grid_generation:
            return

        # Three candidate cards per row.
        columns = 3
        batch_size = 30

        end = min(
            start + batch_size,
            len(rows),
        )

        for display_pos in range(start, end):
            row = rows.iloc[display_pos]

            row_id = int(
                row["_row_id"]
            )

            if (
                row[
                    "original_path_a"
                ]
                == anchor
            ):
                candidate_path = row[
                    "original_path_b"
                ]

                candidate_dataset = row[
                    "dataset_b"
                ]

            else:
                candidate_path = row[
                    "original_path_a"
                ]

                candidate_dataset = row[
                    "dataset_a"
                ]

            r = (
                display_pos
                // columns
            )

            c = (
                display_pos
                % columns
            )

            card = tk.Frame(
                self.grid_frame,
                bd=2,
                relief="groove",
                padx=5,
                pady=5,
            )

            card.grid(
                row=r,
                column=c,
                sticky="nsew",
                padx=5,
                pady=5,
            )

            self.grid_frame.grid_columnconfigure(
                c,
                weight=1,
            )

            # Image

            path = resolve_image(
                self.data_root,
                candidate_path,
            )

            image_label = tk.Label(
                card,
                text="",
            )

            image_label.pack()

            try:
                im = load_thumbnail(
                    path,
                    size=(210, 210),
                )

                photo = (
                    ImageTk.PhotoImage(
                        im
                    )
                )

                self.photos[
                    row_id
                ] = photo

                image_label.config(
                    image=photo,
                )

            except Exception as exc:

                image_label.config(
                    text=(
                        "IMAGE ERROR\n"
                        f"{type(exc).__name__}"
                    )
                )

            # Candidate metadata

            info = (
                f"{candidate_dataset}\n"
                f"{candidate_path}\n\n"
                f"method: "
                f"{row['candidate_method']}\n"
                f"pHash distance: "
                f"{row['phash_distance']}\n"
                f"crop regions: "
                f"{row['crop_matching_regions']}\n"
                f"crop sum: "
                f"{row['crop_distance_sum']}"
            )

            tk.Label(
                card,
                text=info,
                justify="left",
                anchor="w",
                wraplength=260,
            ).pack(
                fill="x",
                pady=4,
            )

            current_status = str(
                self.df.at[
                    row_id,
                    "review_status",
                ]
            )

            status_label = tk.Label(
                card,
                text=current_status,
                font=(
                    "Segoe UI",
                    10,
                    "bold",
                ),
            )

            status_label.pack(
                pady=3,
            )

            # Decision buttons

            buttons = tk.Frame(
                card
            )

            buttons.pack(
                fill="x",
                pady=4,
            )

            tk.Button(
                buttons,
                text="C",
                width=4,
                command=lambda rid=row_id,
                sl=status_label:
                    self.set_decision(
                        rid,
                        "confirmed",
                        sl,
                    ),
            ).pack(
                side="left",
                padx=2,
            )

            tk.Button(
                buttons,
                text="F",
                width=4,
                command=lambda rid=row_id,
                sl=status_label:
                    self.set_decision(
                        rid,
                        "false_alarm",
                        sl,
                    ),
            ).pack(
                side="left",
                padx=2,
            )

            tk.Button(
                buttons,
                text="U",
                width=4,
                command=lambda rid=row_id,
                sl=status_label:
                    self.set_decision(
                        rid,
                        "uncertain",
                        sl,
                    ),
            ).pack(
                side="left",
                padx=2,
            )

            tk.Label(
                buttons,
                text=(
                    "C=yes   "
                    "F=no   "
                    "U=unsure"
                ),
            ).pack(
                side="left",
                padx=5,
            )

        if end < len(rows):
            # Yield back to the event loop between batches so the
            # window stays responsive on large anchors instead of
            # freezing while it builds hundreds of cards at once.
            self.root.after(
                1,
                self._build_grid_batch,
                anchor,
                rows,
                end,
                generation,
            )

    # Human decision

    def set_decision(
        self,
        row_id,
        status,
        status_label,
    ):
        if status not in (
            DECISION_STATUSES
        ):
            raise ValueError(
                status
            )

        # Allows changing your mind while the pair is still on screen.
        self.df.at[
            row_id,
            "review_status",
        ] = status

        append_journal(
            self.journal_path,
            row_id,
            status,
        )

        status_label.config(
            text=status
        )

        self.decisions_since_flush += 1

        if (
            self.decisions_since_flush
            >= self.flush_every
        ):
            self.flush()

        self.update_progress()

    def mark_all_visible_false(self):
        if self.current_rows is None:
            return

        changed = 0

        for _, row in self.current_rows.iterrows():
            row_id = int(row["_row_id"])

            # Only bulk-mark rows that are still unresolved.
            if (
                self.df.at[
                    row_id,
                    "review_status",
                ]
                == "candidate"
            ):
                self.df.at[
                    row_id,
                    "review_status",
                ] = "false_alarm"

                append_journal(
                    self.journal_path,
                    row_id,
                    "false_alarm",
                )

                changed += 1

        self.decisions_since_flush += changed

        if (
            self.decisions_since_flush
            >= self.flush_every
        ):
            self.flush()

        # Refresh the same screen so exceptions can be changed to C or U.
        self.show_candidate_grid(
            self.current_anchor,
            self.current_rows,
        )

        self.update_progress()

    # Next

    def next_group(self):
        self.flush()

        self.load_group()

        self.canvas.yview_moveto(
            0
        )

    # Exit

    def save_and_quit(self):
        try:
            self.flush()

        finally:
            self.root.destroy()


# Main

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

    replayed = replay_journal(
        df,
        journal_path,
    )

    if replayed:
        atomic_write_csv(
            df,
            csv_path,
        )

    backup = make_backup(
        csv_path
    )

    unresolved = (
        unresolved_for_mode(
            df,
            args.mode,
        )
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
        f"Unresolved pairs "
        f"for mode '{args.mode}':",
        len(unresolved),
    )

    if unresolved.empty:
        print(
            "Nothing left to review."
        )
        return

    root = tk.Tk()

    GridReviewer(
        root=root,
        df=df,
        csv_path=csv_path,
        data_root=data_root,
        journal_path=journal_path,
        mode=args.mode,
        page_size=args.page_size,
        flush_every=args.flush_every,
    )

    root.mainloop()


if __name__ == "__main__":
    main()
