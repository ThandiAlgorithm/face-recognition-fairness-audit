from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from src.retinaface import RetinaFace


EXPECTED_SHA256 = (
    "2979b33ffafda5d74b6948cd7a5b9a7a"
    "62f62b949cef24e95fd15d2883a65220"
)

EXPECTED_STATE_DICT_ENTRIES = 300


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def main() -> None:
    checkpoint_path = (
        Path.home()
        / ".cache"
        / "torch"
        / "hub"
        / "checkpoints"
        / "mobilenet0.25_Final.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    actual_sha256 = sha256_file(
        checkpoint_path
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Bytes:",
        checkpoint_path.stat().st_size,
    )

    print(
        "SHA256:",
        actual_sha256,
    )

    if (
        actual_sha256
        != EXPECTED_SHA256
    ):
        raise RuntimeError(
            "Checkpoint SHA256 mismatch."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Checkpoint is not a state dict."
        )

    model = RetinaFace()

    model_state = (
        model.state_dict()
    )

    checkpoint_keys = set(
        checkpoint.keys()
    )

    model_keys = set(
        model_state.keys()
    )

    missing_from_model = sorted(
        checkpoint_keys
        - model_keys
    )

    missing_from_checkpoint = sorted(
        model_keys
        - checkpoint_keys
    )

    shape_mismatches = [
        key
        for key
        in sorted(
            checkpoint_keys
            & model_keys
        )
        if tuple(
            checkpoint[key].shape
        )
        != tuple(
            model_state[key].shape
        )
    ]

    print()
    print(
        "Checkpoint state-dict entries:",
        len(checkpoint),
    )

    print(
        "Model state-dict entries:",
        len(model_state),
    )

    print(
        "Model trainable parameters:",
        sum(
            parameter.numel()
            for parameter
            in model.parameters()
        ),
    )

    print(
        "Checkpoint keys absent from model:",
        len(
            missing_from_model
        ),
    )

    print(
        "Model keys absent from checkpoint:",
        len(
            missing_from_checkpoint
        ),
    )

    print(
        "Shape mismatches:",
        len(
            shape_mismatches
        ),
    )

    if (
        len(checkpoint)
        != EXPECTED_STATE_DICT_ENTRIES
    ):
        raise RuntimeError(
            "Unexpected number of checkpoint "
            f"state-dict entries: {len(checkpoint)}"
        )

    if missing_from_model:
        print()
        print(
            "Checkpoint keys missing from model:"
        )

        for key in missing_from_model[:20]:
            print(
                " ",
                key,
            )

    if missing_from_checkpoint:
        print()
        print(
            "Model keys missing from checkpoint:"
        )

        for key in missing_from_checkpoint[:20]:
            print(
                " ",
                key,
            )

    if shape_mismatches:
        print()
        print(
            "Shape mismatches:"
        )

        for key in shape_mismatches[:20]:
            print(
                " ",
                key,
                "checkpoint=",
                tuple(
                    checkpoint[key].shape
                ),
                "model=",
                tuple(
                    model_state[key].shape
                ),
            )

    if (
        missing_from_model
        or missing_from_checkpoint
        or shape_mismatches
    ):
        raise RuntimeError(
            "Checkpoint architecture "
            "compatibility test failed."
        )

    model.load_state_dict(
        checkpoint,
        strict=True,
    )

    model.eval()

    print()
    print(
        "strict=True load: PASS"
    )

    test_input = torch.zeros(
        (
            1,
            3,
            240,
            320,
        ),
        dtype=torch.float32,
    )

    with torch.inference_mode():
        (
            boxes,
            classes,
            landmarks,
        ) = model(
            test_input
        )

    print(
        "Forward pass: PASS"
    )

    print(
        "boxes:",
        tuple(
            boxes.shape
        ),
    )

    print(
        "classes:",
        tuple(
            classes.shape
        ),
    )

    print(
        "landmarks:",
        tuple(
            landmarks.shape
        ),
    )


if __name__ == "__main__":
    main()
