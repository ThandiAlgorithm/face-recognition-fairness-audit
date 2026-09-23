from __future__ import annotations

import torch


def decode_boxes(
    locations: torch.Tensor,
    priors: torch.Tensor,
    variance: list[float] | tuple[float, float],
) -> torch.Tensor:
    centers = (
        priors[:, :2]
        + locations[:, :2]
        * variance[0]
        * priors[:, 2:]
    )

    sizes = (
        priors[:, 2:]
        * torch.exp(
            locations[:, 2:]
            * variance[1]
        )
    )

    top_left = (
        centers
        - sizes / 2
    )

    bottom_right = (
        centers
        + sizes / 2
    )

    return torch.cat(
        [
            top_left,
            bottom_right,
        ],
        dim=1,
    )


def decode_landmarks(
    landmarks: torch.Tensor,
    priors: torch.Tensor,
    variance: list[float] | tuple[float, float],
) -> torch.Tensor:
    decoded = []

    for index in range(5):
        start = index * 2

        point = (
            priors[:, :2]
            + landmarks[
                :,
                start:start + 2,
            ]
            * variance[0]
            * priors[:, 2:]
        )

        decoded.append(
            point
        )

    return torch.cat(
        decoded,
        dim=1,
    )
