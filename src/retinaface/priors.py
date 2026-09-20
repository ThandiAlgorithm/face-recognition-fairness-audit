from __future__ import annotations

from itertools import product
from math import ceil

import torch

from .model import MNET_CONFIG


class PriorBox:
    def __init__(
        self,
        image_size: tuple[int, int],
        config: dict | None = None,
    ) -> None:
        cfg = (
            MNET_CONFIG
            if config is None
            else config
        )

        self.image_size = image_size
        self.min_sizes = cfg[
            "min_sizes"
        ]
        self.steps = cfg[
            "steps"
        ]
        self.clip = bool(
            cfg["clip"]
        )

        self.feature_maps = [
            [
                ceil(
                    self.image_size[0]
                    / step
                ),
                ceil(
                    self.image_size[1]
                    / step
                ),
            ]
            for step
            in self.steps
        ]

    def generate(
        self,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        anchors: list[float] = []

        for level, feature_map in enumerate(
            self.feature_maps
        ):
            for row, column in product(
                range(
                    feature_map[0]
                ),
                range(
                    feature_map[1]
                ),
            ):
                for min_size in self.min_sizes[
                    level
                ]:
                    width = (
                        min_size
                        / self.image_size[1]
                    )

                    height = (
                        min_size
                        / self.image_size[0]
                    )

                    center_x = (
                        (column + 0.5)
                        * self.steps[level]
                        / self.image_size[1]
                    )

                    center_y = (
                        (row + 0.5)
                        * self.steps[level]
                        / self.image_size[0]
                    )

                    anchors.extend(
                        [
                            center_x,
                            center_y,
                            width,
                            height,
                        ]
                    )

        output = torch.tensor(
            anchors,
            dtype=torch.float32,
            device=device,
        ).view(
            -1,
            4,
        )

        if self.clip:
            output.clamp_(
                min=0.0,
                max=1.0,
            )

        return output
