"""Step 11: Optuna hyperparameter search.

Searches lr / weight_decay / batch_size with TPESampler + MedianPruner.
Every trial uses the same fixed TUNING_SEED -- HPO should compare
hyperparameter choices, not random noise. Multiple seeds only enter later,
at the 3-seed final-training step that uses whatever this search picks.

Each trial is a short training run (--epochs-per-trial) against the real
frozen split; the pruner can stop a clearly-bad trial early from its
intermediate validation loss (reported via train_dataset's epoch_callback,
so train.py itself never has to import optuna). The study is persisted to
experiments/<dataset>/tuning/study.db (SQLite) so a crash or restart
resumes rather than starting over.

This is a correctness-tested scaffold, not a search meant to actually run
here: train.py's own smoke test measured ~10.5s/batch for ResNet-34 on this
CPU, i.e. ~18h for one FairFace epoch -- real searches belong on the
cluster. --smoke-test-steps exists only to verify the search loop itself
(trial creation, pruning, storage) end-to-end on a handful of batches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from train import train_dataset

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
TUNING_SEED = 0


def objective(
    trial: optuna.Trial,
    dataset: str,
    epochs_per_trial: int,
    device: str,
    allow_dirty: bool,
    smoke_test_steps: int | None,
) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    def epoch_callback(epoch: int, val_metrics: dict) -> None:
        trial.report(val_metrics["total"], step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    out_dir = EXPERIMENTS_DIR / dataset / "tuning" / f"trial{trial.number}"
    result = train_dataset(
        dataset=dataset,
        epochs=epochs_per_trial,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=TUNING_SEED,
        device_str=device,
        allow_dirty=allow_dirty,
        max_steps_per_epoch=smoke_test_steps,
        out_dir=out_dir,
        epoch_callback=epoch_callback,
    )
    return result["best_val_loss"]


def run_study(
    dataset: str,
    n_trials: int,
    epochs_per_trial: int,
    device: str,
    allow_dirty: bool,
    smoke_test_steps: int | None = None,
) -> optuna.Study:
    study_dir = EXPERIMENTS_DIR / dataset / "tuning"
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(study_dir / 'study.db').as_posix()}"

    study = optuna.create_study(
        study_name=f"{dataset}_resnet34",
        storage=storage,
        direction="minimize",
        sampler=TPESampler(seed=TUNING_SEED),
        pruner=MedianPruner(n_warmup_steps=1),
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, dataset, epochs_per_trial, device, allow_dirty, smoke_test_steps),
        n_trials=n_trials,
    )

    (study_dir / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    study.trials_dataframe().to_csv(study_dir / "trials_summary.csv", index=False)
    return study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fairface", "utkface", "lfw"])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs-per-trial", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--smoke-test-steps",
        type=int,
        default=None,
        help="Truncate train/val to this many batches per epoch -- wiring checks only, never a real search.",
    )
    args = parser.parse_args()

    study = run_study(
        dataset=args.dataset,
        n_trials=args.n_trials,
        epochs_per_trial=args.epochs_per_trial,
        device=args.device,
        allow_dirty=args.allow_dirty,
        smoke_test_steps=args.smoke_test_steps,
    )
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
