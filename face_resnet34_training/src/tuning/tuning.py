"""Step 11: small learning-rate pilot (not a full Optuna search).

Given the two-day CPU-only training budget (see train.py's module docstring
for why the K40c isn't usable), optimizer/momentum/weight_decay/scheduler
are frozen (SGD, 0.9, 1e-4, cosine -- train.py's build_optimizer_and_scheduler)
across all three datasets, and batch size comes from train.py's --benchmark
measurement, not from search. The only thing tuned here is --lr, over a
small fixed grid, which is also the hyperparameter most likely to matter.
Using one largely-common training protocol across datasets additionally
keeps the cross-dataset comparison this project cares about cleaner --
independently-optimized per-dataset recipes would confound it.

Every trial uses the same fixed TUNING_SEED -- this pilot should compare lr
choices, not random noise. Multiple seeds only enter later, at the
final-training step that uses whatever this pilot picks.

Each trial is a short training run (--max-epochs-per-trial, with its own
--early-stopping-patience) against the real frozen split; the pruner can
stop a clearly-bad trial early from its intermediate validation loss
(reported via train_dataset's epoch_callback, so train.py itself never has
to import optuna). The study is persisted to experiments/<dataset>/tuning/
study.db (SQLite) so a crash or restart resumes rather than starting over.

Always run this with the same --num-threads/--num-workers allocation the
real final training run will use, same as train.py's --benchmark rule.
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
from train import train_dataset, DEFAULT_WEIGHT_DECAY

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
TUNING_SEED = 0
LR_GRID = [0.003, 0.01, 0.03]


def objective(
    trial: optuna.Trial,
    dataset: str,
    batch_size: int,
    max_epochs_per_trial: int,
    early_stopping_patience: int,
    num_threads: int | None,
    num_workers: int,
    device: str,
    allow_dirty: bool,
    smoke_test_steps: int | None,
) -> float:
    lr = trial.suggest_categorical("lr", LR_GRID)

    def epoch_callback(epoch: int, val_metrics: dict) -> None:
        trial.report(val_metrics["total"], step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    out_dir = EXPERIMENTS_DIR / dataset / "tuning" / f"trial{trial.number}"
    result = train_dataset(
        dataset=dataset,
        max_epochs=max_epochs_per_trial,
        batch_size=batch_size,
        lr=lr,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        seed=TUNING_SEED,
        device_str=device,
        allow_dirty=allow_dirty,
        num_threads=num_threads,
        num_workers=num_workers,
        early_stopping_patience=early_stopping_patience,
        max_steps_per_epoch=smoke_test_steps,
        out_dir=out_dir,
        epoch_callback=epoch_callback,
    )
    return result["best_val_loss"]


def run_study(
    dataset: str,
    batch_size: int,
    n_trials: int,
    max_epochs_per_trial: int,
    early_stopping_patience: int,
    num_threads: int | None,
    num_workers: int,
    device: str,
    allow_dirty: bool,
    smoke_test_steps: int | None = None,
) -> optuna.Study:
    study_dir = EXPERIMENTS_DIR / dataset / "tuning"
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(study_dir / 'study.db').as_posix()}"

    study = optuna.create_study(
        study_name=f"{dataset}_resnet34_lr_pilot",
        storage=storage,
        direction="minimize",
        sampler=TPESampler(seed=TUNING_SEED),
        pruner=MedianPruner(n_warmup_steps=1),
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            dataset,
            batch_size,
            max_epochs_per_trial,
            early_stopping_patience,
            num_threads,
            num_workers,
            device,
            allow_dirty,
            smoke_test_steps,
        ),
        n_trials=n_trials,
    )

    (study_dir / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    study.trials_dataframe().to_csv(study_dir / "trials_summary.csv", index=False)
    return study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fairface", "utkface", "lfw"])
    parser.add_argument("--batch-size", type=int, required=True, help="Fixed from --benchmark, not searched.")
    parser.add_argument("--n-trials", type=int, default=len(LR_GRID))
    parser.add_argument("--max-epochs-per-trial", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--smoke-test-steps",
        type=int,
        default=None,
        help="Truncate train/val to this many batches per epoch -- wiring checks only, never a real pilot.",
    )
    args = parser.parse_args()

    study = run_study(
        dataset=args.dataset,
        batch_size=args.batch_size,
        n_trials=args.n_trials,
        max_epochs_per_trial=args.max_epochs_per_trial,
        early_stopping_patience=args.early_stopping_patience,
        num_threads=args.num_threads,
        num_workers=args.num_workers,
        device=args.device,
        allow_dirty=args.allow_dirty,
        smoke_test_steps=args.smoke_test_steps,
    )
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
