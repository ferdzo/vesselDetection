from __future__ import annotations

import csv
import os
from pathlib import Path

DEFAULT_DATA_PATH = Path("ships-aerial-images/data.yaml")
DEFAULT_PROJECT_DIR = Path("runs/autoresearch")
DEFAULT_TIME_HOURS = 5 / 60
PRIMARY_METRIC_KEY = "metrics/mAP50-95(B)"


def ensure_dataset_exists(data_path: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found at '{data_path}'. Set YOLO_DATA or add the dataset before training."
        )


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_train_kwargs(defaults: dict[str, object]) -> dict[str, object]:
    kwargs = dict(defaults)
    kwargs["project"] = os.getenv("YOLO_PROJECT", str(kwargs["project"]))
    kwargs["name"] = os.getenv("YOLO_RUN_NAME", str(kwargs["name"]))
    kwargs["exist_ok"] = env_bool("YOLO_EXIST_OK", bool(kwargs.get("exist_ok", True)))

    time_override = os.getenv("YOLO_TIME_HOURS")
    if time_override:
        kwargs["time"] = float(time_override)

    device_override = os.getenv("YOLO_DEVICE")
    if device_override:
        kwargs["device"] = device_override

    return kwargs


def resolve_save_dir(
    project_dir: Path, run_name: str, expected_save_dir: Path | None = None
) -> Path:
    candidates: list[Path] = []
    if expected_save_dir is not None:
        candidates.append(expected_save_dir)
    candidates.append(project_dir / run_name)

    for candidate in candidates:
        if (candidate / "results.csv").exists():
            return candidate

    matches = sorted(
        (path for path in project_dir.glob(f"{run_name}*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for match in matches:
        if (match / "results.csv").exists():
            return match

    return expected_save_dir or (project_dir / run_name)


def _to_float(value: str | None) -> float | None:
    if value in {None, "", "nan", "None"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _first_float(
    row: dict[str, str], keys: list[str]
) -> tuple[str | None, float | None]:
    for key in keys:
        if key in row:
            value = _to_float(row.get(key))
            if value is not None:
                return key, value
    return None, None


def extract_experiment_summary(
    save_dir: Path,
    elapsed_seconds: float,
    peak_vram_mb: float,
    data_path: Path,
    model_name: str,
) -> dict[str, object]:
    results_csv = save_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Expected training metrics at '{results_csv}'.")

    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"Training metrics file '{results_csv}' is empty.")

    last_row = rows[-1]
    fitness_key, fitness = _first_float(
        last_row, [PRIMARY_METRIC_KEY, "metrics/mAP50(B)", "metrics/precision(B)"]
    )
    _, precision = _first_float(last_row, ["metrics/precision(B)"])
    _, recall = _first_float(last_row, ["metrics/recall(B)"])
    _, map50 = _first_float(last_row, ["metrics/mAP50(B)"])
    _, map50_95 = _first_float(last_row, [PRIMARY_METRIC_KEY])
    _, epoch = _first_float(last_row, ["epoch"])

    best_weights = save_dir / "weights/best.pt"
    last_weights = save_dir / "weights/last.pt"

    return {
        "fitness_key": fitness_key or PRIMARY_METRIC_KEY,
        "fitness": fitness,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "epoch": epoch,
        "training_seconds": elapsed_seconds,
        "total_seconds": elapsed_seconds,
        "peak_vram_mb": peak_vram_mb,
        "data_path": str(data_path),
        "model_name": model_name,
        "save_dir": str(save_dir),
        "results_csv": str(results_csv),
        "best_weights": str(best_weights),
        "best_weights_exists": best_weights.exists(),
        "last_weights": str(last_weights),
        "last_weights_exists": last_weights.exists(),
    }


def _format_metric(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def print_experiment_summary(summary: dict[str, object]) -> None:
    print("---")
    print(f"fitness_key:       {summary['fitness_key']}")
    print(f"fitness:           {_format_metric(summary['fitness'])}")
    print(f"training_seconds:  {_format_metric(summary['training_seconds'], digits=1)}")
    print(f"total_seconds:     {_format_metric(summary['total_seconds'], digits=1)}")
    print(f"peak_vram_mb:      {_format_metric(summary['peak_vram_mb'], digits=1)}")
    print(f"precision:         {_format_metric(summary['precision'])}")
    print(f"recall:            {_format_metric(summary['recall'])}")
    print(f"map50:             {_format_metric(summary['map50'])}")
    print(f"map50_95:          {_format_metric(summary['map50_95'])}")
    print(f"epoch:             {_format_metric(summary['epoch'], digits=0)}")
    print(f"data_path:         {summary['data_path']}")
    print(f"model:             {summary['model_name']}")
    print(f"save_dir:          {summary['save_dir']}")
    print(f"results_csv:       {summary['results_csv']}")
    print(f"best_weights:      {summary['best_weights']}")
    print(f"best_weights_ok:   {str(summary['best_weights_exists']).lower()}")
    print(f"last_weights:      {summary['last_weights']}")
    print(f"last_weights_ok:   {str(summary['last_weights_exists']).lower()}")
