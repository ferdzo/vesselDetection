from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from prepare import (
    DEFAULT_DATA_PATH,
    DEFAULT_PROJECT_DIR,
    DEFAULT_TIME_HOURS,
    build_train_kwargs,
    ensure_dataset_exists,
    extract_experiment_summary,
    print_experiment_summary,
    resolve_save_dir,
)

# The agent is expected to iterate on this file only.
MODEL_WEIGHTS = os.getenv("YOLO_MODEL", "yolo11l.pt")
DATA_PATH = Path(os.getenv("YOLO_DATA", str(DEFAULT_DATA_PATH)))
RUN_NAME = "vessel_detection_yolo11l"

TRAIN_PARAMS = {
    "epochs": 40,
    "time": DEFAULT_TIME_HOURS,
    "batch": 32,
    "imgsz": 640,
    "lr0": 5e-4,
    "lrf": 0.1,
    "warmup_epochs": 5,
    "warmup_bias_lr": 1e-6,
    "momentum": 0.937,
    "weight_decay": 1e-4,
    "optimizer": "AdamW",
    "device": "0",
    "project": str(DEFAULT_PROJECT_DIR),
    "name": RUN_NAME,
    "exist_ok": True,
    "save_period": 2,
    "workers": 8,
    "patience": 20,
    "cos_lr": True,
    "seed": 42,
    "deterministic": True,
    "plots": False,
}


def main() -> None:
    ensure_dataset_exists(DATA_PATH)

    train_kwargs = build_train_kwargs(TRAIN_PARAMS)
    save_dir = Path(str(train_kwargs["project"])) / str(train_kwargs["name"])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = YOLO(MODEL_WEIGHTS)

    start_time = time.time()
    train_result = model.train(data=str(DATA_PATH), **train_kwargs)
    elapsed_seconds = time.time() - start_time
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available()
        else 0.0
    )
    result_save_dir = getattr(train_result, "save_dir", None)
    save_dir = resolve_save_dir(
        project_dir=Path(str(train_kwargs["project"])),
        run_name=str(train_kwargs["name"]),
        expected_save_dir=Path(result_save_dir) if result_save_dir else save_dir,
    )

    summary = extract_experiment_summary(
        save_dir=save_dir,
        elapsed_seconds=elapsed_seconds,
        peak_vram_mb=peak_vram_mb,
        data_path=DATA_PATH,
        model_name=MODEL_WEIGHTS,
    )
    print_experiment_summary(summary)


if __name__ == "__main__":
    main()
