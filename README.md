# vesselDetection

Ship detection using YOLO for the course Digital Processing of Image (Дигитално Процесирање на Слика).

This repo now includes a lightweight `autoresearch`-style workflow adapted from `karpathy/autoresearch`: the idea is to let an AI agent iterate on `train.py`, run short fixed-budget experiments, and keep only changes that improve validation quality.

## Files that matter

- `prepare.py` - fixed utilities for dataset checks, runtime overrides, and metric extraction
- `train.py` - the single training file the agent edits
- `program.md` - instructions for the research agent

## Metric

The primary objective is `metrics/mAP50-95(B)` from Ultralytics validation results. Higher is better.

## Setup

Install dependencies with `uv`, make sure the dataset YAML exists at `ships-aerial-images/data.yaml`, then run:

```bash
uv sync
```

## Training

Run the baseline or any experiment with:

```bash
uv run train.py
```

By default, the training script uses a fixed 5-minute budget through the Ultralytics `time` argument and prints a compact summary at the end so an agent can compare runs automatically.

## Autoresearch loop

1. Create a fresh branch such as `autoresearch/mar24`
2. Read `program.md`
3. Run a baseline with `uv run train.py > run.log 2>&1`
4. Iterate only on `train.py`
5. Log outcomes to `results.tsv`
6. Keep only commits that improve `metrics/mAP50-95(B)`
