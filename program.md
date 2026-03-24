# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar24`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed runtime utilities, summary extraction, and dataset checks. Do not modify.
   - `train.py` — the file you modify. Model choice, optimizer, hyperparameters, image size, and training loop entrypoint all live here.
4. **Verify data exists**: Check that `ships-aerial-images/data.yaml` exists, or that `YOLO_DATA` points to a valid dataset YAML. If not, tell the human to add the dataset first.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs through `uv run train.py`.

The training script uses a **fixed 5-minute time budget** through Ultralytics' `time` argument, so experiments are approximately comparable and always short enough to iterate quickly.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything there is fair game: model size, model weights, image size, batch size, optimizer, learning rate schedule, augmentation knobs, worker count, freeze settings, and similar training parameters.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness outside the normal Ultralytics validation outputs produced by the training run.

**The goal is simple: get the highest `metrics/mAP50-95(B)`.** Higher is better. Since the time budget is fixed, the core job is to find the best-performing experiment under that fixed budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains, but avoid ideas that blow up memory or make experiments fragile.

**Simplicity criterion**: All else being equal, simpler is better. A tiny gain that adds ugly complexity is usually not worth it. Removing complexity while keeping equal or better quality is a win.

**The first run**: Your very first run should always be the baseline, so run the training script as is before changing anything.

## Output format

Once the script finishes it prints a summary like this:

```
---
fitness_key:       metrics/mAP50-95(B)
fitness:           0.612345
training_seconds:  300.1
total_seconds:     300.1
peak_vram_mb:      8240.5
precision:         0.801234
recall:            0.745678
map50:             0.822222
map50_95:          0.612345
epoch:             18
```

You can extract the key metric from the log file with:

```
grep "^fitness:\|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break descriptions).

The TSV has a header row and 5 columns:

```
commit	metric	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `metrics/mAP50-95(B)` achieved (e.g. 0.612345) — use `0.000000` for crashes
3. peak memory in GB, round to `.1f` (divide `peak_vram_mb` by 1024) — use `0.0` for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what the experiment tried

Example:

```
commit	metric	memory_gb	status	description
a1b2c3d	0.612345	8.1	keep	baseline yolo11l 640 adamw
b2c3d4e	0.618901	9.4	keep	increase image size to 768
c3d4e5f	0.605100	7.9	discard	reduce batch and switch optimizer
d4e5f6g	0.000000	0.0	crash	batch too large caused OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar24`).

LOOP FOREVER:

1. Look at the git state: the current branch and commit.
2. Tune `train.py` with one experimental idea.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read out the results: `grep "^fitness:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Read the traceback from `run.log`, attempt a fix if it is easy, otherwise mark it as a crash and move on.
7. Record the result in `results.tsv` (do not commit `results.tsv`; leave it untracked).
8. If the metric improved, keep the commit.
9. If the metric is equal or worse, reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. Advance the branch only with improvements.

**Timeout**: Each experiment should take about 5 minutes total, plus a small amount of overhead. If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, bad hyperparameters, a typo, etc.), use judgment. If it is something dumb and easy to fix, fix it and re-run. If the idea is fundamentally broken, log it as `crash` and move on.

**NEVER STOP**: Once the experiment loop has begun, do not pause to ask whether you should continue. Keep going until the human interrupts you.
