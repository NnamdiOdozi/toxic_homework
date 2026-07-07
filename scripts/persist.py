from __future__ import annotations

import datetime as dt
import gc
import json
from pathlib import Path
from typing import Any, Sequence


try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None


DEFAULT_VAR_NAMES = [
    "base_greedy",
    "base_sampled",
    "sft_greedy",
    "sft_sampled",
    "dpo_greedy",
    "dpo_sampled",
    "rm_metrics",
    "raw_grpo_greedy",
    "raw_grpo_sampled",
    "raw_rm_grpo_greedy",
    "raw_rm_grpo_sampled",
    "reward_values",
]


def cleanup_cuda() -> None:
    """Free Python references and clear CUDA cache after large model stages."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _jsonable(x: Any) -> Any:
    """Convert common notebook/PyTorch objects into JSON-safe values."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    if isinstance(x, Path):
        return str(x)

    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.numel() == 1:
            return x.item()
        return x.tolist()

    if np is not None:
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()

    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]

    return str(x)


def _read_json(path: Path, default: Any) -> Any:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=2, ensure_ascii=False)
    tmp.replace(path)


class Persistence:
    def __init__(self, outputs: Path, data_dir: Path, ckpt_dir: Path) -> None:
        self.outputs = Path(outputs)
        self.data_dir = Path(data_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.samples_dir = self.outputs / "samples"
        self.metrics_path = self.outputs / "metrics.json"
        self.run_notes_path = self.outputs / "run_notes.md"

        for d in (self.outputs, self.data_dir, self.ckpt_dir, self.samples_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.metrics = _read_json(self.metrics_path, {})

    def get_metric(self, path: str, default: Any = None) -> Any:
        cur = self.metrics
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def record_metric(self, path: str, value: Any) -> None:
        cur = self.metrics
        parts = path.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = _jsonable(value)
        self.metrics["updated_at_utc"] = (
            dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
        )
        _write_json_atomic(self.metrics_path, self.metrics)
        print(f"saved metric: {path} -> {self.metrics_path}")

    def persist_vars(self, namespace: dict, names: Sequence[str] = DEFAULT_VAR_NAMES) -> None:
        saved = []
        for name in names:
            if name in namespace:
                self.record_metric(f"variables.{name}", namespace[name])
                saved.append(name)
        print("persisted variables:", saved or "none found")

    def restore_vars(self, namespace: dict, names: Sequence[str] = DEFAULT_VAR_NAMES) -> None:
        restored = []
        for name in names:
            value = self.get_metric(f"variables.{name}")
            if value is not None:
                namespace[name] = value
                restored.append(name)
        print("restored variables:", restored or "none found")

    def append_run_note(self, stage: str, text: str, **fields: Any) -> None:
        ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

        with self.run_notes_path.open("a", encoding="utf-8") as f:
            f.write(f"\n\n## {stage} — {ts}\n\n")
            if fields:
                f.write("```json\n")
                f.write(json.dumps(_jsonable(fields), indent=2, ensure_ascii=False))
                f.write("\n```\n\n")
            f.write(text.strip() + "\n")

        print(f"appended note: {self.run_notes_path}")

    def save_jsonl(self, name_or_path: str | Path, rows: Sequence[dict]) -> Path:
        path = Path(name_or_path)
        if not path.is_absolute():
            path = self.samples_dir / path

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")

        print(f"saved jsonl: {path}")
        return path


def init_persistence(outputs: Path, data_dir: Path, ckpt_dir: Path) -> Persistence:
    return Persistence(outputs, data_dir, ckpt_dir)