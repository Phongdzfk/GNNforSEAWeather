"""YAML config loading with `a.b.c=value` overrides and Kaggle path discovery."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ON_KAGGLE = os.path.isdir("/kaggle/input")
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


class Config(dict):
    """dict with attribute access: cfg.train.lr == cfg['train']['lr']."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(val) if isinstance(val, dict) and not isinstance(val, Config) else val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self))


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def load_config(path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> Config:
    """Load configs/base.yaml, then `path` (which may `inherit:` another file), then overrides.

    Overrides look like ``train.lr=3e-4`` and are parsed with YAML, so numbers,
    booleans and lists work: ``data.train_years=[2020,2021]``.
    """
    def _layer(p) -> dict:
        p = Path(p)
        if not p.exists() and (CONFIG_DIR / p).exists():
            p = CONFIG_DIR / p
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        parent = d.pop("inherit", None)
        if parent and Path(parent).name != "base.yaml":
            return _deep_merge(_layer(parent), d)          # inheritance chains: v3 → v2 → kaggle → base
        return d

    cfg = yaml.safe_load((CONFIG_DIR / "base.yaml").read_text(encoding="utf-8")) or {}
    if path is not None:
        cfg = _deep_merge(cfg, _layer(path))
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return Config(cfg)


# ── Kaggle-aware path discovery ────────────────────────────────────────────

def find_dir(name: str, roots: list[str] | None = None) -> Path:
    """Find a directory called `name` anywhere under the given roots.

    Kaggle mounts notebook outputs at unpredictable depths
    (/kaggle/input/notebooks/<user>/<slug>/...), so we search instead of
    hard-coding paths.
    """
    roots = roots or (["/kaggle/input", "/kaggle/working"] if ON_KAGGLE else ["data"])
    for root in roots:
        direct = Path(root) / name
        if direct.is_dir():
            return direct
        if not Path(root).exists():
            continue
        for dirpath, dirnames, _ in os.walk(root):
            if name in dirnames:
                return Path(dirpath) / name
            # Zarr stores are directories with thousands of chunks — don't descend.
            dirnames[:] = [d for d in dirnames if not d.endswith(".zarr")]
    raise FileNotFoundError(f"Could not find directory '{name}' under {roots}")


def find_file(name: str, roots: list[str] | None = None) -> Path:
    """Find a file by name, or by a path suffix like 'v2/best.pt' (disambiguates several best.pt)."""
    roots = roots or (["/kaggle/input", "/kaggle/working"] if ON_KAGGLE else ["data"])
    suffix = Path(name).parts
    base = suffix[-1]
    for root in roots:
        direct = Path(root) / name
        if direct.is_file():
            return direct
        if not Path(root).exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if base in filenames and Path(dirpath, base).parts[-len(suffix):] == suffix:
                return Path(dirpath) / base
            dirnames[:] = [d for d in dirnames if not d.endswith(".zarr")]
    raise FileNotFoundError(f"Could not find file '{name}' under {roots}")


def resolve_paths(cfg: Config) -> Config:
    """Fill in output/scratch dirs and create them."""
    p = cfg.setdefault("paths", {})
    # NB: base.yaml has `out: null`, so the key exists — setdefault() would keep None and everything
    # landed in ./outputs inside the cloned repo (lost when a Kaggle session ends). Test for falsy instead.
    if not p.get("out"):
        p["out"] = "/kaggle/working" if ON_KAGGLE else "outputs"
    if not p.get("scratch"):
        p["scratch"] = "/kaggle/tmp/wfsea_cache" if ON_KAGGLE else "/tmp/wfsea_cache"
    p["out"] = str(Path(p["out"]).resolve())
    for key in ("out", "scratch"):
        Path(p[key]).mkdir(parents=True, exist_ok=True)
    return cfg
