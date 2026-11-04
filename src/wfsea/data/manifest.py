"""Data manifest: what data exists, where it came from, and a checksum per file.

Week-1 deliverable for Phong. Also answers the review question
"why does the 2026 test split have only 587 samples?" — see time_end / gaps.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def build_manifest(cache_dirs: dict[str, str | Path], out_path: str | Path, checksum: bool = True) -> dict:
    """Summarise materialised splits (train/val/test) into one JSON manifest."""
    from wfsea.data.memmap import open_cache
    manifest = {"splits": {}}
    for split, d in cache_dirs.items():
        dirs = [Path(x) for x in (d if isinstance(d, (list, tuple)) else [d])]
        cache = open_cache(dirs)
        parts = []
        for di in dirs:
            meta = json.loads((di / "meta.json").read_text())
            part = {k: meta.get(k) for k in ("years", "layout", "T", "G", "C", "time_start", "time_end", "gaps",
                                            "nan_count", "tb_coverage", "fp16_relative_rmse", "norm_stats_years")}
            part["dir"] = di.name
            if checksum:
                part["sha256"] = {f.name: sha256_file(f) for f in sorted(di.iterdir()) if f.name != "meta.json"}
            parts.append(part)
        entry = {"years": cache.meta["years"], "T": int(cache.T), "C": int(cache.C),
                 "time_start": str(cache.times[0])[:16], "time_end": str(cache.times[-1])[:16],
                 "n_valid_samples": int(len(cache.valid_starts())), "channels": [list(c) for c in cache.channels],
                 "gaps": sum((p["gaps"] for p in parts), []), "parts": parts}
        if len(parts) == 1:   # flat keys kept for the single-part case
            entry.update({k: parts[0][k] for k in ("nan_count", "tb_coverage", "fp16_relative_rmse")})
            if checksum:
                entry["sha256"] = parts[0]["sha256"]
        manifest["splits"][split] = entry
    # Anti-leakage check: no timestamp may appear in two splits.
    spans = {s: (e["time_start"], e["time_end"]) for s, e in manifest["splits"].items()}
    names = list(spans)
    overlaps = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                if not (spans[a][1] < spans[b][0] or spans[b][1] < spans[a][0])]
    manifest["split_overlap"] = overlaps
    Path(out_path).write_text(json.dumps(manifest, indent=2))
    return manifest
