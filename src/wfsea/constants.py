"""Single source of truth for the domain, variables and channel schema.

Every module (data, model, evaluation, assistant) imports the channel order
from here, so the order can never silently drift between notebooks again.
A checkpoint always stores its own channel list; that stored list wins when a
checkpoint is loaded.
"""
from __future__ import annotations

# ── Spatial domain (0.25° ERA5 grid) ───────────────────────────────────────
LAT_MAX, LAT_MIN = 28.5, -11.0   # ERA5 'area' order is N, W, S, E
LON_MIN, LON_MAX = 92.0, 142.0
GRID_RES = 0.25
N_LAT, N_LON = 159, 201          # 159 × 201 = 31,959 grid nodes
TIME_STEP_HOURS = 6
SYNOPTIC_HOURS = (0, 6, 12, 18)

# ── Splits (by calendar year; no overlap) ──────────────────────────────────
TRAIN_YEARS = (2020, 2021, 2022, 2023, 2024)
VAL_YEARS = (2025,)
TEST_YEARS = (2026,)

# ── Variables ──────────────────────────────────────────────────────────────
SURFACE_VARS = ("t2m", "d2m", "u10", "v10", "msl", "tp")
PRESSURE_VARS = ("z", "q", "u", "v")
PRESSURE_LEVELS = (850, 500)
STATIC_VARS = ("lsm", "geopotential_surf")

# Canonical dynamic channel order: surface first, then (var, level).
DYNAMIC_CHANNELS: tuple[tuple[str, int | None], ...] = tuple(
    [(v, None) for v in SURFACE_VARS]
    + [(v, lev) for v in PRESSURE_VARS for lev in PRESSURE_LEVELS]
)
N_DYN = len(DYNAMIC_CHANNELS)    # 14

UNITS = {
    "t2m": "K", "d2m": "K", "u10": "m/s", "v10": "m/s", "msl": "Pa", "tp": "m",
    "z": "m2/s2", "q": "kg/kg", "u": "m/s", "v": "m/s",
    "lsm": "1", "geopotential_surf": "m2/s2", "Tb": "K",
}


def channel_name(var: str, lev: int | None) -> str:
    """'t2m' or 'z500' — the name used in tables, CSVs and the assistant."""
    return var if lev is None else f"{var}{lev}"


def stats_key(var: str, lev: int | None) -> str:
    """Key format used by the legacy era5_normalization_stats.json."""
    return var if lev is None else f"{var}_{lev}hpa"


CHANNEL_NAMES = tuple(channel_name(v, l) for v, l in DYNAMIC_CHANNELS)


def make_channels(surface_vars=SURFACE_VARS, pressure_vars=PRESSURE_VARS, levels=PRESSURE_LEVELS):
    """Channel list for any variable/level selection (surface first, then var × level)."""
    return tuple([(v, None) for v in surface_vars] + [(v, int(l)) for v in pressure_vars for l in levels])


def schema_from_config(cfg) -> dict:
    """Variables/levels to read from the raw files, taken from cfg['data'] (defaults = 14-channel baseline)."""
    d = cfg.get("data", {}) if hasattr(cfg, "get") else {}
    return {
        "surface_vars": tuple(d.get("surface_vars") or SURFACE_VARS),
        "pressure_vars": tuple(d.get("pressure_vars") or PRESSURE_VARS),
        "levels": tuple(int(x) for x in (d.get("pressure_levels") or PRESSURE_LEVELS)),
        "static_vars": STATIC_VARS,
        "tp_accum_hours": int(d.get("tp_accum_hours", 6)),
        "exclude": tuple(cfg.get("paths", {}).get("era5_exclude") or ()),
    }


def channels_from_config(cfg):
    sc = schema_from_config(cfg)
    return make_channels(sc["surface_vars"], sc["pressure_vars"], sc["levels"])
