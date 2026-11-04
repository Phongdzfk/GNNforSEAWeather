# CLAUDE.md - WeatherForecast-SEA Assistant Guidelines

This document provides context, core commands, code styles, and operational rules for Anthropic Claude (Claude Code CLI and Claude Desktop) when working on the **WeatherForecast-SEA** codebase.

---

## 🎯 Project Overview

- **Project:** WeatherForecast-SEA (*Graph Neural Networks for Regional Weather Forecasting in Southeast Asia Fusing ERA5 Reanalysis and NASA Satellite Imagery*).
- **Core Domain:** $11^\circ\text{S} - 28.5^\circ\text{N}$, $92^\circ\text{E} - 142^\circ\text{E}$ (159 $\times$ 201 = 31,959 observation nodes at $0.25^\circ \times 0.25^\circ$).
- **Core Model:** Encoder–Processor–Decoder GNN operating on a multi-scale Grid-Mesh-Grid graph (500-node icosahedral coarse mesh, 538K parameters).
- **Data Ingestion:** ECMWF ERA5 (14 prognostic atmospheric fields + static fields) combined with NASA GPM MERGIR (4km infrared brightness temperature $T_b$).
- **Key Engineering Solution:** One-time Float16 binary memmap pre-decode (`np.memmap`) reducing training time from 11h to 6h and eliminating memory spikes on Kaggle GPU instances.

---

## ⌨️ Common Commands & Workflows

### Environment & Package Management
```bash
# Initialize and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
uv pip install torch_geometric cdsapi zarr netcdf4 fastapi streamlit pydeck
```

### Local Documentation Server (Docsify)
```bash
# Start documentation server (runs on port 3000 or increments if occupied)
python3 docs/serve.py 3000

# Access docs at http://localhost:3000
```

### Thesis & Paper Compilation
```bash
# Rebuild graduation thesis (.docx) from markdown:
python3 docs/thesis/create_thesis_docx.py

# Compile IEEE LaTeX research paper:
cd docs/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 📐 Strict Coding & Architectural Rules

1. **Strict Data Leakage Discipline:**
   - Normalization statistics (Mean $\mu$, Std $\sigma$) must **always** be computed strictly on the training set (2020–2024).
   - Never recompute or adapt statistics on validation (2025) or test (2026) splits.

2. **Spherical Geometric Correction:**
   - Always implement **Latitude-Weighted RMSE** loss ($w_i \propto \cos \phi_i$) for model training and metric tracking to compensate for meridian convergence.

3. **Residual Prediction Standard:**
   - Model outputs are residual state increments $\Delta \mathbf{X}(t) = \mathbf{X}(t+1) - \mathbf{X}(t)$. Next-step predictions are calculated additively.

4. **Documentation Language & Style:**
   - All documentation in `docs/` must be maintained in **professional academic English** (except `docs/references/` which retains raw reference literature).
   - In-page table of contents cards (`.toc-card`) and custom heading anchor IDs (`:id=...`) must be preserved and kept in sync.

---

## 📂 Key File Map

| Path | Purpose |
| :--- | :--- |
| `docs/index.html` | Docsify documentation entry point with MathJax and Mermaid hooks |
| `docs/README.md` | Core documentation overview, specifications, and architecture diagram |
| `docs/architecture.md` | Detailed GNN graph topology and parameter budget breakdown |
| `docs/data-pipeline.md` | Data retrieval, anti-leakage normalization, and Float16 memmap |
| `docs/models-and-training.md` | Training hyperparameters, latitude-weighted loss, AMP, and checkpointing |
| `docs/evaluation-results.md` | 72h rollout benchmarks, RMSE/ACC tables, and meteorological error analysis |
| `docs/literature-and-novel-methods.md` | Full literature review, 5 research gaps, and 4 novel proposed methods |
| `docs/ui-dashboard.md` | Microservices deployment (FastAPI, ONNX Runtime, Streamlit Mapbox) |
| `docs/thesis/khung-do-an.md` | Comprehensive PTIT undergraduate thesis framework (English) |
| `docs/thesis/create_thesis_docx.py`| Python script generating `Khung_do_an.docx` via OpenXML |
| `docs/paper/` | IEEE journal LaTeX paper suite (`main.tex`, `intro.tex`, `method.tex`, etc.) |

---

## 🧩 Source Code (`src/wfsea/`)

- The notebooks of the course project were ported into the `wfsea` package; Kaggle notebooks in `kaggle/notebooks/` only clone the repo and call `wfsea <command>` (materialize, manifest, climatology, graph, train, evaluate, export-forecast).
- Channel order, domain and variables live only in `src/wfsea/constants.py`; hyper-parameters only in `configs/*.yaml` (override with `key=value`).
- Run `pytest` after every change: tests build synthetic data in the exact legacy Kaggle Zarr layout and run train → evaluate → export end to end. `tests/test_model.py` pins the 538,382-parameter / 500-node / 127,836 + 3,734-edge baseline.
- Raw ERA5 + MERGIR are available for 2015–2026. `configs/base.yaml` trains on 2015–2024 (val 2025, test 2026) straight from the raw NetCDF (`data.zarr_layout: raw`, stats recomputed on the new train split); `configs/legacy_2020.yaml` reproduces the course-project setup on the old Zarr stores. The 2015–2024 split still needs advisor sign-off before the docs above (which say 2020–2024) are updated.
- Known corrections vs. the old notebooks (see `project/KICKOFF.md`): train-split climatology (not test mean), `eval.tb_policy` (no future satellite by default), 6-hour continuity check across year boundaries, the mesh is a regular 2° lat-lon lattice (not icosahedral), the 38 input channels contain no temporal features.

