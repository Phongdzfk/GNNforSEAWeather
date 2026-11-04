# Gemini & Antigravity Assistant Guide: WeatherForecast-SEA

This file defines the project instructions, architectural rules, and operational guidelines for Gemini and Antigravity agents working in the **WeatherForecast-SEA** repository.

---

## 🌦️ 1. Project Context & Current State

- **Repository:** `WeatherForecast` (Project Name: **WeatherForecast-SEA**)
- **Topic:** Regional Weather Forecasting in Southeast Asia using Graph Neural Networks Fusing ERA5 Reanalysis and NASA Satellite Imagery.
- **Geographic Domain:** $11^\circ\text{S} - 28.5^\circ\text{N}$, $92^\circ\text{E} - 142^\circ\text{E}$ (159 $\times$ 201 = 31,959 equirectangular grid nodes at $0.25^\circ \times 0.25^\circ$ horizontal resolution).
- **Core Architecture:** GraphCast-style **Encoder – Processor – Decoder** operating over a multi-scale bipartite graph:
  - Observation Grid $G$ ($0.25^\circ$, 31,959 nodes).
  - Coarse Latent Mesh $M$ ($2.0^\circ$ icosahedron, 500 nodes).
  - Edge sets: Grid-to-Mesh ($E_{G \to M}$, kNN $k=4$, 127,836 edges), Mesh-to-Mesh ($E_{M \to M}$, 8-neighborhood, 3,734 edges), Mesh-to-Grid ($E_{M \to G}$).
  - Parameter Budget: 538,382 parameters (trainable on dual-T4 / P100 GPUs within Kaggle constraints).
- **Data Sources:**
  - **ECMWF ERA5:** Surface fields (`t2m`, `d2m`, `u10`, `v10`, `msl`, `tp`) + Upper-air fields at 850 & 500 hPa (`z`, `q`, `u`, `v`) + Static fields (`lsm`, `geopotential_surf`).
  - **NASA GPM MERGIR:** 4km cloud-top brightness temperature ($T_b$) bilinearly regridded to $0.25^\circ$.
- **High-Performance Memory Engineering:** One-time Float16 binary memmap arrays (`.npy` loaded via `np.memmap`) reducing training wall-clock time from 11h to 6h and eliminating OOM errors.
- **Evaluation Baseline:** Multi-start 12-step autoregressive rollout (+6h to +72h) on 2026 test set. Beats Persistence by 37% RMSE and outperforms convolutional U-Net at $+72h$.

---

## 🛠️ 2. Essential Commands & Workflows

### Environment Setup
```bash
# Using uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt  # when available or install PyTorch / PyG

# Alternatively using conda
conda create -n weatherforecast python=3.10 -y
conda activate weatherforecast
```

### Documentation Preview (Docsify)
```bash
# Start local docs server on port 3000 (auto-increments if port is busy)
python3 docs/serve.py 3000
# Preview in browser at: http://localhost:3000
```

### Thesis & Paper Artifacts
```bash
# Regenerate graduation thesis docx from markdown source:
python3 docs/thesis/create_thesis_docx.py

# Compile IEEE LaTeX research paper:
cd docs/paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## 📐 3. Architectural Invariants & Rules

1. **Anti-Leakage Normalization:**
   - All statistical normalization parameters (Mean $\mu$ and Standard Deviation $\sigma$) must **strictly** be computed on the training split (2020–2024).
   - Validation (2025) and testing splits (2026) must use the frozen $\mu_{train}, \sigma_{train}$. Never calculate batch statistics across temporal boundaries.

2. **Spherical Area Loss Weighting:**
   - Always use **Latitude-Weighted RMSE** loss with normalized spherical metric weights $w_i = \frac{\cos(\phi_i)}{\frac{1}{N}\sum_j \cos(\phi_j)}$.
   - Do NOT use unweighted Euclidean MSE across equirectangular grids.

3. **Residual Learning Formulation:**
   - The neural network predicts single-step residuals: $\Delta \mathbf{X}(t) = \mathbf{X}(t+1) - \mathbf{X}(t)$.
   - Next-step predictions are obtained via additive recurrence: $\hat{\mathbf{X}}(t+1) = \mathbf{X}(t) + \Delta \mathbf{X}(t)$.

4. **Multi-Scale Graph Invariants:**
   - Edge topologies are fixed geometric structures computed once during graph initialization.
   - Embed 3D Cartesian coordinates $(x, y, z)$ on unit sphere and relative displacements $(\Delta x, \Delta y, \Delta z, \|\Delta \mathbf{x}\|)$ in edge attribute tensors.

5. **Documentation Integrity:**
   - All public documentation files in `docs/` must remain in **English** (except `docs/references/` which preserves raw source literature).
   - In-page table of contents cards (`.toc-card`) and heading custom IDs (`:id=...`) must remain consistent.

---

## 🔬 4. Research Gaps & Novel Methodology Roadmap

When developing next-generation enhancements, align with the 4 proposed methods:
1. **Hierarchical Cross-Mesh GNN:** Multi-level icosahedral meshes ($1^\circ, 2^\circ, 4^\circ$) with bidirectional cross-scale message passing to eliminate short-range spatial blurring.
2. **Physics-Informed Solar Forcing:** Analytical Top-of-Atmosphere (TOA) insolation + Solar Zenith Angle ($\theta_z$) + harmonic temporal positional encodings to eliminate continental diurnal sawtooth errors.
3. **Multi-Step Rollout & Spectral Loss:** 1–4 step autoregressive training with Backpropagation Through Time (BPTT) and 2D Fourier spectral loss $\| \mathcal{F}(X) - \mathcal{F}(\hat{X}) \|_1$.
4. **Cross-Attentive Satellite Radiance Fusion:** Cross-attention between satellite infrared cloud features ($4\text{km}$) and synoptic atmospheric pressure levels.
