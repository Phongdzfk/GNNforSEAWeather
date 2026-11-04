# WeatherForecast-SEA: AI-Driven Regional Weather Forecasting for Southeast Asia
<span id="main"></span>

> **Meteorological Deep Learning Research & Engineering:** A regional Graph Neural Network (GNN) framework forecasting atmospheric dynamics across Southeast Asia at $0.25^\circ$ resolution, fusing ECMWF ERA5 reanalysis and NASA GPM MERGIR infrared satellite observations, optimized for training on accessible hardware.

<div class="toc-card">
  <div class="toc-title">📑 Table of Contents</div>
  <ol>
    <li><a href="#research-context">1. Research Motivation & Regional Context</a></li>
    <li><a href="#core-specifications">2. Core Technical Specifications</a></li>
    <li><a href="#system-architecture">3. End-to-End System Architecture</a></li>
    <li><a href="#experimental-summary">4. Summary of Experimental Results</a></li>
    <li><a href="#directory-structure">5. Project Directory Structure</a></li>
    <li><a href="#quickstart">6. Quickstart Guide</a></li>
  </ol>
</div>

---

## 1. Research Motivation & Regional Context :id=research-context

Classical Numerical Weather Prediction (NWP) systems such as ECMWF IFS or NOAA GFS are remarkable scientific achievements but require thousands of CPU/GPU cores running for hours to generate a single forecast cycle. Recently, global deep learning weather models (DeepMind's GraphCast, Huawei's Pangu-Weather, Nvidia's FourCastNet) demonstrated comparable or superior forecasting accuracy to NWP in **mere seconds on a single GPU**.

However, translating these global models to practical deployment in Southeast Asia faces three major hurdles:
1. **Compute Resource Barrier:** Global architectures demand distributed supercomputing clusters (32 TPU v4 or 64–128 A100 GPUs) for pre-training from scratch, exceeding the computational budget of regional meteorological agencies and universities.
2. **Tropical Monsoonal Specifics:** Southeast Asia ($11^\circ\text{S} - 28.5^\circ\text{N}$, $92^\circ\text{E} - 142^\circ\text{E}$) features complex island–peninsula topography, East Sea (South China Sea) typhoons, active monsoon shifts, and sub-grid convective storms.
3. **Underutilization of Real-Time Observations:** Existing data-driven models train almost exclusively on reanalysis products (ERA5) with a 5-day latency and known tropical precipitation biases.

**WeatherForecast-SEA** directly tackles these challenges: building a regionalized GNN framework that fuses a synoptic "Brain" stream (ERA5) and a convective "Eye" stream (NASA GPM MERGIR satellite), fully trainable on free cloud GPU environments (Kaggle).

---

## 2. Core Technical Specifications :id=core-specifications

<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Geographic Domain</div>
    <div class="kpi-value">11°S – 28.5°N</div>
    <div class="kpi-sub">Longitude: 92°E – 142°E (SEA)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Observation Grid Nodes</div>
    <div class="kpi-value">31,959</div>
    <div class="kpi-sub">Uniform grid 159 x 201 (0.25°)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Latent Coarse Mesh</div>
    <div class="kpi-value">500 nodes</div>
    <div class="kpi-sub">Uniform 2° mesh (kNN k=4)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Forecast Horizon</div>
    <div class="kpi-value">+72 Hours</div>
    <div class="kpi-sub">6h step (12 autoregressive rollouts)</div>
  </div>
</div>

| Parameter | Detailed Specification | Description & Context |
| :--- | :--- | :--- |
| **Horizontal Resolution** | $0.25^\circ \times 0.25^\circ$ (~28 km) | Matches native ERA5 resolution |
| **Temporal Frequency** | 6-hourly intervals ($00, 06, 12, 18\text{ UTC}$) | Standard WMO synoptic reporting hours |
| **Forecast State Variables** | 14 dynamic channels | $t2m, d2m, u10, v10, msl, tp$ (surface); $z, q, u, v$ ($850, 500\text{ hPa}$) |
| **Satellite Radiance** | NASA GPM MERGIR $T_b$ | 4 km infrared brightness temp detecting deep convection |
| **Model Size** | **538,382 parameters** | Compact footprint tailored for single GPU training |
| **Training Time** | ~6 hours per full run | Reduced from 11h via float16 binary memmap arrays |
| **Inference Latency** | < 0.1s per +6h step | Real-time operational rollout on consumer GPUs |

---

## 3. End-to-End System Architecture :id=system-architecture

The model follows a hierarchical **Encoder – Processor – Decoder** paradigm across multi-scale graph spaces:

```mermaid
flowchart LR
    subgraph Inputs [Input State (t-1, t)]
        ERA5["ECMWF ERA5<br/>(14 Atmospheric + Static Fields)"]
        NASA["NASA GPM MERGIR<br/>(Brightness Temp Tb 4km)"]
    end

    subgraph GraphTopology [Hierarchical Graph Topology]
        GridIn["Fine Observation Grid G<br/>(0.25°, 31,959 nodes)"]
        Mesh["Latent Coarse Mesh M<br/>(2.0°, 500 nodes)"]
        GridOut["Fine Target Grid G<br/>(0.25°, 31,959 nodes)"]
    end

    subgraph NeuralCore [Neural Graph Processors]
        Enc["Encoder GNN<br/>(kNN k=4, 127k edges)"]
        Proc["Processor GNN<br/>(4 GNN Layers, 3.7k edges)"]
        Dec["Decoder GNN<br/>(Linear Head + Residual)"]
    end

    ERA5 --> GridIn
    NASA --> GridIn
    GridIn --> Enc --> Mesh
    Mesh --> Proc --> Mesh
    Mesh --> Dec --> GridOut
    GridOut --> Pred["Predicted Residual Delta X<br/>X(t+6h) = X(t) + Delta X"]
```

---

## 4. Summary of Experimental Results :id=experimental-summary

Validated using a rigorous **Multi-Start Autoregressive Rollout** protocol (30 randomized initial conditions sampled across the unseen test year 2026, rolled out 12 steps to +72h):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          SUMMARY OF EXPERIMENTAL BENCHMARKS                            │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ ✅ Outperformed Persistence: 37% average RMSE reduction across all 14 variables.       │
│ ✅ Skillful Medium-Range Horizon: Maintained ACC > 0.6 out to +72h for t2m, msl, z@500.│
│ ✅ Physically Consistent: Oceanic errors remain consistently lower than land/mountain. │
│ ⚠️ U-Net Comparison: U-Net slightly sharper at ultra-short range (<=12h), GNN excels long-term.│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Project Directory Structure :id=directory-structure

```
WeatherForecast/
├── docs/
│   ├── index.html                   # Modern Docsify documentation portal
│   ├── README.md                    # Project overview (this page)
│   ├── architecture.md              # Detailed GNN Grid-Mesh-Grid architecture
│   ├── data-pipeline.md             # ERA5 & satellite ingestion, preprocessing, memmap
│   ├── models-and-training.md       # Loss functions, hyperparameters, checkpointing
│   ├── evaluation-results.md        # Rollout benchmark metrics and error diagnostics
│   ├── literature-and-novel-methods.md # Literature review, 5 gaps & 4 proposed methods
│   ├── requirements.md              # Hardware specs & environment setup
│   ├── ui-dashboard.md              # Operational FastAPI backend & interactive dashboard
│   ├── serve.py                     # Local lightweight documentation server
│   ├── references/                  # Historical project slides & reference reports
│   │   ├── Report DeepLearning.pdf  # Original PTIT deep learning technical report
│   │   └── project_slides.html      # Project presentation slide deck
│   ├── paper/                       # Complete IEEE-style LaTeX research paper
│   │   ├── main.tex                 # Master LaTeX document
│   │   ├── intro.tex                # Introduction section
│   │   ├── related.tex              # Related work section
│   │   ├── method.tex               # Methodology section
│   │   ├── exp.tex                  # Experiments & results section
│   │   ├── conclusion.tex           # Conclusion & future work section
│   │   └── references.bib           # BibTeX academic references
│   └── thesis/                      # PTIT graduation thesis package
│       ├── Khung_do_an.docx         # Formatted Word document adhering to PTIT standards
│       ├── khung-do-an.md           # Full thesis specification in Markdown
│       ├── thesis-outline.md        # Executive thesis outline
│       ├── roadmap.md               # 12-week development roadmap
│       └── create_thesis_docx.py    # Automated OpenXML document compiler
```

---

## 6. Quickstart Guide :id=quickstart

### Virtual Environment Setup with `uv`
```bash
# 1. Create lightning-fast virtual environment with uv
uv venv --python 3.12

# 2. Activate virtual environment
source .venv/bin/activate
# (On Windows: .venv\Scripts\activate)

# 3. Launch documentation server
python3 docs/serve.py
# Open browser at: http://localhost:3000
```
