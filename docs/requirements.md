# System Requirements & Setup Guide

> Hardware specifications, Kaggle environment constraints, Python dependency manifest, and step-by-step reproducibility instructions.

<div class="toc-card">
  <div class="toc-title">📑 Table of Contents</div>
  <ol>
    <li><a href="#hardware-requirements">1. Hardware Configuration Requirements</a></li>
    <li><a href="#software-dependencies">2. Software Stack & Core Dependencies</a></li>
    <li>
      <a href="#environment-setup">3. Environment Setup Walkthrough</a>
      <ol>
        <li><a href="#step-1">Step 1: Virtual Environment Setup (uv / venv)</a></li>
        <li><a href="#step-2">Step 2: PyTorch & Meteorological Library Installation</a></li>
        <li><a href="#step-3">Step 3: ECMWF CDS API Account Configuration</a></li>
        <li><a href="#step-4">Step 4: Pipeline Sanity Check</a></li>
      </ol>
    </li>
  </ol>
</div>

---

## 1. Hardware Configuration Requirements :id=hardware-requirements

WeatherForecast-SEA is engineered specifically to operate fully within the resource limits of free cloud GPU environments:

| Hardware Component | Minimum Baseline (Kaggle Standard) | Recommended Configuration (Research Workstation) |
| :--- | :--- | :--- |
| **GPU VRAM** | 1x 16 GB (Nvidia Tesla T4 or P100) | 2x 16 GB or 1x 24 GB (RTX 4090 / A10) |
| **CPU Architecture** | 2–4 vCPU x86_64 | 8–16 Core CPU |
| **System RAM** | 13 GB | 32 GB – 64 GB |
| **Disk Storage** | 30 GB SSD (Kaggle ephemeral storage) | 200 GB NVMe SSD |
| **Operating System** | Linux (Ubuntu 20.04 / Debian 11) | Linux / macOS / Windows WSL2 |

---

## 2. Software Stack & Core Dependencies :id=software-dependencies

The project is built on the standard open-source Python scientific computing ecosystem:

```
┌────────────────────────────────────────────────────────┐
│               CORE DEPENDENCY MANIFEST                 │
├────────────────────────────────────────────────────────┤
│ • torch >= 2.0.0 (CUDA 11.8 / 12.1 and AMP support)    │
│ • torch-geometric >= 2.3.0 (Sparse graph operations)   │
│ • xarray >= 2023.1.0 & netCDF4 >= 1.6.0 (Climate data) │
│ • zarr >= 2.14.0 (Multi-dimensional compressed arrays) │
│ • cdsapi >= 0.6.1 (Automated ECMWF CDS queries)        │
│ • cartopy >= 0.21.0 & matplotlib >= 3.7.0 (Geomapping) │
│ • numpy >= 1.23.0 & scipy >= 1.10.0 (Numeric analysis) │
│ • scikit-learn >= 1.2.0 (kNN graph construction)       │
└────────────────────────────────────────────────────────┘
```

---

## 3. Environment Setup Walkthrough :id=environment-setup

### Step 1: Virtual Environment Setup (Recommended: `uv`) :id=step-1
```bash
# Option 1: Fast setup using uv (10-100x faster package resolution)
uv venv --python 3.12
source .venv/bin/activate
# (On Windows: .venv\Scripts\activate)

# Option 2: Traditional venv
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Libraries :id=step-2
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install xarray netCDF4 zarr cdsapi cartopy matplotlib scikit-learn
```

### Step 3: Configure ECMWF CDS API Account :id=step-3
Create `~/.cdsapirc` populated with your user credentials from the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/):
```ini
url: https://cds.climate.copernicus.eu/api/v2
key: {YOUR_UID}:{YOUR_API_KEY}
```

### Step 4: Pipeline Sanity Check :id=step-4
```bash
python3 -c "import torch, torch_geometric, xarray; print('Environment Ready! CUDA available:', torch.cuda.is_available())"
```
