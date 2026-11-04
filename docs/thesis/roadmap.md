# 12-Week Graduation Thesis Implementation Roadmap

> Sprint-based structured implementation timeline for the graduation thesis "Regional Weather Forecasting in Southeast Asia using Graph Neural Networks Fusing ERA5 Reanalysis and NASA Satellite Imagery" (WeatherForecast-SEA).

---

## 🗓️ 12-Week Milestone Overview

| Phase | Weeks | Sprint Focus | Primary Deliverables |
| :--- | :--- | :--- | :--- |
| **P1: Research & Data** | Weeks 1 – 3 | Literature review of NWP & AI, ERA5 & NASA GPM ingestion pipeline, anti-leakage normalization | Curated Zarr/Memmap dataset (2020–2026), Automated Data Ingestion Module |
| **P2: Architecture & Modeling** | Weeks 4 – 6 | Multi-scale graph topology (kNN k=4, 2° Mesh), Encoder-Processor-Decoder GNN core, latitude-weighted loss | Full PyG model source code, trained WeatherForecast-SEA checkpoint |
| **P3: System Optimization** | Weeks 7 – 8 | Memory-mapped I/O pipeline (float16 memmap pre-decode, AMP, DataParallel), baseline benchmarks (U-Net, Persistence) | Optimized Kaggle GPU training engine (6h/run), quantitative U-Net baseline |
| **P4: Evaluation & UI** | Weeks 9 – 10 | 30-start autoregressive rollout (+72h ACC, RMSE, spectral analysis), interactive Mapbox/Streamlit dashboard | Comprehensive evaluation report, Interactive Web Dashboard |
| **P5: Writing & Defense** | Weeks 11 – 12 | Finalizing academic thesis monograph, LaTeX paper submission draft, presentation slide deck & mock defense | Complete thesis monograph (.docx / PDF), defense slide deck, LaTeX paper |

---

## 📅 Detailed Weekly Task Breakdown

### Week 1: Project Kickoff & Literature Review
- Study core literature: NWP foundations (ECMWF IFS), AI weather models (GraphCast, Pangu-Weather, FourCastNet, FuXi), and specific tropical meteorological challenges across Southeast Asia.
- Define spatial bounding coordinates ($11^\circ\text{S} - 28.5^\circ\text{N}$, $92^\circ\text{E} - 142^\circ\text{E}$, $0.25^\circ \times 0.25^\circ$ resolution, 31,959 grid nodes).
- Finalize detailed project outline and align methodology with Academic Advisor.

### Week 2: ECMWF ERA5 Atmospheric Reanalysis Ingestion
- Configure Copernicus Climate Data Store API credentials (`cdsapi`).
- Develop automated retrieval scripts for 14 atmospheric variables over 2020–2026: surface fields ($t2m, d2m, u10, v10, msl, tp$), upper-air pressure levels at 850 hPa and 500 hPa ($z, q, u, v$), and invariant fields ($lsm, z_{surf}$).
- Clip data to the Southeast Asia bounding box and store in NetCDF/Zarr formats.

### Week 3: NASA GPM MERGIR Satellite Ingestion & Preprocessing
- Implement automated downloader for cloud-top infrared brightness temperature ($T_b$) from NASA GES DISC.
- Synchronize timestamps to synoptic 6-hour intervals ($00, 06, 12, 18\text{ UTC}$).
- Perform bilinear interpolation from the native $4\text{ km}$ satellite grid to the atmospheric $0.25^\circ$ coordinate grid.
- Enforce strict Anti-Leakage protocol: calculate mean and standard deviation statistics exclusively on the training split (2020–2024).

### Week 4: Multi-Scale Graph Topology Design (Grid–Mesh–Grid)
- Construct coarse latent icosahedral mesh at $2^\circ \times 2^\circ$ resolution (500 nodes) spanning Southeast Asia.
- Implement $k$-Nearest Neighbors ($k$-NN, $k=4$) to instantiate 127,836 directed Grid-to-Mesh edges ($E_{G \to M}$).
- Generate 3,734 internal bidirectional Mesh-to-Mesh edges ($E_{M \to M}$) based on 8-neighborhood connectivity and decode edges for Mesh-to-Grid ($E_{M \to G}$).
- Embed spherical geometric spatial attributes onto graph edge feature tensors.

### Week 5: Graph Neural Network Implementation (Encoder – Processor – Decoder)
- Code the Encoder module mapping 38 input features into a 128-dimensional latent space.
- Construct the Processor comprising 4 recurrent GNN layers with SiLU activations, LayerNorm, and residual message-passing connections.
- Implement Decoder module and linear head projecting latent states to single-step residual differences $\Delta \mathbf{X} = \mathbf{X}(t+1) - \mathbf{X}(t)$.
- Constrain total model capacity strictly within 538K parameters for commodity hardware execution.

### Week 6: Objective Formulation & Baseline Benchmarking
- Implement latitude-weighted Root Mean Square Error loss with spherical geometric scaling weights $w_i \propto \cos(\phi_i)$.
- Implement classical reference baselines: Persistence (frozen atmospheric state) and Climatology (historical day-of-year mean).
- Construct deep learning convolutional baseline: standard 4-stage U-Net on the $0.25^\circ$ grid with matched parameter count (~0.6M parameters).

### Week 7: Memory Optimization & Throughput Acceleration (Float16 Binary Memmap)
- Profile and isolate Zarr chunk decompression I/O bottlenecks in Kaggle environments.
- Engineer a one-time pre-decode script saving flattened binary `.npy` arrays loaded via zero-overhead `np.memmap`.
- Integrate Automatic Mixed Precision (`torch.cuda.amp`) and multi-GPU parallelism (DataParallel).
- Verify training wall-clock time reduction from 11 hours to 6 hours per job with zero out-of-memory crashes.

### Week 8: Production Model Training & Checkpointing
- Execute full 50-epoch training run utilizing AdamW optimizer with Cosine Annealing learning rate schedule.
- Monitor validation loss on 2025 data, applying early stopping with patience = 7.
- Save best model checkpoint `weatherforecast_sea_best.pt` (~2.2 MB).

### Week 9: Multi-Start Autoregressive Rollout & Evaluation
- Implement 12-step recursive autoregressive rollout (+6h to +72h) initialized at 30 randomly sampled timestamps from 2026.
- Compute multi-lead RMSE and ACC metric trajectories across all 14 atmospheric variables.
- Quantify geographic error stratification: Ocean vs Plains vs Inland vs Complex Orography.
- Perform 2D spatial Fast Fourier Transform (FFT) analysis to diagnose high-frequency spectral blurring.

### Week 10: Operational Web Dashboard Development
- Export trained PyTorch weights to the open ONNX Runtime standard for sub-100ms real-time inference.
- Build FastAPI backend microservice exposing endpoints for spatial grid rasters and coordinate point meteograms.
- Develop Streamlit web application with PyDeck/Mapbox rendering dynamic particle wind streamlines, thermal overlays, and time scrubbers.

### Week 11: Academic Thesis Monograph & LaTeX Paper Writing
- Author all 7 chapters of the graduation thesis following PTIT university publication standards.
- Prepare complete LaTeX research paper (`docs/paper/`) including `intro`, `related`, `method`, `exp`, `conclusion`, and IEEE bibliography (`references.bib`).
- Perform rigorous cross-verification of numerical metrics, mathematical formulas, and architectural flowcharts.

### Week 12: Defense Slide Deck Preparation & Rehearsals
- Create defense presentation slide deck (25–30 visual, concise slides).
- Prepare live interactive dashboard demonstration showcasing tropical cyclone tracking scenarios over the South China Sea.
- Rehearse technical defenses addressing data leakage, spectral dissipation, mountainous cold biases, and comparisons against WRF numerical modeling.
