# Academic Thesis Outline & Structural Framework

> Standard thesis structural outline for Bachelor of Science / Engineering in Information Technology - Major in Artificial Intelligence / Data Science - Posts and Telecommunications Institute of Technology (PTIT).  
> **Topic:** Regional Weather Forecasting in Southeast Asia using Graph Neural Networks Fusing ERA5 Reanalysis and NASA Satellite Imagery.  
> *(Graph Neural Networks for Regional Weather Forecasting in Southeast Asia Fusing ERA5 Reanalysis and NASA Satellite Imagery - WeatherForecast-SEA)*

---

## 📑 Complete Graduation Thesis Layout

```
GRADUATION THESIS REPORT
├── Front Matter
│   ├── Outer Cover & Inner Title Pages (PTIT University Standard)
│   ├── Thesis Task Assignment Sheet & Statement of Authorship
│   ├── Acknowledgements & Thesis Abstracts (Vietnamese & English)
│   ├── Table of Contents, List of Figures, List of Tables
│   └── List of Acronyms & Abbreviations (NWP, GNN, ERA5, GPM, MERGIR, Tb, ACC, RMSE, MSLP)
│
├── Chapter 1: Introduction
│   ├── 1.1 Research Context & Imperative of Data-Driven Atmospheric Forecasting
│   ├── 1.2 Limitations of Numerical Weather Prediction (NWP) & Tropical Challenges in Southeast Asia
│   ├── 1.3 Research Objectives & Domain Scope (11°S–28.5°N, 92°E–142°E, 0.25°)
│   ├── 1.4 Scientific Contributions & Practical Significance
│   └── 1.5 Thesis Structure and Organization
│
├── Chapter 2: Literature Review & Theoretical Foundations
│   ├── 2.1 Overview of Classical Numerical Weather Prediction (ECMWF IFS, GFS, WRF)
│   ├── 2.2 Rise of Global Atmospheric Deep Learning (FourCastNet, Pangu-Weather, GraphCast, FuXi)
│   ├── 2.3 Mathematical Foundations of Graph Neural Networks (GNNs) on Non-Euclidean Manifolds
│   ├── 2.4 Meteorological Data: ECMWF ERA5 Reanalysis & NASA GPM MERGIR Infrared Remote Sensing
│   ├── 2.5 Analysis of Five Core Research Gaps & Compute Resource Constraints
│   └── 2.6 Technology Stack Evaluation and Selection
│
├── Chapter 3: Proposed Methodology & Architectural Design
│   ├── 3.1 Residual Autoregressive Forecasting Formulation
│   ├── 3.2 Hierarchical Multi-Scale Graph Topology (Grid-to-Mesh, Mesh-to-Mesh, Mesh-to-Grid)
│   ├── 3.3 Message Passing Dynamics & Encoder-Processor-Decoder Block Design
│   ├── 3.4 Multi-Source Fusion: Atmospheric "Brain" (ERA5) & Convective "Eye" (NASA Satellite)
│   ├── 3.5 Objective Optimization: Latitude-Weighted RMSE for Spherical Distortion Correction
│   └── 3.6 Proposed Methodological Advances: Multi-scale Meshes, Solar Forcing, Rollout & Cross-Attention
│
├── Chapter 4: System Analysis & Engineering Architecture
│   ├── 4.1 Functional Requirements & Technical Specifications
│   ├── 4.2 Overall System Architecture & Data Flow Diagrams (DFD)
│   ├── 4.3 Automated Data Ingestion Pipeline (ECMWF CDS API & NASA GES DISC)
│   ├── 4.4 Preprocessing Layer, Anti-Leakage Normalization & Float16 Memmap Engineering
│   └── 4.5 Production Deployment Architecture (ONNX Runtime, FastAPI & Streamlit Map Dashboard)
│
├── Chapter 5: System Implementation & Software Realization
│   ├── 5.1 Experimental Setup & Codebase Directory Organization
│   ├── 5.2 Automated Ingestion & Harmonization Pipeline Implementation
│   ├── 5.3 GNN Neural Core Implementation (PyTorch Geometric, SiLU, LayerNorm, 538K Params)
│   ├── 5.4 Training Optimization Pipeline (AdamW, Cosine Annealing, AMP, DataParallel)
│   └── 5.5 Interactive Weather Dashboard & Geospatial Visualization Interface
│
├── Chapter 6: Empirical Evaluation & In-Depth Discussion
│   ├── 6.1 Experimental Protocol & Data Split (Train 2020–2024, Val 2025, Test 2026)
│   ├── 6.2 Quantitative Benchmark Against Baselines (Persistence, Climatology, U-Net CNN)
│   ├── 6.3 Multi-Step Rollout Skill Evaluation (+6h to +72h, ACC & RMSE Tracking)
│   ├── 6.4 Geographic Spatial Error Stratification (Ocean, Plains, Inland, Mountainous)
│   ├── 6.5 Diagnostic Meteorological Error Analysis (Mountain Cold Bias, Diurnal Sawtooth, Blurring)
│   └── 6.6 Ablation Studies & Memory-Mapped Array Throughput Verification
│
├── Chapter 7: Conclusions & Future Directions
│   ├── 7.1 Summary of Research Accomplishments
│   ├── 7.2 Scientific Contributions & Practical Implications for Southeast Asia
│   ├── 7.3 Identified System Limitations
│   └── 7.4 Future Research Horizons (3D Vertical Stacks, Typhoon Tracking, Real-time Operations)
│
└── Back Matter
    ├── References (IEEE Citation Style, min. 30 peer-reviewed scholarly works)
    ├── Appendix A: Exhaustive Atmospheric Variable Catalogue & Normalization Parameters
    ├── Appendix B: Detailed Error Tables Across All 12 Autoregressive Steps (+72h)
    └── Appendix C: System Installation & Deployment Operations Guide
```
