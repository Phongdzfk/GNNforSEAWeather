# IEEE Research Paper Framework: WeatherForecast-SEA

> Full academic manuscript structure written in LaTeX adhering to **IEEE Transactions on Geoscience and Remote Sensing** and **IEEE Transactions on Neural Networks and Learning Systems** standards.

<div class="toc-card">
  <div class="toc-title">📑 Table of Contents</div>
  <ol>
    <li><a href="#paper-overview">1. Paper Abstract & Metadata</a></li>
    <li><a href="#modular-latex-structure">2. Modular LaTeX Source Code Structure</a></li>
    <li><a href="#compilation-instructions">3. Compilation Runbook</a></li>
    <li><a href="#bibtex-citation">4. IEEE BibTeX Citation</a></li>
  </ol>
</div>

---

## 1. Paper Abstract & Metadata :id=paper-overview

- **Title:** *WeatherForecast-SEA: Regional Graph Neural Networks for Weather Forecasting in Southeast Asia Fusing ERA5 Reanalysis and NASA Satellite Imagery*
- **Authors:** Dang Tuan Phong, Tran Minh Hieu, and Nguyen Xuan Duc (Advisor)
- **Affiliation:** Posts and Telecommunications Institute of Technology (PTIT), Hanoi, Vietnam
- **Target Venues:** IEEE Transactions on Geoscience and Remote Sensing (TGRS) / IEEE Access / NeurIPS Climate Change AI workshop

### Abstract
> Data-driven medium-range weather forecasting using deep learning has achieved groundbreaking success on global domains. However, direct operational downscaling to regional tropical domains—specifically Southeast Asia—faces formidable challenges, including intricate monsoon-cyclone dynamics, intense sub-grid convective cloud processes, and severe computational hardware constraints. In this paper, we propose **WeatherForecast-SEA**, an efficient, accessible regional forecasting system operating on a multi-scale bipartite graph topology ($0.25^\circ \times 0.25^\circ$ equirectangular observation grid with 31,959 nodes coupled to a 500-node $2.0^\circ$ icosahedral latent mesh). We fuse 14 physical atmospheric variables from ECMWF ERA5 reanalysis with geostationary thermal infrared cloud-top brightness temperature from NASA GPM MERGIR. Through one-time float16 binary memory-mapping engineering, our 538K-parameter architecture trains reliably on commodity dual-T4 GPUs within 6 hours. Rigorous multi-start 12-step autoregressive rollouts across an independent 2026 test set demonstrate that WeatherForecast-SEA beats Persistence by 37% RMSE and systematically outperforms convolutional U-Nets at the $+72h$ forecast horizon across surface temperature, sea-level pressure, and mid-tropospheric geopotential.

---

## 2. Modular LaTeX Source Code Structure :id=modular-latex-structure

The complete LaTeX manuscript source code resides in the directory [`docs/paper/`](file:///Users/dxng/Documents/Codes/WeatherForecast/docs/paper):

| File | Section | Core Focus |
| :--- | :--- | :--- |
| [`main.tex`](file:///Users/dxng/Documents/Codes/WeatherForecast/docs/paper/main.tex) | Root Document | IEEEtran document class, package imports, author blocks, and section inclusion |
| [`intro.tex`](file:///Users/dxng/Documents/Codes/WeatherForecast/docs/paper/intro.tex) | Section I | Research motivation, NWP limitations in Southeast Asia, and main scientific contributions |
| [`related.tex`](file:///Users/dxng/Documents/Codes/WeatherForecast/docs/paper/related.tex) | Section II | Literature review: NWP models, global AI models (FourCastNet, Pangu, GraphCast), 2024–2026 foundation & diffusion models (Aurora, AIFS, GenCast, Prithvi WxC), and tropical meteorology |
| [`method.tex`](file:///Users/dxng/Documents/Codes/WeatherForecast/docs/paper/method.tex) | Section III | Problem formulation, multi-scale Grid–Mesh–Grid graph, message passing, and latitude-weighted loss |
| [`exp.tex`](file:///Users/dxng/Documents/Codes/WeatherForecast/docs/paper/exp.tex) | Section IV | Experimental protocol, multi-lead quantitative benchmarks, ACC/RMSE curves, and meteorological error diagnostics |
| [`conclusion.tex`](file:///Users/dxng/Documents/Codes/WeatherForecast/docs/paper/conclusion.tex) | Section V | Summary of findings, identified gaps, and future research roadmap |
| [`references.bib`](file:///Users/dxng/Documents/Codes/WeatherForecast/docs/paper/references.bib) | Bibliography | 30+ peer-reviewed citations in standard BibTeX format |

---

## 3. Compilation Runbook :id=compilation-instructions

To compile the LaTeX source into a publication-ready PDF:

```bash
# Navigate to the paper directory
cd docs/paper

# Compile with pdfLaTeX and BibTeX
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# View generated PDF
open main.pdf  # On macOS
# xdg-open main.pdf  # On Linux
```

---

## 4. IEEE BibTeX Citation :id=bibtex-citation

```bibtex
@article{phong2026weatherforecast_sea,
  author    = {Dang, Tuan Phong and Tran, Minh Hieu and Nguyen, Xuan Duc},
  title     = {WeatherForecast-SEA: Regional Graph Neural Networks for Weather Forecasting in Southeast Asia Fusing ERA5 Reanalysis and NASA Satellite Imagery},
  journal   = {IEEE Transactions on Geoscience and Remote Sensing},
  year      = {2026},
  note      = {In review}
}
```
