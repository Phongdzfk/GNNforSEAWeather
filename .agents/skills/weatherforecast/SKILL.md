---
name: weatherforecast
description: >-
  Expert workflows, domain knowledge, and execution runbooks for WeatherForecast-SEA: regional GraphCast GNNs, ECMWF ERA5 and NASA GPM satellite pipelines, Float16 binary memmap, latitude-weighted loss, Docsify documentation serving, and thesis OpenXML compilation.
---

# WeatherForecast-SEA Skill & Operation Runbook

This skill provides step-by-step procedures and specialized knowledge for engineering, evaluating, and documenting the **WeatherForecast-SEA** atmospheric forecasting system.

---

## 🧭 Repository Layout & Artifact Map

- `docs/index.html`: Docsify web documentation entry point with LaTeX (MathJax) and Mermaid render hooks.
- `docs/*.md`: Technical documentation pages (Architecture, Data Pipeline, Training, Evaluation, Dashboard, Literature & Novel Methods).
- `docs/serve.py`: Standalone HTTP preview server with auto-port incrementation.
- `docs/thesis/khung-do-an.md`: Comprehensive graduation thesis monograph framework (English).
- `docs/thesis/create_thesis_docx.py`: Script compiling `khung-do-an.md` into `Khung_do_an.docx` using OpenXML templates.
- `docs/thesis/thesis-outline.md` & `docs/thesis/roadmap.md`: 7-chapter outline and 12-week implementation sprint timeline.
- `docs/paper/`: Complete IEEE journal-format LaTeX manuscript (`main.tex`, `intro.tex`, `related.tex`, `method.tex`, `exp.tex`, `conclusion.tex`, `references.bib`).

---

## ⚡ Standard Operating Procedures (SOP)

### SOP 1: Previewing Web Documentation
1. Run `python3 docs/serve.py 3000`.
2. The server will start and automatically select an open port if 3000 is occupied.
3. Access `http://localhost:<port>` to inspect changes.
4. Verify LaTeX equations render crisply without unprocessed `$..$` or `$$..$$` artifacts.
5. Verify Mermaid flowcharts render into interactive SVG diagrams.

### SOP 2: Recompiling the Thesis Word Monograph
When edits are made to `docs/thesis/khung-do-an.md`:
1. Execute:
   ```bash
   python3 docs/thesis/create_thesis_docx.py
   ```
2. Verify output confirmation: `✅ Generated DOCX successfully: .../Khung_do_an.docx`.
3. Ensure no unescaped XML characters (`<`, `>`, `&`) break the OpenXML structure.

### SOP 3: Verifying Documentation Anti-Leakage & Consistency
When adding or modifying documentation:
- Keep all documentation in **English** (except `docs/references/`).
- Preserve table of contents cards:
  ```html
  <div class="toc-card">
    <div class="toc-title">📑 Table of Contents</div>
    ...
  </div>
  ```
- Use custom heading IDs: `## Heading Title :id=heading-slug`.
- Check that TOC link anchors match custom IDs exactly (`href="#heading-slug"`).

---

## 🧮 Mathematical & Algorithmic Reference

### 1. Latitude-Weighted RMSE
$$\mathcal{L}_{weighted} = \sqrt{\frac{1}{C \cdot N} \sum_{c=1}^C \sum_{i=1}^N w_i \left(y_{i, c} - \hat{y}_{i, c}\right)^2}, \quad w_i = \frac{\cos(\phi_i)}{\frac{1}{N}\sum_{j=1}^N \cos(\phi_j)}$$

### 2. Multi-Scale Message Passing
$$\mathbf{m}_{u \to v} = \text{MLP}_{edge}\left([\mathbf{h}_u, \mathbf{h}_v, \mathbf{e}_{uv}]\right)$$
$$\mathbf{h}_v' = \text{MLP}_{node}\left([\mathbf{h}_v, \sum_{u \in \mathcal{N}(v)} \mathbf{m}_{u \to v}]\right) + \mathbf{h}_v$$

### 3. Solar Radiation Forcing Formula
$$S_{TOA}(lat, lon, t) = S_0 \cdot \left(\frac{d_0}{d(t)}\right)^2 \cdot \max(0, \cos \theta_z(lat, lon, t))$$
where $S_0 = 1361\text{ W/m}^2$, $\cos \theta_z = \sin(lat)\sin(\delta) + \cos(lat)\cos(\delta)\cos(\omega t)$.

### 4. Memory Optimization (Memmap Array Construction)
```python
# One-time pre-decode from Zarr chunk blocks to flat binary float16
fp16_array = np.memmap(
    "data_cache.npy", dtype="float16", mode="w+", shape=(num_samples, channels, H, W)
)
# Read in PyTorch DataLoader
mapped_data = np.memmap(
    "data_cache.npy", dtype="float16", mode="r", shape=(num_samples, channels, H, W)
)
```
