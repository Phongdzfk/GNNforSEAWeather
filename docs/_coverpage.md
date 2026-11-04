<!-- docs/_coverpage.md -->

# WeatherForecast-SEA <small>v1.0</small>

> **Deep Learning Regional Weather Forecasting for Southeast Asia via Graph Neural Networks**  
> Fusing ECMWF ERA5 Reanalysis & NASA GPM Satellite Observations on Accessible Hardware

<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Spatial Resolution</div>
    <div class="kpi-value">0.25°</div>
    <div class="kpi-sub">~28 km / 31,959 observation nodes</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Skillful Lead Time</div>
    <div class="kpi-value">+72h</div>
    <div class="kpi-sub">ACC &gt; 0.6 (t2m, msl, z500)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Model Parameters</div>
    <div class="kpi-value">538K</div>
    <div class="kpi-sub">Optimized for Kaggle GPU Training</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Accuracy Gain</div>
    <div class="kpi-value">+37%</div>
    <div class="kpi-sub">Outperforming Persistence Baseline</div>
  </div>
</div>

- **Geographic Domain:** Southeast Asia (11°S – 28.5°N, 92°E – 142°E)
- **Architecture:** Encoder–Processor–Decoder GNN on multi-scale Grid–Mesh–Grid (0.25° → 2.0°)
- **Dual Data Streams:** ECMWF ERA5 (14 atmospheric variables) + NASA GPM MERGIR Infrared Satellite (4 km)

[Get Started](README.md?id=main)
[GNN Architecture](architecture.md)
[Research & Gaps](literature-and-novel-methods.md)

<!-- {docsify-ignore-all} -->
