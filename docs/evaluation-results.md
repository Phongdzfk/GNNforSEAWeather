# Experimental Evaluation & Diagnostic Analysis

> Comprehensive evaluation report on the independent 2026 test set: 72-hour autoregressive rollout, quantitative benchmark against Persistence and U-Net, Anomaly Correlation Coefficient (ACC), regional error stratification, and spectral blurring analysis.

<div class="toc-card">
  <div class="toc-title">📑 Table of Contents</div>
  <ol>
    <li><a href="#evaluation-protocol">1. Evaluation Protocol & Benchmark Suite</a></li>
    <li><a href="#quantitative-benchmark">2. Overall Quantitative Benchmark</a></li>
    <li>
      <a href="#meteorological-analysis">3. Diagnostic Meteorological Analysis</a>
      <ol>
        <li><a href="#regional-error-hierarchy">3.1. Regional Error Hierarchy</a></li>
        <li><a href="#mountainous-cold-bias">3.2. Mountainous Cold Bias</a></li>
        <li><a href="#diurnal-aliasing">3.3. Diurnal Cycle Zigzag Aliasing</a></li>
        <li><a href="#spectral-blurring">3.4. Spatial Spectral Blurring</a></li>
      </ol>
    </li>
  </ol>
</div>

---

## 1. Evaluation Protocol & Benchmark Suite :id=evaluation-protocol

To rigorously assess forecasting skill under realistic operational conditions, models undergo **Multi-start Autoregressive Rollout**:
- **Independent Test Split:** Calendar year 2026 (587 synoptic timesteps strictly unseen during training and hyperparameter tuning).
- **Sampling Strategy:** 30 randomly selected initial initialization timestamps ($t_0$) stratified across seasonal regimes (dry season, monsoon, and tropical cyclone seasons).
- **Rollout Length:** From each $t_0$, the model performs 12 sequential recursive inference steps ($\Delta t = 6\text{ h}$), iteratively feeding step $t+1$ predictions back into input slots to project up to lead time $+72$ hours.
- **Reference Baselines:**
  1. **Persistence:** Assumes atmospheric state remains frozen at initial condition $t_0$.
  2. **Climatology:** Long-term historical multi-year mean for the corresponding day-of-year.
  3. **U-Net (Convolutional Baseline):** Standard 4-level encoder-decoder CNN widely used in weather forecasting, parameterized to a comparable capacity (~0.6M parameters).

---

## 2. Overall Quantitative Benchmark :id=quantitative-benchmark

Comparison of Root Mean Square Error (RMSE, lower is better) and Anomaly Correlation Coefficient (ACC, higher is better) at $+24h$ and $+72h$ forecast horizons:

| Atmospheric Variable | Method | RMSE (+24h) $\downarrow$ | ACC (+24h) $\uparrow$ | RMSE (+72h) $\downarrow$ | ACC (+72h) $\uparrow$ | Observations & Findings |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2m Temperature ($t2m$)** | Persistence | 1.84 K | 0.76 | 2.12 K | 0.61 | Marked diurnal cycle zigzag oscillations |
| | U-Net (CNN) | **1.04 K** | **0.94** | 1.68 K | 0.79 | U-Net slightly sharper in short-range |
| | **WeatherForecast-SEA** | 1.12 K | 0.93 | **1.58 K** | **0.82** | **GNN surpasses U-Net at $+72h$ lead time** |
| **Mean Sea Level Pressure ($msl$)** | Persistence | 2.45 hPa | 0.78 | 3.82 hPa | 0.52 | Loses operational skill beyond 48h |
| | U-Net (CNN) | **1.18 hPa** | 0.91 | 2.65 hPa | 0.61 | |
| | **WeatherForecast-SEA** | 1.22 hPa | **0.92** | **2.48 hPa** | **0.65** | **Maintains skillful ACC $> 0.6$ through 72h** |
| **Geopotential ($z@500$)** | Persistence | 124.6 $m^2/s^2$ | 0.74 | 210.5 $m^2/s^2$ | 0.48 | Error diverges exponentially |
| | U-Net (CNN) | 68.2 $m^2/s^2$ | 0.89 | 148.0 $m^2/s^2$ | 0.62 | |
| | **WeatherForecast-SEA** | **65.4 $m^2/s^2$** | **0.90** | **135.2 $m^2/s^2$** | **0.67** | **GNN clearly superior in mid-troposphere** |
| **Zonal Wind ($u10$)** | Persistence | 2.31 m/s | 0.58 | 3.15 m/s | 0.38 | High sub-grid noise accumulation |
| | U-Net (CNN) | **1.62 m/s** | **0.78** | 2.74 m/s | 0.56 | |
| | **WeatherForecast-SEA** | 1.68 m/s | 0.76 | **2.62 m/s** | **0.60** | **Reaches useful skill threshold at $+72h$** |
| **Specific Humidity ($q@850$)**| Persistence | 1.54 g/kg | 0.62 | 2.21 g/kg | 0.35 | Highly non-linear moisture dynamics |
| | U-Net (CNN) | **1.12 g/kg** | **0.81** | 1.94 g/kg | 0.48 | |
| | **WeatherForecast-SEA** | 1.15 g/kg | 0.80 | **1.88 g/kg** | **0.51** | Skill drops below 0.6 around 48h mark |

---

## 3. Diagnostic Meteorological Analysis :id=meteorological-analysis

### 3.1. Regional Error Hierarchy :id=regional-error-hierarchy
Spatially disaggregated verification reveals a consistent physical error hierarchy:
$$\text{RMSE}_{\text{Ocean}} < \text{RMSE}_{\text{Plains}} < \text{RMSE}_{\text{Inland}} < \text{RMSE}_{\text{Complex Orography}}$$

- **Marine Basins (South China Sea, Gulf of Thailand):** Error residual maps remain near-neutral from $+6h$ to $+72h$. The enormous thermal inertia of the ocean provides slow-evolving temperature and pressure boundary conditions that the neural operator models with high fidelity.
- **Lowland Plains:** Displays slight diurnal heating error spikes, but overall RMSE remains well-constrained ($< 1.2\text{ K}$).

### 3.2. Mountainous Cold Bias :id=mountainous-cold-bias
Across rugged terrain—including the Annamite Range (Truong Son), Shan Plateau (Myanmar), and Yunnan Highlands (China)—residual maps reveal prominent negative bias bands (under-predicting temperatures by 3 to 5 Kelvin):
- **Causal Mechanism:** The icosahedral processor mesh ($2^\circ$) averages out steep elevation gradients, failing to resolve localized thermal trapping in deep river valleys and sheltered basins.

### 3.3. Diurnal Cycle Zigzag Aliasing :id=diurnal-aliasing
Over continental landmasses, Persistence error exhibits severe sawtooth oscillations:
- At 24h, 48h, and 72h lead times, Persistence error dips because local solar time aligns with the initial state (midday matching midday).
- In the current baseline, explicit top-of-atmosphere solar insolation angles are not yet provided as input channels, causing the model's relative margin over Persistence to fluctuate periodically.

### 3.4. Spatial Spectral Blurring :id=spectral-blurring
2D Fourier energy spectrum analysis demonstrates that at longer horizons ($+48h$ and $+72h$), spectral energy at high spatial frequencies (small spatial wavelengths) progressively decays:
- Forecasted fields appear visibly smoother compared to ground-truth reanalysis.
- This is a well-documented mathematical characteristic of regression with L2/MSE loss: as atmospheric trajectory uncertainty compounds recursively, minimizing expected squared error converges toward the ensemble conditional mean.
