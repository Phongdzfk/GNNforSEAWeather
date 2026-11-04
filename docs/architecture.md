# Graph Neural Network Architecture of WeatherForecast-SEA

> In-depth analysis of the multi-scale **Encoder – Processor – Decoder** architecture (Grid–Mesh–Grid), message passing formulations, and parameter budget optimization (538K params).

<div class="toc-card">
  <div class="toc-title">📑 Table of Contents</div>
  <ol>
    <li><a href="#gnn-rationale">1. Architectural Rationale: Why Graph Neural Networks?</a></li>
    <li>
      <a href="#graph-topology">2. Multi-Scale Graph Topology (Grid–Mesh–Grid)</a>
      <ol>
        <li><a href="#grid-to-mesh">2.1. Encoding Edges (Grid-to-Mesh)</a></li>
        <li><a href="#mesh-to-mesh">2.2. Processing Edges (Mesh-to-Mesh)</a></li>
        <li><a href="#mesh-to-grid">2.3. Decoding Edges (Mesh-to-Grid)</a></li>
      </ol>
    </li>
    <li>
      <a href="#neural-modules">3. Neural Module Formulations</a>
      <ol>
        <li><a href="#encoder">3.1. Encoder Module</a></li>
        <li><a href="#processor">3.2. Dynamical Processor GNN</a></li>
        <li><a href="#decoder">3.3. Decoder & Residual Formulation</a></li>
      </ol>
    </li>
    <li><a href="#parameter-budget">4. Parameter Budget & Inference Latency</a></li>
  </ol>
</div>

---

## 1. Architectural Rationale: Why Graph Neural Networks? :id=gnn-rationale

In atmospheric deep learning modeling, three major architectural paradigms are considered:

| Architectural Property | Convolutional Networks (CNNs) | Vision Transformers (ViTs) | Graph Neural Networks (GNNs) |
| :--- | :--- | :--- | :--- |
| **Spatial Inductive Bias** | Translation Invariance | Unstructured / Free Attention | Non-Euclidean, flexible topology |
| **Computational Complexity** | $O(N)$ (local small kernels) | $O(N^2)$ (full self-attention) | $O(N + E)$ (linear in edge count) |
| **Spherical Geometry Preservation** | Poor (distortions near poles/latitudes) | Requires complex 2D/3D embeddings | **Excellent** (geodesic spherical features on edges) |
| **Long-Range Spatial Interactions** | Constrained by receptive field | Global but prohibitive memory footprint | **Optimal** (propagated via coarse multi-hop mesh) |

- **Limitations of CNNs:** Translation invariance breaks down on planetary surfaces: equatorial dynamics fundamentally differ from mid-latitude westerlies, and land-sea thermal contrasts violate spatial homogeneity.
- **Limitations of Transformers:** With $N = 31,959$ grid points, full self-attention requires $\sim 10^9$ pairwise interactions per layer, immediately exceeding the memory of commodity GPUs.
- **Superiority of GNNs:** GNNs decouple the **high-resolution observation grid** ($0.25^\circ$) from the **dynamical propagation mesh** ($2.0^\circ$), retaining linear compute scaling while efficiently capturing planetary-scale teleconnections.

---

## 2. Multi-Scale Graph Topology (Grid–Mesh–Grid) :id=graph-topology

WeatherForecast-SEA establishes three distinct, bipartite and internal edge sets:

```mermaid
graph LR
    subgraph GridSpace [Observation Space 0.25°]
        G["31,959 Grid Points (N_lat=159, N_lon=201)"]
    end

    subgraph MeshSpace [Latent Dynamical Space 2.0°]
        M["500 Coarse Mesh Nodes"]
    end

    G -->|Grid-to-Mesh Edges: kNN k=4 (127,836 edges)| M
    M -->|Mesh-to-Mesh Edges: 8-neighborhood (3,734 edges)| M
    M -->|Mesh-to-Grid Edges: Reverse Projection| G
```

### 2.1. Encoding Edges (Grid-to-Mesh: $E_{G \to M}$) :id=grid-to-mesh
- Each fine observation point $v \in G$ is connected to its $k = 4$ nearest coarse mesh nodes $m \in M$ based on great-circle spherical distance.
- Total edge count: $31,959 \times 4 = 127,836$ directed edges.
- Each edge carries invariant geometric features:
  $$\mathbf{e}_{vm} = \left[ \Delta lat, \Delta lon, \|\Delta \mathbf{x}\|, \frac{\Delta lat}{\|\Delta \mathbf{x}\|}, \frac{\Delta lon}{\|\Delta \mathbf{x}\|} \right]$$

### 2.2. Processing Edges (Mesh-to-Mesh: $E_{M \to M}$) :id=mesh-to-mesh
- Coarse mesh $M$ contains 500 nodes evenly spaced on a $2^\circ \times 2^\circ$ resolution.
- Connected via an 8-nearest neighbor (Moore neighborhood) topology in latitude–longitude space.
- Total edge count: 3,734 directed edges.
- Enables atmospheric energy propagation across 4 message-passing hops, achieving an effective receptive radius $> 800\text{ km}$.

### 2.3. Decoding Edges (Mesh-to-Grid: $E_{M \to G}$) :id=mesh-to-grid
- Projects the updated latent representation from coarse mesh nodes $M$ back onto fine observation grid points $G$.

---

## 3. Neural Module Formulations :id=neural-modules

```
┌────────────────────────────────────────────────────────────────────────┐
│               WEATHERFORECAST-SEA NEURAL HYPERPARAMETERS               │
├────────────────────────────────────────────────────────────────────────┤
│ • Input Feature Dimension (N_in): 38 channels                          │
│ • Latent Hidden Dimension (hidden_dim): 128 channels                   │
│ • GNN Processor Layers: 4 message passing layers                       │
│ • Activation Function: SiLU (Sigmoid Linear Unit)                      │
│ • Normalization: LayerNorm                                             │
│ • Output Residual Channels (N_out): 14 atmospheric delta channels      │
│ • TOTAL PARAMETER COUNT: 538,382 parameters (~2.1 MB binary footprint) │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.1. Encoder Module :id=encoder
Input features at each observation node $v \in G$ encompass:
- Atmospheric states across two consecutive time steps: $\mathbf{X}(t-1)$ and $\mathbf{X}(t)$ (28 features).
- NASA GPM MERGIR satellite infrared radiance $T_b$ (1 feature).
- Static orography and surface fields: land-sea mask ($lsm$) and surface geopotential ($z_{surf}$) (2 features).
- Normalized spherical coordinates: $\sin(lat), \cos(lat), \sin(lon), \cos(lon)$ (4 features).
- Solar/temporal cyclic harmonics (3 features).

A 2-layer MLP maps the 38-dimensional input into the 128-dimensional latent space, aggregating onto mesh nodes $M$:
$$\mathbf{m}_{v \to m} = \text{MLP}_{enc}\left(\left[\mathbf{h}_v, \mathbf{h}_m, \mathbf{e}_{vm}\right]\right)$$
$$\mathbf{h}_m^{(0)} = \text{LayerNorm}\left(\mathbf{h}_m + \frac{1}{|\mathcal{N}_G(m)|} \sum_{v \in \mathcal{N}_G(m)} \mathbf{m}_{v \to m}\right)$$

### 3.2. Dynamical Processor GNN :id=processor
Comprises 4 iterative GNN layers. At each layer $l \in \{1, 2, 3, 4\}$:
1. **Edge Update:**
   $$\mathbf{e}_{uv}^{(l)} = \mathbf{e}_{uv}^{(l-1)} + \text{MLP}_{edge}^{(l)}\left(\left[\mathbf{h}_u^{(l-1)}, \mathbf{h}_v^{(l-1)}, \mathbf{e}_{uv}^{(l-1)}\right]\right)$$
2. **Node Aggregation & Update:**
   $$\bar{\mathbf{m}}_v^{(l)} = \frac{1}{|\mathcal{N}_M(v)|} \sum_{u \in \mathcal{N}_M(v)} \mathbf{e}_{uv}^{(l)}$$
   $$\mathbf{h}_v^{(l)} = \mathbf{h}_v^{(l-1)} + \text{MLP}_{node}^{(l)}\left(\left[\mathbf{h}_v^{(l-1)}, \bar{\mathbf{m}}_v^{(l)}\right]\right)$$

### 3.3. Decoder & Residual Formulation :id=decoder
Latent embeddings on the coarse mesh are projected back onto the $0.25^\circ$ fine grid, followed by a linear projection head predicting normalized residual differences:
$$\Delta \hat{\mathbf{X}}(t+1) = \text{Linear}\left(\mathbf{h}_v^{final}\right)$$
The next atmospheric state is reconstructed via residual addition:
$$\hat{\mathbf{X}}(t+1) = \mathbf{X}(t) + \Delta \hat{\mathbf{X}}(t+1)$$

---

## 4. Parameter Budget & Inference Latency :id=parameter-budget

With a compact budget of 538K parameters:
- **Training VRAM (Batch size = 4):** Consumes only $\sim 6.2\text{ GB}$ VRAM.
- **Inference Latency:** $\sim 0.08$ seconds per $+6\text{h}$ step on an Nvidia Tesla T4 GPU.
- **Edge Deployment:** Readily compilable into ONNX Runtime for deployment on standard workstations without high-performance computing infrastructure.
