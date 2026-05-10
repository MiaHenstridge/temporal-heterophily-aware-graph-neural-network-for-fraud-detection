# Bridging Temporal and Heterophilic Graph Learning for First-Party Financial Fraud Detection

## Overview

This repo implements and benchmarks **THeGCN** (Temporal Heterophilic Graph Convolutional Network) for detecting fraudulent loan borrowers on [DGraphFin](https://dgraph.xinye.com/dataset), a real-world financial graph with 3.7M nodes, 4.3M edges, and a 1.27% fraud rate.

THeGCN combines a **temporal edge sampler** (via TGL) with a **dual-filter aggregator** using adaptive low-pass and high-pass weights, allowing the model to jointly capture interaction timing and the masking behavior of fraudsters hiding among legitimate users.

## Models Compared

| Category | Models |
|----------|--------|
| ML baselines | Naïve Bayes, Linear SVM, XGBoost |
| Static GNNs | GraphSAGE, GAT, GATv2, FAGCN |
| Temporal GNNs | TGAT, **THeGCN** |

All models are evaluated under two feature configurations: default (17 anonymized node features) and augmented (+7 engineered interaction features).

## Key Findings

- Temporal context drives larger performance gains than spatial context alone under sparse network structure.
- Heterophily-aware aggregation becomes significantly beneficial when paired with engineered interaction features.
- THeGCN achieves the best overall AP, Recall@k, and Precision@k among all models.

## Project Structure

```
src/           # Model implementations, training scripts, and utilities
```

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). The `pyproject.toml` is pre-configured for PyTorch 2.4 + CUDA 12.4.

```bash
uv sync
```

This resolves all dependencies including PyTorch, PyG, DGL, and their CUDA-specific wheels. No manual index juggling needed — `uv` handles the custom sources defined in `pyproject.toml`.

The DGraphFin dataset must be downloaded separately from [dgraph.xinye.com/dataset](https://dgraph.xinye.com/dataset).

## References

- Yan et al., "THeGCN: Temporal Heterophilic Graph Convolutional Network" (2025)
- Huang et al., "DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection", NeurIPS 2022