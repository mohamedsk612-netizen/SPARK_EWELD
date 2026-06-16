# SPARK Pipeline

**Extreme Weather Events Load Dataset** — Full analysis pipeline replicating and extending Liu et al. (2023), *Scientific Data*, 10, 615.

## Package Structure

```
spark/
├── run.py                  ← Main entry point (CLI)
├── setup.py
├── requirements.txt
└── spark/
    ├── __init__.py
    ├── config.py           ← All hyperparameters & constants
    ├── io.py               ← Data loading & saving
    ├── preprocessing.py    ← Cleaning + EW classification
    ├── statistics.py       ← Tables 5–7
    ├── visualisation.py    ← Figures 2–8 (paper replication)
    ├── clustering.py       ← Phase 5: K-means/DBSCAN/SHAP
    ├── forecasting.py      ← Phase 6: LSTM per cluster
    ├── attention.py        ← Phase 7: Attention-LSTM
    ├── demand.py           ← Phase 8: Demand response
    └── utils.py            ← Shared helpers
```

## Installation

```bash
# Clone or download the package
cd spark

# Install core dependencies
pip install -e .

# Install all dependencies including ML
pip install -e ".[all]"

# Or install manually
pip install -r requirements.txt
```

## Usage

```bash
# Full pipeline (all phases)
python run.py \
    --root   /path/to/EWELD \
    --output /path/to/output \
    --cluster --forecast --attention --demand

# Paper replication only (Figs 2–8 + Tables 5–7)
python run.py \
    --root   /path/to/EWELD \
    --output /path/to/output

# Specific phases
python run.py --root ... --output ... --cluster
python run.py --root ... --output ... --forecast
python run.py --root ... --output ... --attention
python run.py --root ... --output ... --demand
```

## Dataset Structure (EWELD)

```
EWELD/
├── Electricity Consumption/
│   ├── A01/ U1.csv, U2.csv ...
│   ├── C10/ ...
│   └── ... (45 division subfolders, 386 CSVs total)
├── Weather Data/
│   ├── W1.csv  (City CT1)
│   ├── W2.csv  (City CT2)
│   └── W3.csv  (City CT3)
└── User Location/
    ├── U_CT1.csv
    ├── U_CT2.csv
    └── U_CT3.csv
```

## Phases

| Phase | Flag | Description | Key Outputs |
|-------|------|-------------|-------------|
| 1–4 | *(always)* | Preprocessing, EW classification, statistics, Figs 2–8 | `fig2`–`fig8`, `spark_statistics.xlsx` |
| 5 | `--cluster` | K-means + DBSCAN + SHAP clustering of 386 buildings | `c4_tsne.png`, `building_cluster_assignments.csv` |
| 6 | `--forecast` | LSTM load forecasting per cluster | `f1_forecast_examples.png`, `forecast_metrics.xlsx` |
| 7 | `--attention` | Attention-LSTM with EW-conditioned heads | `p7f2_attention_heatmap.png`, `attention_lstm_metrics.xlsx` |
| 8 | `--demand` | Demand response analysis (increase/decrease per EW type) | `p8f1_demand_change_heatmap.png`, `demand_response_results.xlsx` |

## Configuration

All hyperparameters are in `spark/config.py`:

```python
# Forecasting
FORE_LOOKBACK         = 96   # 24h history window
FORE_HORIZON          = 96   # 24h prediction horizon
FORE_MAX_WIN_PER_USER = 500  # RAM control (raise for more accuracy)
FORE_EPOCHS           = 30

# Clustering
CLUSTER_K_MAX         = 20
CLUSTER_DBSCAN_SIL    = 0.40  # DBSCAN override threshold
```

## Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install catboost shap -q

!python "/content/drive/MyDrive/spark/run.py" \
    --root   "/content/drive/MyDrive/EWELD" \
    --output "/content/drive/MyDrive/EWELD_Output" \
    --cluster --forecast --attention --demand
```

## Citation

```bibtex
@article{liu2023eweld,
  title   = {EWELD: A Large-Scale Industrial and Commercial Load Dataset
             in Extreme Weather Events},
  author  = {Liu, Guolong and Liu, Jinjie and Bai, Yan and others},
  journal = {Scientific Data},
  volume  = {10},
  pages   = {615},
  year    = {2023},
  doi     = {10.1038/s41597-023-02503-6}
}
```
