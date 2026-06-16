"""
spark/config.py
---------------
All hyperparameters, thresholds, and constants in one place.
Edit here — nowhere else — to change pipeline behaviour.
"""

from pathlib import Path

# ── Extreme weather thresholds (95% CI from paper Table 3) ───────────────────
CITY_THRESHOLDS = {
    "CT1": {"low_temp": 50.14, "high_temp": 94.46, "high_hum": 97.15},
    "CT2": {"low_temp": 52.88, "high_temp": 89.93, "high_hum": 98.92},
    "CT3": {"low_temp": 54.76, "high_temp": 89.27, "high_hum": 98.08},
}

EW_DEFINITIONS = {
    "EW1":  "Low temperature",
    "EW2":  "High temperature",
    "EW3":  "High humidity",
    "EW4":  "High heat and humidity",
    "EW5":  "Severe thunderstorm - Damaging Wind Gusts",
    "EW6":  "Severe thunderstorm - Very Damaging Wind Gusts",
    "EW7":  "Severe thunderstorm - Violent Wind Gusts",
    "EW8":  "Tropical Storm",
    "EW9":  "Severe Tropical Storm",
    "EW10": "Typhoon",
    "EW11": "Strong Typhoon",
    "EW12": "Super Typhoon",
    "EW13": "Heavy Rain",
    "EW14": "Heavy Rain/Windy",
    "EW15": "Heavy Rain Shower",
    "EW16": "Heavy Rain Shower/Windy",
    "EW17": "Heavy T-Storm",
    "EW18": "Heavy T-Storm/Windy",
    "EW19": "Light Sleet",
    "EW20": "Light Sleet/Windy",
}

EW_COLS = [f"EW{i}" for i in range(1, 21)]

CONDITION_EW_MAP = {
    "Heavy Rain":              "EW13",
    "Heavy Rain/Windy":        "EW14",
    "Heavy Rain Shower":       "EW15",
    "Heavy Rain Shower/Windy": "EW16",
    "Heavy T-Storm":           "EW17",
    "Heavy T-Storm/Windy":     "EW18",
    "Light Sleet":             "EW19",
    "Light Sleet/Windy":       "EW20",
}

SECTION_NAMES = {
    "A": "Agriculture, forestry and fishing",
    "C": "Manufacturing",
    "D": "Electricity, gas, steam and air conditioning supply",
    "E": "Water supply; sewerage, waste management",
    "F": "Construction",
    "G": "Wholesale and retail trade",
    "H": "Transportation and storage",
    "I": "Accommodation and food service",
    "J": "Information and communication",
    "K": "Financial and insurance activities",
    "L": "Real estate activities",
    "M": "Professional, scientific and technical activities",
    "N": "Administrative and support service activities",
    "O": "Public administration and defence",
    "P": "Education",
    "Q": "Human health and social work activities",
    "S": "Other service activities",
}

# ── Paper-exact user lists (from figure captions) ────────────────────────────
FIG3_USERS  = [
    "U10","U99","U165","U256","U258","U263",
    "U267","U271","U276","U280","U283","U317",
    "U353","U357","U364","U380","U381","U386",
]
FIG456_USERS = ["U10","U99","U165","U263","U283","U317","U364","U380","U381"]
FIG78_UID    = "U380"
FIG78_CITY   = "CT2"
FIG78_YEAR   = 2018

# ── Preprocessing ─────────────────────────────────────────────────────────────
ZSCORE_THRESHOLD  = 3.0
IQR_LOOKBACK_PREV = 2     # obs to replace Z-score outlier
IQR_LOOKBACK_LONG = 96    # obs to replace IQR outlier

# ── Clustering ────────────────────────────────────────────────────────────────
CLUSTER_K_MAX       = 20
CLUSTER_GAP_B       = 10       # Gap Statistic reference datasets
CLUSTER_DBSCAN_SIL  = 0.40     # DBSCAN override threshold
CLUSTER_MIN_SPLIT   = 10       # min cluster size to attempt split

# ── Forecasting (Phases 6 & 7) ────────────────────────────────────────────────
FORE_LOOKBACK        = 96      # 24h history as input
FORE_HORIZON         = 96      # 24h ahead to predict
FORE_TRAIN_FRAC      = 0.70
FORE_VAL_FRAC        = 0.15
FORE_EPOCHS          = 30
FORE_BATCH_SIZE      = 64
FORE_MAX_WIN_PER_USER= 500     # cap windows per building (RAM control)
FORE_LSTM_HIDDEN     = 128
FORE_LSTM_LAYERS     = 2
FORE_LSTM_DROPOUT    = 0.2

# ── Attention-LSTM (Phase 7) ──────────────────────────────────────────────────
ATT_EPOCHS       = 40
ATT_LR           = 5e-4
ATT_WEIGHT_DECAY = 1e-5
ATT_PATIENCE     = 6
ATT_HUBER_DELTA  = 0.5

# ── Peak Demand Forecasting (Phase 8) ────────────────────────────────────────
PEAK_LOOKBACK_DAYS   = 7       # days of 15-min history fed to Transformer
PEAK_EPOCHS          = 60      # base epochs (large clusters)
PEAK_BATCH_SIZE      = 64
PEAK_LR              = 1e-3
PEAK_WEIGHT_DECAY    = 1e-4
PEAK_PATIENCE        = 10      # early-stopping patience (large clusters)
PEAK_D_MODEL         = 64      # Transformer embedding dim
PEAK_NHEAD           = 4       # attention heads (d_model must be divisible)
PEAK_NUM_LAYERS      = 2       # Transformer encoder layers
PEAK_DIM_FF          = 128     # feed-forward hidden dim
PEAK_DROPOUT         = 0.1
PEAK_MAX_WIN_PER_USER= 300     # cap windows per building (RAM control)
# Cluster-size thresholds for architecture selection
PEAK_SMALL_CLUSTER   = 20      # < 20 buildings → MLP instead of Transformer
PEAK_SMALL_EPOCHS    = 100     # more epochs for small clusters (less data)
PEAK_SMALL_PATIENCE  = 15      # more patience for small clusters
PEAK_MIN_CITY_BLDGS  = 8       # cities with fewer buildings → merged into largest city group

# ── Matplotlib global style ───────────────────────────────────────────────────
MPL_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        False,
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "legend.frameon":   True,
}

SLOT_LABELS = [f"{h:02d}:{m:02d}"
               for h in range(24) for m in range(0, 60, 15)]
