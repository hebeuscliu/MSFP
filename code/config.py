"""
MSFP-MD: Multi-Scale Feature Fusion and Ensemble Prediction Framework
for Breast Cancer Inhibitor Discovery Validated by MD Simulation

Global configuration for the refactored pipeline.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")

# ── Original Data Sources (read-only) ───────────────────────────────
ORIGINAL_PROJECT = "/root/disk1/senchenliu/Project/Git_clone/MSFP-MD"
POSITIVE_CSV = os.path.join(
    ORIGINAL_PROJECT,
    "Fusion/MolCLR_BC_test_3/data_and_scripts/positive_origin.csv"
)
ZINC_NEG_CSV = os.path.join(
    ORIGINAL_PROJECT,
    "Fusion/MolCLR_BC_test_3/data_and_scripts/ZINC_forsale_invivo4NegSample.csv"
)

# External encoder roots
MOLCLR_ROOT = "/root/disk1/senchenliu/Project/Git_clone/MolCLR"
UNIMOL_ROOT = "/root/disk1/senchenliu/Project/Git_clone/Uni-Mol-main"

# ── Model Architecture ─────────────────────────────────────────────
MOLCLR_VEC_DIM = 512
UNIMOL_VEC_DIM = 1536
OUTPUT_DIM = 1
BATCH_SIZE = 32

# ── Training ───────────────────────────────────────────────────────
K_FOLDS = 10
RANDOM_SEED = 42
EARLY_STOP_PATIENCE = 15
MAX_EPOCHS = 200

# Default best hyperparams (from original Optuna search)
DEFAULT_HPARAMS = {
    "EMBED_DIM": 256,
    "EPOCHS": 200,
    "LEARNING_RATE": 5.4486466044365087e-05,
    "META_LR": 0.001,
    "DROPOUT": 0.12907693474843712,
}

# ── Property Matching ──────────────────────────────────────────────
PROPERTY_NAMES = ["MW", "LogP", "HBD", "HBA", "RotB", "TPSA"]
PROPERTY_TOLERANCE = 0.20          # ±20% property window
TANIMOTO_MAX = 0.5                 # max structural similarity allowed
NEG_PER_POS_RATIO = 50             # target negatives per positive
N_SAMPLING_REPLICATES = 10         # independent replicate datasets
MIN_NEG_PER_POS = 10               # minimum negatives per positive

# ── Scaffold Split ─────────────────────────────────────────────────
SCAFFOLD_SPLIT_METHOD = "bemis_murcko"   # bemis_murcko or generic

# ── Logging ────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
