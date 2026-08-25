"""
Central configuration file for paths, constants, and execution parameters.

Serves as the single source of truth for repository paths, experimental design choices,
and hyperparameter defaults.
"""

from pathlib import Path
import os
import random


# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
# Set this path to your parent data directory containing kermany/, togunwa/, and musa/
DATA_ROOT = Path("/content/drive/MyDrive/MSc AI DISSERTATION/data")

# Infer repository root based on file position
REPO_ROOT = Path(__file__).resolve().parent

# Dataset subdirectories
KERMANY_DIR = DATA_ROOT / "kermany"
TOGUNWA_DIR = DATA_ROOT / "togunwa"
MUSA_DIR = DATA_ROOT / "musa"

# Dataset directory structures
KERMANY_SPLITS = {
    "train": {
        "NORMAL": KERMANY_DIR / "train" / "NORMAL",
        "PNEUMONIA": KERMANY_DIR / "train" / "PNEUMONIA",
    },
    "test": {
        "NORMAL": KERMANY_DIR / "test" / "NORMAL",
        "PNEUMONIA": KERMANY_DIR / "test" / "PNEUMONIA",
    },
}

TOGUNWA_CLASSES = {
    "NORMAL": TOGUNWA_DIR / "NORMAL",
    "PNEUMONIA": TOGUNWA_DIR / "PNEUMONIA",
}

# Musa dataset configuration (used for OOD evaluations)
MUSA_CLASSES = ["NORMAL", "PNEUMONIA", "TB", "COVID"]
MUSA_ID_CLASSES = ["NORMAL", "PNEUMONIA"]
MUSA_OOD_CLASSES = ["TB", "COVID"]

MUSA_SPLITS = {
    "train": {
        "NORMAL": MUSA_DIR / "train" / "NORMAL",
        "PNEUMONIA": MUSA_DIR / "train" / "PNEUMONIA",
        "TB": MUSA_DIR / "train" / "TB",
        "COVID": MUSA_DIR / "train" / "COVID",
    },
    "test": {
        "NORMAL": MUSA_DIR / "test" / "NORMAL",
        "PNEUMONIA": MUSA_DIR / "test" / "PNEUMONIA",
        "TB": MUSA_DIR / "test" / "TB",
        "COVID": MUSA_DIR / "test" / "COVID",
    },
}

# Output directories inside the repository
OUTPUTS_DIR = REPO_ROOT / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
FIGURES_DIR = OUTPUTS_DIR / "figures"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"


# ---------------------------------------------------------------------------
# Experimental Settings
# ---------------------------------------------------------------------------
ARCHITECTURES = ["mobilenet_v2", "efficientnet_b0"]
DIRECTIONS = ["K2N", "N2K"]
MATCHED_SIZES = [50, 100, 190]

CLASS_BALANCE_RATIO = 1.0  # 1:1 Normal to Pneumonia
CLASSES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}
IDX_TO_CLASS = {0: "NORMAL", 1: "PNEUMONIA"}

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 16
MIXED_PRECISION = True

# Cross-Validation Configuration
N_FOLDS = 5
TOGUNWA_HOLDOUT_SIZE = 30

# Evaluation Parameters
BOOTSTRAP_N = 2000
BOOTSTRAP_CI = 0.95
FIXED_SENSITIVITY = 0.95
ECE_N_BINS = 15

# Reproducibility
SEED = 42
SEEDS = [42, 43, 44, 45, 46, 47]

# Styling Palette
BRAND = {"C1": "#B70D50", "C2": "#621B40", "C3": "#0070C0"}

# Bin-count bounds for sample-size-scaled ECE (n_bins ~ sqrt(n), clipped).
# Fixed 15-bin ECE is retained only as a labelled secondary diagnostic,
# never as the primary calibration estimate on small evaluation sets.
ECE_MIN_BINS = 5
ECE_MAX_BINS = 15
# Bootstrap repetitions for the ECE bias-correction procedure (bootstrap_debiased_ece).
DEBIAS_ECE_N_BOOT = 200
# Repetitions for the Kermany-624 -> n=30 size-matched subsampling sanity check.
SUBSAMPLE_N_REPEATS = 500
# TOST equivalence margin = EQUIVALENCE_MARGIN_MULTIPLIER * (observed per-seed
# SD of a fixed configuration). Pre-registered here rather than chosen per test.
EQUIVALENCE_MARGIN_MULTIPLIER = 2.0

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def get_device(verbose=True):
    """Determine available compute hardware (CUDA GPU or CPU fallback)."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            if verbose:
                print(f"Compute Device: GPU ({torch.cuda.get_device_name(0)})")
            return device
    except ImportError:
        pass

    if verbose:
        print("Compute Device: CPU")
    return "cpu"


def set_all_seeds(seed=SEED):
    """Seed Python, NumPy, and PyTorch random number generators for exact reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def set_plot_style():
    """Apply global Matplotlib styles across visualization scripts."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def ensure_output_dirs():
    """Ensure all expected output subdirectories exist."""
    for directory in (RESULTS_DIR, FIGURES_DIR, CHECKPOINTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def summary():
    """Print experiment setup configuration to screen."""
    n_runs = len(ARCHITECTURES) * len(DIRECTIONS) * len(MATCHED_SIZES)
    print("Cross-Population CXR Asymmetry Experiment Configuration")
    print("-" * 55)
    print(f"DATA_ROOT           : {DATA_ROOT}")
    print(f"Architectures       : {ARCHITECTURES}")
    print(f"Directions          : {DIRECTIONS}")
    print(f"Matched sizes       : {MATCHED_SIZES}")
    print(f"Class balance       : 1:1 Ratio")
    print(f"Total training runs : {n_runs} ({len(ARCHITECTURES)} archs x {len(DIRECTIONS)} dirs x {len(MATCHED_SIZES)} sizes)")
    print(f"CV folds            : {N_FOLDS} | Holdout: {TOGUNWA_HOLDOUT_SIZE}")
    print(f"Seeds               : {SEEDS}")


if __name__ == "__main__":
    summary()
