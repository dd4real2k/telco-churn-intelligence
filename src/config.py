from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "telco_churn_cleaned.csv"
FEATURED_DATA_PATH = PROCESSED_DATA_DIR / "telco_churn_featured.csv"

# Output paths
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Target column
TARGET_COL = "Churn"

# Random state for reproducibility
RANDOM_STATE = 42
