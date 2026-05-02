from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "student_dropout_dataset.csv"
MODEL_DIR = ROOT_DIR / "models"

TARGET_COLUMN = "dropout_status"
ID_COLUMN = "student_id"
PRIMARY_APP_FEATURES = [
    "attendance_rate",
    "gpa",
    "subject_failure_count",
    "engagement_score",
]
DERIVED_FEATURES = [
    "performance_score",
    "risk_score",
    "engagement_intensity",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5
