from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#Dataset path
DATASET_DIR = PROJECT_ROOT / "Brain Tumor Data"


# Training configuration

IMAGE_SIZE = 128

BATCH_SIZE = 8

LEARNING_RATE = 1e-4

EPOCHS = 30

NUM_WORKERS = 2