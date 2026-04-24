# experimental/config_sbert.py
from pathlib import Path

# Пути (свои, не пересекаются с рабочими)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
FEEDBACK_DIR = DATA_DIR / "feedback"

EXPERIMENTAL_DIR = BASE_DIR / "experimental"
MODELS_DIR = EXPERIMENTAL_DIR / "models_sbert"
RESULTS_DIR = EXPERIMENTAL_DIR / "results_sbert"

for d in [MODELS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Данные (те же, что и для рабочей модели)
CLASSIFIED_FILE = RAW_DATA_DIR / "classified_train.xlsx"
UNCLASSIFIED_FILE = RAW_DATA_DIR / "classified_new.xlsx"

# Колонки
ID_COLUMN = "ID"
TEXT_COLUMN = "Наименование"
LABEL_COLUMN = "Шаблон класса"

# Параметры SBERT
SBERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"  # Лёгкая мультиязычная модель
# Альтернативы:
# "distiluse-base-multilingual-cased-v1"
# "sentence-transformers/LaBSE"

# Параметры классификатора для SBERT
CLASSIFIER_CONFIG = {
    "classifier_type": "LogisticRegression",  # Или SVM, RandomForest
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
}

# Параметры обучения
TRAIN_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
}

MODEL_VERSION = "sbert_v1_0_0"