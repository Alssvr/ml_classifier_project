# config.py
# -*- coding: utf-8 -*-
"""
Конфигурация проекта классификации текстов
"""

import os
from pathlib import Path
from datetime import datetime

# ============================================================
# БАЗОВЫЕ ПУТИ
# ============================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
FEEDBACK_DIR = DATA_DIR / "feedback"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"

for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, 
                 FEEDBACK_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# ФАЙЛЫ ДАННЫХ
# ============================================================

CLASSIFIED_FILE = RAW_DATA_DIR / "classified_augmented.xlsx"
UNCLASSIFIED_FILE = RAW_DATA_DIR / "classified_new.xlsx"
STOPWORDS_FILE = EXTERNAL_DATA_DIR / "stopwords_ru.txt"

# ============================================================
# КОЛОНКИ В ДАННЫХ
# ============================================================

ID_COLUMN = "ID"
TEXT_COLUMN = "Наименование"
LABEL_COLUMN = "Шаблон класса"
PROCESSED_TEXT_COLUMN = "text_processed"

# ============================================================
# ПАРАМЕТРЫ ПРЕДОБРАБОТКИ ТЕКСТА
# ============================================================

PREPROCESSING_CONFIG = {
    "use_lemmatization": True,
    "remove_numbers": False,
    "min_word_length": 2,
    "lowercase": True,
}

CUSTOM_STOPWORDS = [
    "шт", "шт.", "г", "г.", "кг", "мм", "см", "м", 
    "арт", "арт.", "артикул", "номер", "тип"
]

# ============================================================
# ПАРАМЕТРЫ TF-IDF ВЕКТОРИЗАЦИИ (ОСНОВНЫЕ)
# ============================================================

TFIDF_CONFIG = {
    "max_features": 30000,
    "ngram_range": (1, 3),
    "min_df": 1,
    "max_df": 0.8,
    "sublinear_tf": True,
    "norm": "l2",
    "use_idf": True,
    "smooth_idf": True,
}

# ============================================================
# ПАРАМЕТРЫ МОДЕЛИ
# ============================================================

MODEL_CONFIG = {
    "classifier_type": "logistic_regression",
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "random_state": 42,
    "n_jobs": -1,
}

MODEL_NAME = "product_classifier"
MODEL_VERSION = "1.2.0"  # Новая версия!

# ============================================================
# ПАРАМЕТРЫ ОБУЧЕНИЯ
# ============================================================

TRAIN_CONFIG = {
    "test_size": 0.2,
    "cv_folds": 5,
    "random_state": 42,
    "stratify": True,
}

# ============================================================
# ПАРАМЕТРЫ CLEANLAB (АНАЛИЗ ШУМА)
# ============================================================

CLEANLAB_CONFIG = {
    "cv_folds": 5,
    "confidence_threshold": 0.9,
    "use_tfidf": True,
    "max_features": 5000,  # Для анализа можно меньше
}

# ============================================================
# ПАРАМЕТРЫ АКТИВНОГО ОБУЧЕНИЯ
# ============================================================

ACTIVE_LEARNING_CONFIG = {
    "samples_per_cycle": 500,
    "confidence_threshold": 0.7,
    "uncertainty_ratio": 0.4,
    "margin_ratio": 0.3,
    "diversity_ratio": 0.3,
}

# ============================================================
# ПАРАМЕТРЫ ПРЕДСКАЗАНИЯ
# ============================================================

PREDICTION_CONFIG = {
    "batch_size": 5000,
    "confidence_threshold": 0.1,   # Снижаем с 0.7 до 0.1
    "high_confidence_threshold": 0.25,  # Снижаем с 0.9 до 0.25
    "margin_threshold": 0.1,       # Добавляем: отрыв от второго класса
}

# ============================================================
# ЛОГИРОВАНИЕ
# ============================================================

LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / f"classifier_{datetime.now():%Y%m%d}.log",
}

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def get_model_path(version: str = None) -> Path:
    if version is None:
        version = MODEL_VERSION
    version_str = version.replace(".", "_")
    return MODELS_DIR / f"{MODEL_NAME}_v{version_str}.pkl"

def get_metadata_path() -> Path:
    return MODELS_DIR / "model_metadata.json"

def print_config():
    print("="*60)
    print("КОНФИГУРАЦИЯ ПРОЕКТА")
    print("="*60)
    print(f"\nДанные:")
    print(f"  Размеченные: {CLASSIFIED_FILE}")
    print(f"  Неразмеченные: {UNCLASSIFIED_FILE}")
    print(f"\nМодель: {MODEL_NAME} v{MODEL_VERSION}")
    print(f"\nПрепроцессинг: лемматизация={PREPROCESSING_CONFIG['use_lemmatization']}")
    print(f"\nTF-IDF: max_features={TFIDF_CONFIG['max_features']}, ngrams={TFIDF_CONFIG['ngram_range']}")
    print(f"\nОбучение: test_size={TRAIN_CONFIG['test_size']}, cv_folds={TRAIN_CONFIG['cv_folds']}")
    print("="*60)

if __name__ == "__main__":
    print_config()