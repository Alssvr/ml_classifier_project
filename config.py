# config.py
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

# Создаем директории если их нет
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, 
                 FEEDBACK_DIR, MODELS_DIR, LOGS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# ФАЙЛЫ ДАННЫХ
# ============================================================

# Входные файлы (замените на свои имена)
CLASSIFIED_FILE = RAW_DATA_DIR / "classified_40k.xlsx"
UNCLASSIFIED_FILE = RAW_DATA_DIR / "unclassified_18k.xlsx"

# Выходные файлы
TRAIN_CLEAN_FILE = PROCESSED_DATA_DIR / "train_clean.csv"
PREDICTIONS_FILE = PROCESSED_DATA_DIR / "predictions.xlsx"
LABEL_ISSUES_FILE = FEEDBACK_DIR / "label_issues.xlsx"

# Внешние ресурсы
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
    "remove_numbers": False,  # В технических названиях числа важны
    "min_word_length": 2,
    "lowercase": True,
}

# Список русских стоп-слов (дополнительные)
CUSTOM_STOPWORDS = [
    "шт", "шт.", "г", "г.", "кг", "мм", "см", "м", 
    "арт", "арт.", "артикул", "номер", "тип"
]

# Полный список стоп-слов (базовые + кастомные)
RUSSIAN_STOPWORDS = [
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то',
    'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за',
    'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще',
    'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли',
    'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь',
    'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей',
    'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя',
    'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже',
    'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому',
    'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти',
    'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех',
    'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть',
    'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего',
    'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем',
    'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том',
    'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
] + CUSTOM_STOPWORDS

# ============================================================
# ПАРАМЕТРЫ TF-IDF ВЕКТОРИЗАЦИИ
# ============================================================

TFIDF_CONFIG = {
    "max_features": 7000,
    "ngram_range": (1, 2),  # Униграммы и биграммы
    "min_df": 2,            # Минимальная частота документа
    "max_df": 0.7,          # Максимальная частота документа (игнорируем слишком частые)
    "sublinear_tf": True,   # 1 + log(tf)
    "norm": "l2",           # Нормализация
    "use_idf": True,
    "smooth_idf": True,
}

# ============================================================
# ПАРАМЕТРЫ МОДЕЛИ
# ============================================================

MODEL_CONFIG = {
    "classifier_type": "logistic_regression",
    "C": 1.0,                    # Регуляризация
    "max_iter": 1000,
    "class_weight": "balanced",  # Учитываем дисбаланс классов
    "solver": "lbfgs",
    "multi_class": "auto",
    "random_state": 42,
    "n_jobs": -1,               # Использовать все ядра
}

# Версионирование модели
MODEL_NAME = "product_classifier"
MODEL_VERSION = "1.0.0"

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
# ПАРАМЕТРЫ CLEANLAB
# ============================================================

CLEANLAB_CONFIG = {
    "cv_folds": 5,
    "confidence_threshold": 0.9,
    "use_tfidf": True,
    "max_features": 5000,
}

# ============================================================
# ПАРАМЕТРЫ АКТИВНОГО ОБУЧЕНИЯ
# ============================================================

ACTIVE_LEARNING_CONFIG = {
    "samples_per_cycle": 500,
    "confidence_threshold": 0.7,
    "uncertainty_ratio": 0.4,    # Доля неуверенных примеров
    "margin_ratio": 0.3,         # Доля примеров с близкими вероятностями
    "diversity_ratio": 0.3,      # Доля разнообразных примеров
}

# ============================================================
# ПАРАМЕТРЫ ПРЕДСКАЗАНИЯ
# ============================================================

PREDICTION_CONFIG = {
    "batch_size": 5000,
    "confidence_threshold": 0.7,
    "high_confidence_threshold": 0.9,
}

# ============================================================
# ПАРАМЕТРЫ ЭКСПОРТА
# ============================================================

EXPORT_CONFIG = {
    "include_probabilities": False,
    "create_review_file": True,
    "max_review_samples": 500,
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
    """Получение пути к файлу модели"""
    if version is None:
        version = MODEL_VERSION
    version_str = version.replace(".", "_")
    return MODELS_DIR / f"{MODEL_NAME}_v{version_str}.pkl"


def get_metadata_path() -> Path:
    """Получение пути к файлу метаданных моделей"""
    return MODELS_DIR / "model_metadata.json"


def print_config():
    """Вывод текущей конфигурации"""
    print("="*60)
    print("КОНФИГУРАЦИЯ ПРОЕКТА")
    print("="*60)
    print(f"\nДанные:")
    print(f"  Размеченные: {CLASSIFIED_FILE}")
    print(f"  Неразмеченные: {UNCLASSIFIED_FILE}")
    print(f"\nМодель: {MODEL_NAME} v{MODEL_VERSION}")
    print(f"\nTF-IDF: max_features={TFIDF_CONFIG['max_features']}, "
          f"ngrams={TFIDF_CONFIG['ngram_range']}")
    print(f"\nОбучение: test_size={TRAIN_CONFIG['test_size']}, "
          f"cv_folds={TRAIN_CONFIG['cv_folds']}")
    print(f"\nАктивное обучение: samples_per_cycle={ACTIVE_LEARNING_CONFIG['samples_per_cycle']}")
    print("="*60)


if __name__ == "__main__":
    print_config()