# src/utils.py
import logging
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

def setup_logging(name: str = "text_classifier") -> logging.Logger:
    """Настройка логирования"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Обработчик для файла
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(
        log_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log",
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def load_excel_with_progress(filepath: Path, sheet_name=0) -> pd.DataFrame:
    """Загрузка Excel с индикацией прогресса"""
    logger = logging.getLogger("text_classifier")
    logger.info(f"Загрузка файла: {filepath}")
    
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        logger.info(f"Загружено {len(df):,} записей, колонки: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")
        raise

def save_excel_safe(df: pd.DataFrame, filepath: Path, **kwargs):
    """Безопасное сохранение Excel (создает директорию если нужно)"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(filepath, index=False, **kwargs)
    logger = logging.getLogger("text_classifier")
    logger.info(f"Файл сохранен: {filepath} ({len(df):,} записей)")

def clean_text(text: str) -> str:
    """Базовая очистка текста"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    # Приводим к нижнему регистру
    text = text.lower()
    # Заменяем множественные пробелы на один
    text = ' '.join(text.split())
    # Убираем лишние символы (можно расширить)
    for char in ['\n', '\r', '\t', '"', "'"]:
        text = text.replace(char, ' ')
    
    return text.strip()