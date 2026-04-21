# src/data_preprocessing.py
import re
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import pymorphy3
from nltk.corpus import stopwords
import nltk
from src.utils import setup_logging, clean_text

# Скачиваем стоп-слова NLTK (один раз)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = setup_logging()

class RussianTextPreprocessor:
    """Препроцессор для русских технических текстов"""
    
    def __init__(self, 
                 use_lemmatization: bool = True,
                 remove_numbers: bool = False,  # В тех.названиях числа важны!
                 custom_stopwords: Optional[List[str]] = None):
        
        self.use_lemmatization = use_lemmatization
        self.remove_numbers = remove_numbers
        self.morph = pymorphy3.MorphAnalyzer() if use_lemmatization else None
        
        # Загружаем стоп-слова
        self.stopwords = set(stopwords.words('russian'))
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
            
        # Специальные паттерны для технических текстов
        self.tech_patterns = {
            'dimensions': re.compile(r'\d+[xх×]\d+'),  # 3x13x19
            'gost': re.compile(r'ГОСТ\s*\d+-\d+', re.IGNORECASE),
            'model': re.compile(r'[A-Z]{2,}-\d+[A-Z]*'),  # DS-2CD2347G2
        }
        
        logger.info(f"Препроцессор инициализирован. Лемматизация: {use_lemmatization}")
    
    def lemmatize_word(self, word: str) -> str:
        """Лемматизация одного слова"""
        if not self.use_lemmatization or len(word) < 3:
            return word
        
        try:
            parsed = self.morph.parse(word)[0]
            return parsed.normal_form
        except Exception as e:
            logger.debug(f"Ошибка лемматизации '{word}': {e}")
            return word
    
    def extract_tech_features(self, text: str) -> str:
        """Извлечение и нормализация технических особенностей"""
        # Сохраняем размеры в едином формате
        text = self.tech_patterns['dimensions'].sub(
            lambda m: m.group().replace('х', 'x').replace('×', 'x'), 
            text
        )
        # Нормализуем ГОСТ
        text = self.tech_patterns['gost'].sub(
            lambda m: m.group().upper().replace(' ', ''), 
            text
        )
        return text
    
    def process_text(self, text: str) -> str:
        """Полный цикл обработки одного текста"""
        if pd.isna(text) or text == "":
            return ""
        
        # Базовая очистка
        text = clean_text(text)
        
        # Сохраняем технические паттерны (временно заменяем на токены)
        text = self.extract_tech_features(text)
        
        # Разбиваем на слова
        words = text.split()
        
        # Фильтрация и лемматизация
        processed_words = []
        for word in words:
            # Пропускаем стоп-слова
            if word.lower() in self.stopwords:
                continue
            
            # Пропускаем короткие слова (если не числа)
            if len(word) < 2 and not word.isdigit():
                continue
            
            # Удаляем числа если нужно (но в тех.текстах лучше оставлять)
            if self.remove_numbers and word.isdigit():
                continue
            
            # Лемматизация
            if self.use_lemmatization:
                word = self.lemmatize_word(word)
            
            processed_words.append(word)
        
        return ' '.join(processed_words)
    
    def process_dataframe(self, 
                          df: pd.DataFrame, 
                          text_column: str,
                          id_column: str = 'ID',
                          class_column: Optional[str] = None) -> pd.DataFrame:
        """Обработка всего датафрейма"""
        logger.info(f"Начало обработки {len(df)} записей...")
        
        df_processed = df.copy()
        
        # Обрабатываем тексты
        df_processed['text_processed'] = df_processed[text_column].apply(self.process_text)
        
        # Удаляем пустые строки после обработки
        empty_before = len(df_processed)
        df_processed = df_processed[df_processed['text_processed'] != ""]
        empty_after = len(df_processed)
        
        if empty_before > empty_after:
            logger.warning(f"Удалено {empty_before - empty_after} пустых записей")
        
        # Статистика
        original_words = df[text_column].astype(str).str.split().str.len().sum()
        processed_words = df_processed['text_processed'].str.split().str.len().sum()
        
        logger.info(f"Обработка завершена:")
        logger.info(f"  - Исходное кол-во слов: {original_words:,}")
        logger.info(f"  - После обработки: {processed_words:,}")
        logger.info(f"  - Сжатие словаря: {(1 - processed_words/original_words)*100:.1f}%")
        
        # Формируем выходной датафрейм
        output_columns = [id_column, text_column, 'text_processed']
        if class_column and class_column in df_processed.columns:
            output_columns.append(class_column)
        
        return df_processed[output_columns]
    
    def get_vocabulary_stats(self, texts: pd.Series, top_n: int = 20) -> pd.DataFrame:
        """Статистика по словарю после обработки"""
        from collections import Counter
        
        all_words = []
        for text in texts:
            all_words.extend(text.split())
        
        word_counts = Counter(all_words)
        
        stats_df = pd.DataFrame(
            word_counts.most_common(top_n),
            columns=['word', 'frequency']
        )
        stats_df['percentage'] = (stats_df['frequency'] / len(all_words)) * 100
        
        return stats_df


def prepare_training_data(df: pd.DataFrame, 
                          text_col: str = 'Наименование',
                          class_col: str = 'Шаблон класса',
                          id_col: str = 'ID') -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Подготовка данных для обучения"""
    
    # Проверяем наличие колонок
    required_cols = [id_col, text_col, class_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")
    
    # Удаляем дубликаты по ID
    df = df.drop_duplicates(subset=[id_col], keep='first')
    
    # Удаляем строки с пустым классом
    df = df.dropna(subset=[class_col])
    
    # Возвращаем ID, тексты и классы отдельно
    return df[id_col], df[text_col], df[class_col]


if __name__ == "__main__":
    # Тестирование модуля
    from config import CLASSIFIED_FILE
    
    # Загружаем тестовые данные
    test_df = pd.read_excel(CLASSIFIED_FILE, nrows=100)
    
    # Создаем препроцессор
    preprocessor = RussianTextPreprocessor(use_lemmatization=True)
    
    # Обрабатываем
    processed = preprocessor.process_dataframe(
        test_df, 
        text_column='Наименование',
        class_column='Шаблон класса'
    )
    
    print("\nПримеры обработки:")
    print(processed[['Наименование', 'text_processed']].head(10))
    
    # Статистика по словарю
    vocab_stats = preprocessor.get_vocabulary_stats(processed['text_processed'])
    print("\nТоп-20 слов после обработки:")
    print(vocab_stats)