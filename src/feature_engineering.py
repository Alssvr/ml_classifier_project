# src/feature_engineering.py
import joblib
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from src.utils import setup_logging

logger = setup_logging()

class TechnicalTfidfVectorizer:
    """TF-IDF векторизатор с учетом технических терминов"""
    
    def __init__(self, 
                 max_features: int = 7000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2,
                 max_df: float = 0.7):
        
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        # Создаем векторизатор с оптимизированными настройками
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,      # 1 + log(tf)
            strip_accents=None,     # Не удаляем акценты (важно для русских букв ё)
            lowercase=False,        # Уже привели к нижнему регистру в препроцессоре
            token_pattern=r'(?u)\b\w+\b',  # Слова с буквами и цифрами
            analyzer='word',
            norm='l2',
            use_idf=True,
            smooth_idf=True
        )
        
        self.feature_names: Optional[List[str]] = None
        self.idf_weights: Optional[np.ndarray] = None
        
        logger.info(f"TF-IDF инициализирован: max_features={max_features}, "
                   f"ngrams={ngram_range}")
    
    def fit(self, texts: pd.Series) -> 'TechnicalTfidfVectorizer':
        """Обучение векторизатора"""
        logger.info(f"Обучение TF-IDF на {len(texts):,} текстах...")
        
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.idf_weights = self.vectorizer.idf_
        
        logger.info(f"Размер словаря: {len(self.feature_names):,} признаков")
        
        # Выводим топ признаков по IDF (самые редкие/специфичные)
        top_idf_indices = np.argsort(self.idf_weights)[-10:][::-1]
        logger.info("Топ-10 самых специфичных признаков (высокий IDF):")
        for idx in top_idf_indices:
            logger.info(f"  {self.feature_names[idx]}: IDF={self.idf_weights[idx]:.3f}")
        
        return self
    
    def transform(self, texts: pd.Series) -> csr_matrix:
        """Трансформация текстов в TF-IDF матрицу"""
        logger.info(f"Трансформация {len(texts):,} текстов...")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: pd.Series) -> csr_matrix:
        """Обучение и трансформация"""
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_importance(self, 
                               classifier_coef: np.ndarray,
                               class_names: List[str],
                               top_n: int = 20) -> pd.DataFrame:
        """Получение важности признаков для каждого класса"""
        
        importance_data = []
        
        for class_idx, class_name in enumerate(class_names):
            if len(classifier_coef.shape) == 1:
                # Бинарная классификация
                coef = classifier_coef
            else:
                # Мультикласс
                coef = classifier_coef[class_idx]
            
            # Топ положительные признаки
            top_pos_idx = np.argsort(coef)[-top_n:][::-1]
            for rank, idx in enumerate(top_pos_idx, 1):
                if coef[idx] > 0:
                    importance_data.append({
                        'class': class_name,
                        'feature': self.feature_names[idx],
                        'weight': coef[idx],
                        'rank': rank,
                        'direction': 'positive'
                    })
            
            # Топ отрицательные признаки
            top_neg_idx = np.argsort(coef)[:top_n]
            for rank, idx in enumerate(top_neg_idx, 1):
                if coef[idx] < 0:
                    importance_data.append({
                        'class': class_name,
                        'feature': self.feature_names[idx],
                        'weight': abs(coef[idx]),
                        'rank': rank,
                        'direction': 'negative'
                    })
        
        return pd.DataFrame(importance_data)
    
    def save(self, path: str):
        """Сохранение векторизатора"""
        joblib.dump(self, path)
        logger.info(f"Векторизатор сохранен: {path}")
    
    @staticmethod
    def load(path: str) -> 'TechnicalTfidfVectorizer':
        """Загрузка векторизатора"""
        vectorizer = joblib.load(path)
        logger.info(f"Векторизатор загружен: {path}")
        return vectorizer


def analyze_tfidf_matrix(matrix: csr_matrix, 
                         feature_names: List[str],
                         texts: pd.Series,
                         sample_size: int = 5) -> pd.DataFrame:
    """Анализ TF-IDF матрицы для отладки"""
    
    # Сумма TF-IDF по всем документам
    total_importance = np.array(matrix.sum(axis=0)).flatten()
    
    # Топ-20 самых важных слов во всей коллекции
    top_indices = np.argsort(total_importance)[-20:][::-1]
    
    analysis = []
    for idx in top_indices:
        # Находим документы где этот признак максимален
        col = matrix[:, idx].toarray().flatten()
        top_doc_indices = np.argsort(col)[-sample_size:][::-1]
        
        analysis.append({
            'feature': feature_names[idx],
            'total_weight': total_importance[idx],
            'avg_weight': col.mean(),
            'max_weight': col.max(),
            'top_docs': ' | '.join([
                texts.iloc[i][:50] + '...' for i in top_doc_indices if col[i] > 0
            ][:3])
        })
    
    return pd.DataFrame(analysis)


if __name__ == "__main__":
    # Тестирование
    from src.data_preprocessing import RussianTextPreprocessor
    from config import CLASSIFIED_FILE
    
    # Загружаем тестовые данные
    df = pd.read_excel(CLASSIFIED_FILE, nrows=1000)
    
    # Препроцессинг
    preprocessor = RussianTextPreprocessor(use_lemmatization=False)
    processed = preprocessor.process_dataframe(
        df, 
        text_column='Наименование',
        class_column='Шаблон класса'
    )
    
    # TF-IDF
    vectorizer = TechnicalTfidfVectorizer(max_features=500)
    tfidf_matrix = vectorizer.fit_transform(processed['text_processed'])
    
    print(f"\nРазмер матрицы: {tfidf_matrix.shape}")
    print(f"Разреженность: {(1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100:.2f}%")
    
    # Анализ
    analysis = analyze_tfidf_matrix(
        tfidf_matrix, 
        vectorizer.feature_names,
        processed['text_processed']
    )
    print("\nТоп-10 признаков:")
    print(analysis.head(10))