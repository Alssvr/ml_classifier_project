# experimental/main_sbert.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.data_preprocessing import RussianTextPreprocessor
from src.utils import setup_logging
import experimental.config_sbert as config

logger = setup_logging("sbert_experiment")


class SBERTClassifier:
    """Классификатор на основе SentenceTransformer"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.SBERT_MODEL_NAME
        self.encoder = None
        self.classifier = None
        self.class_names = None
    
    def load_encoder(self):
        """Загрузка SBERT модели"""
        logger.info(f"Загрузка SBERT модели: {self.model_name}")
        self.encoder = SentenceTransformer(self.model_name)
        logger.info(f"Размерность эмбеддингов: {self.encoder.get_sentence_embedding_dimension()}")
    
    def encode(self, texts: pd.Series, batch_size: int = 32) -> np.ndarray:
        """Преобразование текстов в эмбеддинги"""
        logger.info(f"Кодирование {len(texts)} текстов...")
        return self.encoder.encode(texts.tolist(), batch_size=batch_size, show_progress_bar=True)
    
    def train(self, texts: pd.Series, labels: pd.Series):
        """Обучение классификатора"""
        self.class_names = sorted(labels.unique())
        
        # Обработка редких классов (как в TF-IDF версии)
        class_counts = labels.value_counts()
        rare_classes = class_counts[class_counts < 2].index.tolist()
        
        if rare_classes:
            logger.info(f"Редких классов (< 2 примеров): {len(rare_classes)}")
            
            # Маски для редких и обычных классов
            rare_mask = labels.isin(rare_classes)
            common_mask = ~rare_mask    
        
            # Кодируем тексты
            X = self.encode(texts)
            
            # Разбиваем обычные классы
            X_train, X_test, y_train, y_test = train_test_split(
                X_common, y_common,
                test_size=config.TRAIN_CONFIG['test_size'],
                random_state=config.TRAIN_CONFIG['random_state'],
                stratify=y_common
            )
            
            # Добавляем редкие в обучение
            X_train = np.vstack([X_train, X_rare]) if len(X_rare) > 0 else X_train
            y_train = pd.concat([y_train, y_rare], ignore_index=True)
            
            logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        else:
            # Нет редких классов — обычное разбиение
            X = self.encode(texts)
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels,
                test_size=config.TRAIN_CONFIG['test_size'],
                random_state=config.TRAIN_CONFIG['random_state'],
                stratify=labels
            )
            
            # Обучаем классификатор
            self.classifier = LogisticRegression(
                C=config.CLASSIFIER_CONFIG['C'],
                max_iter=config.CLASSIFIER_CONFIG['max_iter'],
                class_weight=config.CLASSIFIER_CONFIG['class_weight'],
                random_state=42
            )
            
            self.classifier.fit(X_train, y_train)
            
            # Оцениваем (только для классов, известных модели)
            train_classes = set(y_train.unique())
            valid_mask = y_test.isin(train_classes)
            y_test_valid = y_test[valid_mask]
            
            if len(y_test_valid) > 0:
                X_test_valid = X_test[valid_mask]
                y_pred = self.classifier.predict(X_test_valid)
                accuracy = accuracy_score(y_test_valid, y_pred)
            else:
                accuracy = 0.0
            
            logger.info(f"Accuracy: {accuracy:.4f}")
            
            return {'accuracy': accuracy, 'n_classes': len(self.class_names)}
        
    def predict(self, texts: pd.Series) -> np.ndarray:
        """Предсказание"""
        X = self.encode(texts)
        return self.classifier.predict(X)
    
    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        """Предсказание с вероятностями"""
        X = self.encode(texts)
        return self.classifier.predict_proba(X)
    
    def save(self, path: Path):
        """Сохранение модели"""
        joblib.dump({
            'classifier': self.classifier,
            'class_names': self.class_names,
            'model_name': self.model_name
        }, path)
        logger.info(f"Модель сохранена: {path}")
    
    @staticmethod
    def load(path: Path, encoder_name: str = None):
        """Загрузка модели"""
        data = joblib.load(path)
        clf = SBERTClassifier(data['model_name'])
        clf.encoder = SentenceTransformer(data['model_name'])
        clf.classifier = data['classifier']
        clf.class_names = data['class_names']
        return clf


def train_sbert():
    """Обучение SBERT модели"""
    print("="*60)
    print("ОБУЧЕНИЕ SBERT МОДЕЛИ")
    print("="*60)
    
    # Загружаем данные
    df = pd.read_excel(config.CLASSIFIED_FILE)
    logger.info(f"Загружено {len(df)} записей")
    
    # Препроцессор
    preprocessor = RussianTextPreprocessor(use_lemmatization=True)
    df['text_processed'] = df[config.TEXT_COLUMN].apply(preprocessor.process_text)
    
    # Создаём классификатор
    clf = SBERTClassifier()
    clf.load_encoder()
    
    # Обучаем
    metrics = clf.train(df['text_processed'], df[config.LABEL_COLUMN])
    
    # Сохраняем
    model_path = config.MODELS_DIR / f"sbert_{config.MODEL_VERSION}.pkl"
    clf.save(model_path)
    
    print(f"\nМодель: SBERT {config.MODEL_VERSION}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Классов: {metrics['n_classes']}")
    
    return clf


def predict_sbert():
    """Классификация SBERT моделью"""
    # Загружаем модель
    model_path = config.MODELS_DIR / f"sbert_{config.MODEL_VERSION}.pkl"
    clf = SBERTClassifier.load(model_path)
    
    # Загружаем данные
    df = pd.read_excel(config.UNCLASSIFIED_FILE)
    preprocessor = RussianTextPreprocessor(use_lemmatization=True)
    df['text_processed'] = df[config.TEXT_COLUMN].apply(preprocessor.process_text)
    
    # Предсказываем
    predictions = clf.predict(df['text_processed'])
    probabilities = clf.predict_proba(df['text_processed'])
    confidence = probabilities.max(axis=1)
    
    df['predicted_class'] = predictions
    df['confidence'] = confidence
    
    # Сохраняем
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = config.RESULTS_DIR / f"sbert_classified_{timestamp}.xlsx"
    df.to_excel(output_path, index=False)
    
    print(f"Результаты сохранены: {output_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_sbert()
    else:
        predict_sbert()