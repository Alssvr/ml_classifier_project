# src/train_model.py
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

from src.utils import setup_logging, save_excel_safe
from src.feature_engineering import TechnicalTfidfVectorizer

logger = setup_logging()


class TextClassifier:
    """Основной класс для обучения и управления моделью классификации"""
    
    def __init__(self, 
                 model_name: str = "text_classifier",
                 version: str = "1.0.0"):
        
        self.model_name = model_name
        self.version = version
        self.pipeline: Optional[Pipeline] = None
        self.vectorizer: Optional[TechnicalTfidfVectorizer] = None
        self.class_names: Optional[List[str]] = None
        self.metrics: Dict[str, Any] = {}
        self.training_date: Optional[str] = None
        
        logger.info(f"Инициализация классификатора {model_name} v{version}")
    
    def build_pipeline(self, 
                       tfidf_max_features: int = 7000,
                       tfidf_ngram_range: Tuple[int, int] = (1, 2),
                       tfidf_min_df: int = 2,
                       tfidf_max_df: float = 0.7,
                       classifier_c: float = 1.0,
                       classifier_max_iter: int = 1000,
                       class_weight: str = 'balanced') -> Pipeline:
        """
        Построение пайплайна: TF-IDF векторизация + Логистическая регрессия
        """
        
        # Создаем векторизатор
        self.vectorizer = TechnicalTfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            min_df=tfidf_min_df,
            max_df=tfidf_max_df
        )
        
        # Создаем классификатор
        classifier = LogisticRegression(
            C=classifier_c,
            max_iter=classifier_max_iter,
            class_weight=class_weight,
            solver='lbfgs',
            multi_class='auto',
            random_state=42,
            n_jobs=-1
        )
        
        # Собираем пайплайн
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer.vectorizer),
            ('classifier', classifier)
        ])
        
        logger.info(f"Пайплайн построен:")
        logger.info(f"  - TF-IDF: max_features={tfidf_max_features}, ngrams={tfidf_ngram_range}")
        logger.info(f"  - Classifier: C={classifier_c}, class_weight={class_weight}")
        
        return self.pipeline
    
    def train(self, 
              texts: pd.Series, 
              labels: pd.Series,
              test_size: float = 0.2,
              random_state: int = 42) -> Dict[str, Any]:
        """
        Обучение модели с разбиением на train/test
        """
        
        if self.pipeline is None:
            raise ValueError("Сначала вызовите build_pipeline()")
        
        self.class_names = sorted(labels.unique())
        logger.info(f"Обучение модели. Классов: {len(self.class_names)}")
        logger.info(f"Распределение классов:\n{labels.value_counts()}")
        
        # Разбиваем данные
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        logger.info(f"Train: {len(X_train)} записей, Test: {len(X_test)} записей")
        
        # Обучаем
        logger.info("Обучение модели...")
        self.pipeline.fit(X_train, y_train)
        self.training_date = datetime.now().isoformat()
        
        # Оценка на тестовой выборке
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Обучение завершено. Accuracy на тесте: {accuracy:.4f}")
        
        # Сохраняем метрики
        self.metrics = {
            'test_accuracy': accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_classes': len(self.class_names),
            'class_distribution': labels.value_counts().to_dict()
        }
        
        # Детальный отчет
        self._print_classification_report(y_test, y_pred)
        
        return self.metrics
    
    def cross_validate(self, 
                       texts: pd.Series, 
                       labels: pd.Series,
                       cv_folds: int = 5) -> Dict[str, float]:
        """
        Кросс-валидация для оценки стабильности модели
        """
        
        if self.pipeline is None:
            raise ValueError("Сначала вызовите build_pipeline()")
        
        logger.info(f"Запуск кросс-валидации ({cv_folds} фолдов)...")
        
        # Создаем полный пайплайн с векторизацией
        full_pipeline = Pipeline([
            ('vectorizer', self.vectorizer.vectorizer),
            ('classifier', self.pipeline.named_steps['classifier'])
        ])
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        scores = cross_val_score(
            full_pipeline, 
            texts, 
            labels, 
            cv=cv, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        cv_results = {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'min_accuracy': scores.min(),
            'max_accuracy': scores.max(),
            'scores': scores.tolist()
        }
        
        self.metrics['cv_results'] = cv_results
        
        logger.info(f"Кросс-валидация завершена:")
        logger.info(f"  Mean accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy']:.4f})")
        logger.info(f"  Range: [{cv_results['min_accuracy']:.4f}, {cv_results['max_accuracy']:.4f}]")
        
        return cv_results
    
    def predict(self, texts: pd.Series) -> np.ndarray:
        """Предсказание классов для новых текстов"""
        
        if self.pipeline is None:
            raise ValueError("Модель не обучена")
        
        logger.info(f"Предсказание для {len(texts)} текстов...")
        return self.pipeline.predict(texts)
    
    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        """Получение вероятностей для каждого класса"""
        
        if self.pipeline is None:
            raise ValueError("Модель не обучена")
        
        return self.pipeline.predict_proba(texts)
    
    def predict_with_confidence(self, texts: pd.Series) -> pd.DataFrame:
        """
        Предсказание с оценкой уверенности
        Возвращает DataFrame с колонками:
        - predicted_class
        - confidence (максимальная вероятность)
        - all_probabilities (вероятности всех классов)
        """
        
        probas = self.predict_proba(texts)
        predicted_indices = np.argmax(probas, axis=1)
        confidences = np.max(probas, axis=1)
        
        results = []
        for i, (pred_idx, conf, proba) in enumerate(zip(predicted_indices, confidences, probas)):
            # Создаем словарь вероятностей по классам
            class_probas = {
                self.class_names[j]: float(p) 
                for j, p in enumerate(proba)
            }
            
            results.append({
                'index': i,
                'predicted_class': self.class_names[pred_idx],
                'confidence': float(conf),
                'is_confident': conf > 0.7,  # Порог уверенности
                'second_guess': self._get_second_guess(proba, pred_idx),
                'class_probabilities': class_probas
            })
        
        return pd.DataFrame(results)
    
    def _get_second_guess(self, proba: np.ndarray, pred_idx: int) -> str:
        """Получение второго по вероятности класса"""
        sorted_indices = np.argsort(proba)[::-1]
        if len(sorted_indices) > 1:
            second_idx = sorted_indices[1]
            return self.class_names[second_idx]
        return "N/A"
    
    def _print_classification_report(self, y_true: pd.Series, y_pred: np.ndarray):
        """Вывод детального отчета по классификации"""
        
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            zero_division=0
        )
        
        logger.info("\n" + "="*50)
        logger.info("ОТЧЕТ ПО КЛАССИФИКАЦИИ")
        logger.info("="*50)
        for line in report.split('\n'):
            logger.info(line)
        
        # Сохраняем confusion matrix в метрики
        cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
        self.metrics['confusion_matrix'] = cm.tolist()
        self.metrics['class_names'] = self.class_names
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Получение важных признаков для каждого класса
        """
        if self.pipeline is None:
            raise ValueError("Модель не обучена")
        
        classifier = self.pipeline.named_steps['classifier']
        
        # Получаем имена признаков от векторизатора
        feature_names = self.pipeline.named_steps['vectorizer'].get_feature_names_out()
        
        # Получаем коэффициенты модели
        coef = classifier.coef_
        
        importance_data = []
        
        for class_idx, class_name in enumerate(self.class_names):
            if len(self.class_names) == 2 and len(coef.shape) == 1:
                class_coef = coef
            else:
                class_coef = coef[class_idx]
            
            # Топ положительные признаки
            top_pos_idx = np.argsort(class_coef)[-top_n:][::-1]
            for rank, idx in enumerate(top_pos_idx, 1):
                if class_coef[idx] > 0:
                    importance_data.append({
                        'class': class_name,
                        'feature': feature_names[idx],
                        'coefficient': float(class_coef[idx]),
                        'rank': rank,
                        'direction': 'positive'
                    })
            
            # Топ отрицательные признаки
            top_neg_idx = np.argsort(class_coef)[:top_n]
            for rank, idx in enumerate(top_neg_idx, 1):
                if class_coef[idx] < 0:
                    importance_data.append({
                        'class': class_name,
                        'feature': feature_names[idx],
                        'coefficient': float(abs(class_coef[idx])),
                        'rank': rank,
                        'direction': 'negative'
                    })
        
        return pd.DataFrame(importance_data)
    
    def save(self, models_dir: Path):
        """Сохранение модели и метаданных"""
        
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Имена файлов с версией
        model_file = models_dir / f"{self.model_name}_v{self.version.replace('.', '_')}.pkl"
        metadata_file = models_dir / "model_metadata.json"
        
        # Сохраняем модель
        model_data = {
            'pipeline': self.pipeline,
            'vectorizer': self.vectorizer,
            'class_names': self.class_names,
            'metrics': self.metrics,
            'model_name': self.model_name,
            'version': self.version,
            'training_date': self.training_date
        }
        
        joblib.dump(model_data, model_file)
        logger.info(f"Модель сохранена: {model_file}")
        
        # Обновляем метаданные
        self._update_metadata(metadata_file, model_file.name)
        
        return model_file
    
    def _update_metadata(self, metadata_file: Path, model_filename: str):
        """Обновление файла с метаданными моделей"""
        
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {"models": []}
        
        # Добавляем информацию о новой модели
        model_info = {
            "filename": model_filename,
            "model_name": self.model_name,
            "version": self.version,
            "training_date": self.training_date,
            "metrics": self.metrics,
            "n_classes": len(self.class_names) if self.class_names else 0,
            "class_names": self.class_names
        }
        
        # Обновляем или добавляем запись
        existing_idx = None
        for i, m in enumerate(metadata["models"]):
            if m["version"] == self.version:
                existing_idx = i
                break
        
        if existing_idx is not None:
            metadata["models"][existing_idx] = model_info
        else:
            metadata["models"].append(model_info)
        
        # Устанавливаем последнюю версию как активную
        metadata["active_version"] = self.version
        metadata["last_updated"] = datetime.now().isoformat()
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Метаданные обновлены: {metadata_file}")
    
    @staticmethod
    def load(models_dir: Path, version: Optional[str] = None) -> 'TextClassifier':
        """
        Загрузка модели
        
        Args:
            models_dir: директория с моделями
            version: версия для загрузки (если None - загружается последняя активная)
        """
        
        models_dir = Path(models_dir)
        metadata_file = models_dir / "model_metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Файл метаданных не найден: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Определяем версию для загрузки
        if version is None:
            version = metadata.get("active_version")
            if version is None:
                raise ValueError("Не указана версия и нет активной версии в метаданных")
        
        # Находим файл модели
        model_info = None
        for m in metadata["models"]:
            if m["version"] == version:
                model_info = m
                break
        
        if model_info is None:
            raise ValueError(f"Версия {version} не найдена в метаданных")
        
        model_file = models_dir / model_info["filename"]
        
        if not model_file.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_file}")
        
        # Загружаем модель
        model_data = joblib.load(model_file)
        
        # Создаем экземпляр классификатора
        classifier = TextClassifier(
            model_name=model_data['model_name'],
            version=model_data['version']
        )
        
        classifier.pipeline = model_data['pipeline']
        classifier.vectorizer = model_data['vectorizer']
        classifier.class_names = model_data['class_names']
        classifier.metrics = model_data['metrics']
        classifier.training_date = model_data['training_date']
        
        logger.info(f"Модель загружена: {model_file}")
        logger.info(f"  Версия: {classifier.version}")
        logger.info(f"  Обучена: {classifier.training_date}")
        logger.info(f"  Accuracy: {classifier.metrics.get('test_accuracy', 'N/A')}")
        
        return classifier


def train_and_evaluate_model(
    df_train: pd.DataFrame,
    text_column: str = 'text_processed',
    class_column: str = 'Шаблон класса',
    models_dir: Path = Path('models'),
    version: str = '1.0.0'
) -> TextClassifier:
    """
    Полный цикл обучения и оценки модели
    """
    
    # Создаем классификатор
    classifier = TextClassifier(model_name="product_classifier", version=version)
    
    # Строим пайплайн
    classifier.build_pipeline(
        tfidf_max_features=7000,
        tfidf_ngram_range=(1, 2),
        tfidf_min_df=2,
        tfidf_max_df=0.7,
        classifier_c=1.0,
        class_weight='balanced'
    )
    
    # Обучаем
    texts = df_train[text_column]
    labels = df_train[class_column]
    
    classifier.train(texts, labels, test_size=0.2)
    
    # Кросс-валидация
    classifier.cross_validate(texts, labels, cv_folds=5)
    
    # Важность признаков
    feature_importance = classifier.get_feature_importance(top_n=15)
    logger.info("\nТоп-5 признаков для каждого класса:")
    for class_name in classifier.class_names[:5]:  # Первые 5 классов
        class_features = feature_importance[
            (feature_importance['class'] == class_name) & 
            (feature_importance['direction'] == 'positive')
        ].head(5)
        if not class_features.empty:
            logger.info(f"\n  {class_name}:")
            for _, row in class_features.iterrows():
                logger.info(f"    - {row['feature']}: {row['coefficient']:.3f}")
    
    # Сохраняем модель
    classifier.save(models_dir)
    
    return classifier


if __name__ == "__main__":
    # Тестирование модуля
    from src.data_preprocessing import RussianTextPreprocessor
    from config import CLASSIFIED_FILE, MODELS_DIR
    
    # Загружаем и обрабатываем данные
    df = pd.read_excel(CLASSIFIED_FILE, nrows=5000)
    
    preprocessor = RussianTextPreprocessor(use_lemmatization=False)
    processed = preprocessor.process_dataframe(
        df,
        text_column='Наименование',
        class_column='Шаблон класса'
    )
    
    # Обучаем модель
    classifier = train_and_evaluate_model(
        processed,
        text_column='text_processed',
        class_column='Шаблон класса',
        models_dir=MODELS_DIR,
        version='1.0.0-test'
    )
    
    # Тестовое предсказание
    test_texts = [
        "винт гребной 3 лопасти",
        "фланец стальной ГОСТ 12820",
        "камера видеонаблюдения уличная"
    ]
    
    # Препроцессинг
    test_processed = [preprocessor.process_text(t) for t in test_texts]
    
    # Предсказание
    predictions = classifier.predict_with_confidence(pd.Series(test_processed))
    
    print("\nТестовые предсказания:")
    for i, text in enumerate(test_texts):
        row = predictions.iloc[i]
        print(f"\nТекст: {text}")
        print(f"  Класс: {row['predicted_class']}")
        print(f"  Уверенность: {row['confidence']:.2%}")
        print(f"  Второй вариант: {row['second_guess']}")