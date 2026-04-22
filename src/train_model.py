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
        self.rare_classes_info: Dict[str, Any] = {}
        
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
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Создаем векторизатор напрямую
        vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            min_df=tfidf_min_df,
            max_df=tfidf_max_df,
            sublinear_tf=True,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            token_pattern=r'(?u)\b\w+\b',
            lowercase=False  # Уже обработано препроцессором
        )

        self.vectorizer = vectorizer  # Сохраняем для совместимости
                
        # Создаем классификатор (без multi_class='auto' для совместимости)
        classifier = LogisticRegression(
            C=classifier_c,
            max_iter=classifier_max_iter,
            class_weight=class_weight,
            solver='lbfgs',
            random_state=42,
            n_jobs=-1
        )
        
        # Собираем пайплайн с ПРАВИЛЬНЫМ векторизатором
        self.pipeline = Pipeline([
            ('vectorizer', vectorizer),
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
        
        Редкие классы (с 1 примером) полностью помещаются в обучающую выборку
        """
        
        if self.pipeline is None:
            raise ValueError("Сначала вызовите build_pipeline()")
        
        # Анализируем распределение классов
        class_counts = labels.value_counts()
        rare_classes = class_counts[class_counts < 2].index.tolist()
        common_classes = class_counts[class_counts >= 2].index.tolist()
        
        logger.info(f"Всего классов: {len(class_counts)}")
        logger.info(f"  - Редких (< 2 примеров): {len(rare_classes)}")
        logger.info(f"  - Обычных (>= 2 примеров): {len(common_classes)}")
        
        if rare_classes:
            logger.info(f"Примеры редких классов: {rare_classes[:10]}")
        
        self.class_names = sorted(labels.unique())
        
        # Если есть редкие классы - особая логика разбиения
        if rare_classes:
            logger.info("Применяется специальная стратегия разбиения для сохранения редких классов")
            
            # Создаем маски
            rare_mask = labels.isin(rare_classes)
            common_mask = ~rare_mask
            
            # Данные с обычными классами (можно стратифицировать)
            texts_common = texts[common_mask]
            labels_common = labels[common_mask]
            
            # Данные с редкими классами
            texts_rare = texts[rare_mask]
            labels_rare = labels[rare_mask]
            
            logger.info(f"Распределение данных:")
            logger.info(f"  - Обычные классы: {len(texts_common)} записей")
            logger.info(f"  - Редкие классы: {len(texts_rare)} записей")
            
            # Разбиваем обычные классы на train/test
            if len(texts_common) > 0:
                try:
                    X_train_common, X_test_common, y_train_common, y_test_common = train_test_split(
                        texts_common, labels_common,
                        test_size=test_size,
                        random_state=random_state,
                        stratify=labels_common
                    )
                except ValueError as e:
                    logger.warning(f"Не удалось стратифицировать обычные классы: {e}")
                    X_train_common, X_test_common, y_train_common, y_test_common = train_test_split(
                        texts_common, labels_common,
                        test_size=test_size,
                        random_state=random_state,
                        stratify=None
                    )
            else:
                # Если обычных классов нет (все классы редкие)
                X_train_common = pd.Series([], dtype=object)
                X_test_common = pd.Series([], dtype=object)
                y_train_common = pd.Series([], dtype=object)
                y_test_common = pd.Series([], dtype=object)
            
            # Редкие классы полностью идут в обучающую выборку
            X_train_rare = texts_rare
            y_train_rare = labels_rare
            X_test_rare = pd.Series([], dtype=object)
            y_test_rare = pd.Series([], dtype=object)
            
            # Объединяем выборки
            X_train = pd.concat([X_train_common, X_train_rare], ignore_index=True)
            X_test = pd.concat([X_test_common, X_test_rare], ignore_index=True)
            y_train = pd.concat([y_train_common, y_train_rare], ignore_index=True)
            y_test = pd.concat([y_test_common, y_test_rare], ignore_index=True)
            
            logger.info(f"Итоговое разбиение:")
            logger.info(f"  - Train: {len(X_train)} записей (включая {len(X_train_rare)} редких)")
            logger.info(f"  - Test: {len(X_test)} записей (только обычные классы)")
            
            # Сохраняем информацию о редких классах
            self.rare_classes_info = {
                'count': len(rare_classes),
                'classes': rare_classes,
                'train_only': True
            }
            
        else:
            # Нет редких классов - обычное разбиение со стратификацией
            logger.info("Все классы имеют >= 2 примеров, обычное разбиение")
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    texts, labels,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=labels
                )
            except ValueError as e:
                logger.warning(f"Не удалось стратифицировать: {e}")
                X_train, X_test, y_train, y_test = train_test_split(
                    texts, labels,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=None
                )
            
            logger.info(f"Train: {len(X_train)} записей, Test: {len(X_test)} записей")
            
            self.rare_classes_info = {
                'count': 0,
                'classes': [],
                'train_only': False
            }
        
        # Проверяем, что в тестовой выборке нет неожиданных классов
        unexpected_in_test = set(y_test.unique()) - set(y_train.unique())
        if unexpected_in_test:
            logger.warning(f"В тестовой выборке есть классы, отсутствующие в обучающей: {unexpected_in_test}")
            logger.info("Это нормально для редких классов, они будут исключены из метрик теста")
        
        # Обучаем модель
        logger.info("Обучение модели...")
        self.pipeline.fit(X_train, y_train)
        self.training_date = datetime.now().isoformat()
        
        # Оценка на тестовой выборке
        # Исключаем из оценки классы, которых нет в обучающей выборке
        train_classes = set(y_train.unique())
        
        # Создаем маску для тестовых примеров, классы которых есть в обучении
        valid_test_mask = y_test.isin(train_classes)
        y_test_valid = y_test[valid_test_mask]
        
        if len(y_test_valid) > 0:
            X_test_valid = X_test[valid_test_mask]
            y_pred_valid = self.pipeline.predict(X_test_valid)
            accuracy = accuracy_score(y_test_valid, y_pred_valid)
            logger.info(f"Accuracy на валидных тестовых данных ({len(y_test_valid)} примеров): {accuracy:.4f}")
        else:
            logger.warning("В тестовой выборке нет классов, известных модели. Accuracy не вычисляется.")
            accuracy = 0.0
            y_pred_valid = []
        
        # Полное предсказание для отчета
        if len(X_test) > 0:
            y_pred_all = self.pipeline.predict(X_test)
        else:
            y_pred_all = []
        
        # Сохраняем метрики
        self.metrics = {
            'test_accuracy': accuracy,
            'test_accuracy_note': 'Только для классов, присутствующих в обучении',
            'train_size': len(X_train),
            'test_size': len(X_test),
            'valid_test_size': len(y_test_valid),
            'n_classes_total': len(self.class_names),
            'n_classes_in_train': len(train_classes),
            'class_distribution': labels.value_counts().to_dict(),
            'rare_classes_info': self.rare_classes_info
        }
        
        # Детальный отчет (только для валидных предсказаний)
        if len(y_test_valid) > 0:
            self._print_classification_report(y_test_valid, y_pred_valid)
        
        # Информация о редких классах
        if rare_classes:
            logger.info("\n" + "="*50)
            logger.info("ИНФОРМАЦИЯ О РЕДКИХ КЛАССАХ")
            logger.info("="*50)
            logger.info(f"Количество редких классов (1 пример): {len(rare_classes)}")
            logger.info("Эти классы присутствуют только в обучающей выборке")
            logger.info("Для новых данных модель сможет их предсказывать,")
            logger.info("но качество на тесте для них не оценивается")
        
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
            zero_division=0
        )
        
        logger.info("\n" + "="*50)
        logger.info("ОТЧЕТ ПО КЛАССИФИКАЦИИ")
        logger.info("="*50)
        for line in report.split('\n'):
            logger.info(line)
        
        # Сохраняем confusion matrix в метрики
        try:
            cm = confusion_matrix(y_true, y_pred)
            self.metrics['confusion_matrix'] = cm.tolist()
        except:
            pass
    
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
            'vectorizer': self.pipeline.named_steps['vectorizer'],
            'class_names': self.class_names,
            'metrics': self.metrics,
            'model_name': self.model_name,
            'version': self.version,
            'training_date': self.training_date,
            'rare_classes_info': self.rare_classes_info
        }
        
        joblib.dump(model_data, model_file)
        logger.info(f"Модель сохранена: {model_file}")
        
        # Обновляем метаданные
        self._update_metadata(metadata_file, model_file.name)
        
        return model_file

    def _update_metadata(self, metadata_file: Path, model_filename: str):
        """Обновление файла с метаданными моделей"""
        
        # Всегда создаем новую структуру, если файл поврежден или отсутствует
        try:
            if metadata_file.exists():
                # Пробуем прочитать как UTF-8
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    # Если не получается - пробуем другие кодировки
                    for encoding in ['cp1251', 'latin-1', 'utf-8-sig']:
                        try:
                            with open(metadata_file, 'r', encoding=encoding) as f:
                                metadata = json.load(f)
                            logger.info(f"Файл метаданных прочитан в кодировке {encoding}")
                            break
                        except:
                            continue
                    else:
                        # Если все кодировки не подошли - создаем новый
                        logger.warning("Не удалось прочитать файл метаданных, создаем новый")
                        metadata = {"models": []}
            else:
                metadata = {"models": []}
        except Exception as e:
            logger.warning(f"Ошибка чтения метаданных: {e}, создаем новый")
            metadata = {"models": []}
        
        # Гарантируем правильную структуру
        if not isinstance(metadata, dict):
            metadata = {"models": []}
        if "models" not in metadata:
            metadata["models"] = []
        
        # Добавляем информацию о новой модели
        # Конвертируем numpy типы в обычные Python типы для JSON сериализации
        metrics_serializable = {}
        for key, value in self.metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_serializable[key] = float(value)
            elif isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            elif isinstance(value, dict):
                # Рекурсивно конвертируем вложенные словари
                metrics_serializable[key] = {
                    k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                    for k, v in value.items()
                }
            else:
                metrics_serializable[key] = value
        
        model_info = {
            "filename": model_filename,
            "model_name": self.model_name,
            "version": self.version,
            "training_date": self.training_date,
            "metrics": metrics_serializable,
            "n_classes": len(self.class_names) if self.class_names else 0,
            "class_names": self.class_names,
            "rare_classes_info": self.rare_classes_info
        }
        
        # Обновляем или добавляем запись
        existing_idx = None
        for i, m in enumerate(metadata["models"]):
            if m.get("version") == self.version:
                existing_idx = i
                break
        
        if existing_idx is not None:
            metadata["models"][existing_idx] = model_info
        else:
            metadata["models"].append(model_info)
        
        # Устанавливаем последнюю версию как активную
        metadata["active_version"] = self.version
        metadata["last_updated"] = datetime.now().isoformat()
        
        # Сохраняем с правильной кодировкой
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Метаданные обновлены: {metadata_file}")
        except Exception as e:
            logger.error(f"Не удалось сохранить метаданные: {e}")
            # Пробуем сохранить без ensure_ascii
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=True)
            logger.info(f"Метаданные сохранены с ASCII-экранированием")
    
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
        classifier.rare_classes_info = model_data.get('rare_classes_info', {})
        
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
    
    # Импортируем config внутри функции
    import config
    
    classifier = TextClassifier(model_name="product_classifier", version=version)
    
    # Берем параметры из config
    classifier.build_pipeline(
        tfidf_max_features=config.TFIDF_CONFIG['max_features'],
        tfidf_ngram_range=config.TFIDF_CONFIG['ngram_range'],
        tfidf_min_df=config.TFIDF_CONFIG['min_df'],
        tfidf_max_df=config.TFIDF_CONFIG['max_df'],
        classifier_c=config.MODEL_CONFIG['C'],
        class_weight=config.MODEL_CONFIG['class_weight']
    )
    
    texts = df_train[text_column]
    labels = df_train[class_column]
    
    classifier.train(texts, labels, test_size=config.TRAIN_CONFIG['test_size'])
    # ...
    
    # Кросс-валидация (только для обычных классов)
    try:
        classifier.cross_validate(texts, labels, cv_folds=5)
    except Exception as e:
        logger.warning(f"Кросс-валидация не выполнена: {e}")
    
    # Важность признаков
    try:
        feature_importance = classifier.get_feature_importance(top_n=15)
        logger.info("\nТоп-5 признаков для первых классов:")
        for class_name in classifier.class_names[:5]:
            class_features = feature_importance[
                (feature_importance['class'] == class_name) & 
                (feature_importance['direction'] == 'positive')
            ].head(5)
            if not class_features.empty:
                logger.info(f"\n  {class_name}:")
                for _, row in class_features.iterrows():
                    logger.info(f"    - {row['feature']}: {row['coefficient']:.3f}")
    except Exception as e:
        logger.warning(f"Не удалось вычислить важность признаков: {e}")
    
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