# src/cleanlab_analysis.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime

from cleanlab.classification import CleanLearning
from cleanlab.filter import find_label_issues
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import setup_logging, save_excel_safe
from src.data_preprocessing import RussianTextPreprocessor

logger = setup_logging()


class LabelNoiseDetector:
    """Класс для обнаружения шума и ошибок в разметке данных"""
    
    def __init__(self, 
                 model_name: str = "label_noise_detector"):
        
        self.model_name = model_name
        self.label_issues: Optional[pd.DataFrame] = None
        self.quality_scores: Optional[pd.DataFrame] = None
        self.statistics: Dict[str, Any] = {}
        
        logger.info(f"Инициализация детектора шума: {model_name}")
    
    def analyze_with_cleanlab(self,
                              texts: pd.Series,
                              labels: pd.Series,
                              ids: Optional[pd.Series] = None,
                              cv_folds: int = 5,
                              use_tfidf: bool = True,
                              max_features: int = 5000) -> pd.DataFrame:
        """
        Анализ качества разметки с помощью Cleanlab
        
        Args:
            texts: тексты документов
            labels: метки классов
            ids: идентификаторы записей
            cv_folds: количество фолдов для кросс-валидации
            use_tfidf: использовать TF-IDF векторизацию
            max_features: максимальное количество признаков для TF-IDF
        
        Returns:
            DataFrame с оценками качества для каждой записи
        """
        
        logger.info(f"Запуск анализа Cleanlab на {len(texts)} записях...")
        
        # Подготовка признаков
        if use_tfidf:
            logger.info("Векторизация текстов с помощью TF-IDF...")
            vectorizer = TfidfVectorizer(max_features=max_features)
            X = vectorizer.fit_transform(texts)
        else:
            # Если тексты уже в числовом формате
            X = texts
        
        # Создаем классификатор
        clf = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Оборачиваем в CleanLearning
        logger.info(f"Запуск CleanLearning с {cv_folds}-фолд кросс-валидацией...")
        cl = CleanLearning(clf, cv_n_folds=cv_folds)
        
        # Обучаем и получаем label issues
        _ = cl.fit(X, labels)
        
        # Получаем детальную информацию о качестве меток
        label_issues_df = cl.get_label_issues()
        
        # Добавляем дополнительную информацию
        if ids is not None:
            label_issues_df['ID'] = ids.values
        
        label_issues_df['text'] = texts.values
        label_issues_df['given_label'] = labels.values
        
        # Получаем предсказанные метки от cleanlab
        try:
            pred_probs = cl.predict_proba(X)
            pred_labels = cl.clf.classes_[np.argmax(pred_probs, axis=1)]
            label_issues_df['predicted_label'] = pred_labels
            
            # Максимальная вероятность
            label_issues_df['confidence'] = np.max(pred_probs, axis=1)
        except:
            logger.warning("Не удалось получить предсказанные метки")
        
        self.label_issues = label_issues_df
        self.quality_scores = label_issues_df[['label_quality']].copy() if 'label_quality' in label_issues_df.columns else None
        
        # Собираем статистику
        self._calculate_statistics(label_issues_df, labels)
        
        logger.info("Анализ Cleanlab завершен")
        self._print_summary()
        
        return label_issues_df
    
    def analyze_with_cross_validation(self,
                                      texts: pd.Series,
                                      labels: pd.Series,
                                      ids: Optional[pd.Series] = None,
                                      cv_folds: int = 5,
                                      confidence_threshold: float = 0.9) -> pd.DataFrame:
        """
        Альтернативный метод анализа через кросс-валидацию
        (не требует установки cleanlab)
        """
        
        logger.info(f"Запуск анализа через кросс-валидацию на {len(texts)} записях...")
        
        # Векторизация
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(texts)
        
        # Классификатор
        clf = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Получаем предсказания через кросс-валидацию
        logger.info(f"Получение предсказаний через {cv_folds}-фолд CV...")
        y_pred = cross_val_predict(clf, X, labels, cv=cv_folds, method='predict')
        
        # Получаем вероятности
        y_proba = cross_val_predict(clf, X, labels, cv=cv_folds, method='predict_proba')
        max_proba = np.max(y_proba, axis=1)
        
        # Определяем подозрительные записи
        is_suspicious = (y_pred != labels) & (max_proba > confidence_threshold)
        is_uncertain = max_proba < 0.5
        
        # Формируем результат
        results_df = pd.DataFrame({
            'text': texts.values,
            'given_label': labels.values,
            'predicted_label': y_pred,
            'confidence': max_proba,
            'is_label_issue': is_suspicious,
            'is_uncertain': is_uncertain,
            'label_quality': 1 - (y_pred != labels).astype(float) * max_proba
        })
        
        if ids is not None:
            results_df.insert(0, 'ID', ids.values)
        
        self.label_issues = results_df
        self.quality_scores = results_df[['label_quality']]
        
        self._calculate_statistics(results_df, labels)
        
        logger.info("Анализ через кросс-валидацию завершен")
        self._print_summary()
        
        return results_df
    
    def _calculate_statistics(self, issues_df: pd.DataFrame, labels: pd.Series):
        """Подсчет статистики по найденным проблемам"""
        
        total = len(issues_df)
        
        # Количество проблемных записей
        if 'is_label_issue' in issues_df.columns:
            issues_count = issues_df['is_label_issue'].sum()
        else:
            # Используем порог по label_quality
            issues_count = (issues_df['label_quality'] < 0.5).sum()
        
        # Статистика по классам
        class_stats = labels.value_counts().to_dict()
        
        # Проблемы по классам
        if 'is_label_issue' in issues_df.columns:
            issues_by_class = issues_df[issues_df['is_label_issue']]['given_label'].value_counts().to_dict()
        else:
            issues_by_class = {}
        
        self.statistics = {
            'total_records': total,
            'suspected_issues': int(issues_count),
            'issue_percentage': round(issues_count / total * 100, 2),
            'classes': class_stats,
            'issues_by_class': issues_by_class,
            'analysis_date': datetime.now().isoformat()
        }
    
    def _print_summary(self):
        """Вывод сводки по анализу"""
        
        logger.info("\n" + "="*50)
        logger.info("СВОДКА ПО КАЧЕСТВУ РАЗМЕТКИ")
        logger.info("="*50)
        logger.info(f"Всего записей: {self.statistics['total_records']:,}")
        logger.info(f"Подозрительных записей: {self.statistics['suspected_issues']:,} "
                   f"({self.statistics['issue_percentage']}%)")
        
        if self.statistics['issues_by_class']:
            logger.info("\nКлассы с наибольшим количеством проблем:")
            sorted_issues = sorted(
                self.statistics['issues_by_class'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for class_name, count in sorted_issues[:5]:
                total_in_class = self.statistics['classes'].get(class_name, 1)
                pct = count / total_in_class * 100
                logger.info(f"  - {class_name}: {count} проблем ({pct:.1f}% от класса)")
    
    def get_high_confidence_errors(self, 
                                   confidence_threshold: float = 0.9) -> pd.DataFrame:
        """
        Получение записей с высокой уверенностью в ошибке
        
        Это наиболее вероятные кандидаты на исправление
        """
        
        if self.label_issues is None:
            raise ValueError("Сначала запустите анализ (analyze_with_cleanlab)")
        
        if 'is_label_issue' in self.label_issues.columns:
            errors = self.label_issues[
                self.label_issues['is_label_issue'] & 
                (self.label_issues['confidence'] > confidence_threshold)
            ]
        else:
            # По label_quality
            errors = self.label_issues[
                self.label_issues['label_quality'] < (1 - confidence_threshold)
            ]
        
        return errors.sort_values('confidence', ascending=False)
    
    def get_uncertain_examples(self, 
                               confidence_threshold: float = 0.5) -> pd.DataFrame:
        """
        Получение примеров с низкой уверенностью модели
        
        Это сложные случаи, которые стоит проверить эксперту
        """
        
        if self.label_issues is None:
            raise ValueError("Сначала запустите анализ")
        
        if 'is_uncertain' in self.label_issues.columns:
            uncertain = self.label_issues[self.label_issues['is_uncertain']]
        else:
            uncertain = self.label_issues[
                self.label_issues['confidence'] < confidence_threshold
            ]
        
        return uncertain.sort_values('confidence')
    
    def create_expert_review_file(self,
                                  output_path: Path,
                                  n_high_confidence: int = 100,
                                  n_uncertain: int = 100) -> Tuple[pd.DataFrame, Path]:
        """
        Создание файла для передачи экспертам на проверку
        
        Включает:
        - Наиболее вероятные ошибки (высокая уверенность в ошибке)
        - Неуверенные примеры (сложные случаи)
        
        Returns:
            DataFrame для проверки и путь к сохраненному файлу
        """
        
        if self.label_issues is None:
            raise ValueError("Сначала запустите анализ")
        
        # Получаем высокоуверенные ошибки
        high_conf_errors = self.get_high_confidence_errors(0.8).head(n_high_confidence)
        high_conf_errors['review_reason'] = 'Высокая вероятность ошибки в разметке'
        high_conf_errors['priority'] = 'HIGH'
        
        # Получаем неуверенные примеры
        uncertain = self.get_uncertain_examples(0.5).head(n_uncertain)
        uncertain['review_reason'] = 'Низкая уверенность модели'
        uncertain['priority'] = 'MEDIUM'
        
        # Объединяем
        review_df = pd.concat([high_conf_errors, uncertain], ignore_index=True)
        
        # Убираем дубликаты по ID если есть
        if 'ID' in review_df.columns:
            review_df = review_df.drop_duplicates(subset=['ID'], keep='first')
        
        # Добавляем колонки для ответов эксперта
        review_df['expert_decision'] = ''  # CORRECT / INCORRECT / UNCLEAR
        review_df['corrected_label'] = ''
        review_df['expert_comment'] = ''
        review_df['reviewed_at'] = ''
        review_df['reviewed_by'] = ''
        
        # Переупорядочиваем колонки для удобства
        first_cols = ['ID', 'text', 'given_label', 'predicted_label', 
                      'confidence', 'priority', 'review_reason']
        other_cols = [c for c in review_df.columns if c not in first_cols]
        review_df = review_df[first_cols + other_cols]
        
        # Сохраняем
        save_excel_safe(review_df, output_path)
        
        logger.info(f"Файл для экспертной проверки создан: {output_path}")
        logger.info(f"  - Высокоприоритетных: {len(high_conf_errors)}")
        logger.info(f"  - Среднеприоритетных: {len(uncertain)}")
        logger.info(f"  - Всего к проверке: {len(review_df)}")
        
        return review_df, output_path
    
    def apply_expert_feedback(self,
                              feedback_df: pd.DataFrame,
                              original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Применение экспертной обратной связи к исходным данным
        
        Args:
            feedback_df: DataFrame с колонками 'ID', 'expert_decision', 'corrected_label'
            original_df: Исходный DataFrame с данными
        
        Returns:
            Исправленный DataFrame
        """
        
        corrected_df = original_df.copy()
        
        # Применяем исправления
        corrections_made = 0
        
        for _, row in feedback_df.iterrows():
            if row['expert_decision'] == 'INCORRECT' and pd.notna(row['corrected_label']):
                # Находим запись в оригинальном датафрейме
                mask = corrected_df['ID'] == row['ID']
                if mask.any():
                    old_label = corrected_df.loc[mask, 'Шаблон класса'].values[0]
                    corrected_df.loc[mask, 'Шаблон класса'] = row['corrected_label']
                    corrections_made += 1
                    logger.debug(f"ID {row['ID']}: {old_label} -> {row['corrected_label']}")
        
        logger.info(f"Применено исправлений: {corrections_made}")
        
        return corrected_df
    
    def save_report(self, output_path: Path):
        """Сохранение полного отчета по анализу"""
        
        if self.label_issues is None:
            raise ValueError("Сначала запустите анализ")
        
        # Создаем отчет
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Лист 1: Все записи с оценками
            self.label_issues.to_excel(writer, sheet_name='all_with_scores', index=False)
            
            # Лист 2: Высокоуверенные ошибки
            high_conf = self.get_high_confidence_errors(0.8)
            high_conf.to_excel(writer, sheet_name='high_confidence_errors', index=False)
            
            # Лист 3: Неуверенные примеры
            uncertain = self.get_uncertain_examples(0.5)
            uncertain.to_excel(writer, sheet_name='uncertain_examples', index=False)
            
            # Лист 4: Статистика
            stats_df = pd.DataFrame([self.statistics])
            stats_df.to_excel(writer, sheet_name='statistics', index=False)
            
            # Лист 5: Проблемы по классам
            if self.statistics['issues_by_class']:
                issues_df = pd.DataFrame(
                    list(self.statistics['issues_by_class'].items()),
                    columns=['class', 'issues_count']
                )
                issues_df['total_in_class'] = issues_df['class'].map(self.statistics['classes'])
                issues_df['issue_rate'] = issues_df['issues_count'] / issues_df['total_in_class']
                issues_df = issues_df.sort_values('issue_rate', ascending=False)
                issues_df.to_excel(writer, sheet_name='issues_by_class', index=False)
        
        logger.info(f"Отчет сохранен: {output_path}")


def analyze_label_quality(df: pd.DataFrame,
                          text_column: str = 'text_processed',
                          label_column: str = 'Шаблон класса',
                          id_column: str = 'ID',
                          output_dir: Path = Path('data/feedback')) -> Dict[str, Any]:
    """
    Полный цикл анализа качества разметки
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем детектор
    detector = LabelNoiseDetector()
    
    # Запускаем анализ
    try:
        # Пробуем с cleanlab
        issues_df = detector.analyze_with_cleanlab(
            texts=df[text_column],
            labels=df[label_column],
            ids=df[id_column],
            cv_folds=5
        )
        method_used = 'cleanlab'
    except Exception as e:
        logger.warning(f"Cleanlab не сработал: {e}")
        logger.info("Используем альтернативный метод через кросс-валидацию...")
        
        # Fallback на кросс-валидацию
        issues_df = detector.analyze_with_cross_validation(
            texts=df[text_column],
            labels=df[label_column],
            ids=df[id_column],
            cv_folds=5
        )
        method_used = 'cross_validation'
    
    # Сохраняем полный отчет
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f"label_quality_report_{timestamp}.xlsx"
    detector.save_report(report_path)
    
    # Создаем файл для экспертной проверки
    review_path = output_dir / f"expert_review_{timestamp}.xlsx"
    review_df, _ = detector.create_expert_review_file(
        review_path,
        n_high_confidence=200,
        n_uncertain=200
    )
    
    return {
        'method_used': method_used,
        'statistics': detector.statistics,
        'report_path': report_path,
        'review_path': review_path,
        'detector': detector
    }


if __name__ == "__main__":
    # Тестирование модуля
    from src.data_preprocessing import RussianTextPreprocessor
    from config import CLASSIFIED_FILE, PROCESSED_DATA_DIR
    
    # Загружаем и обрабатываем данные
    df = pd.read_excel(CLASSIFIED_FILE, nrows=5000)
    
    preprocessor = RussianTextPreprocessor(use_lemmatization=False)
    processed = preprocessor.process_dataframe(
        df,
        text_column='Наименование',
        class_column='Шаблон класса'
    )
    
    # Анализируем качество
    results = analyze_label_quality(
        processed,
        text_column='text_processed',
        label_column='Шаблон класса',
        id_column='ID',
        output_dir=Path('data/feedback')
    )
    
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("="*50)
    print(f"Метод: {results['method_used']}")
    print(f"Всего записей: {results['statistics']['total_records']}")
    print(f"Подозрительных: {results['statistics']['suspected_issues']} ({results['statistics']['issue_percentage']}%)")
    print(f"\nОтчет сохранен: {results['report_path']}")
    print(f"Файл для проверки: {results['review_path']}")