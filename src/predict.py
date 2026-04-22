# src/predict.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from src.utils import setup_logging, save_excel_safe, load_excel_with_progress
from src.data_preprocessing import RussianTextPreprocessor
from src.train_model import TextClassifier

logger = setup_logging()


class Predictor:
    """Класс для классификации новых данных"""
    
    def __init__(self,
                 classifier: TextClassifier,
                 preprocessor: RussianTextPreprocessor):
        
        self.classifier = classifier
        self.preprocessor = preprocessor
        
        logger.info(f"Predictor инициализирован с моделью {classifier.version}")
    
    def predict_dataframe(self,
                          df: pd.DataFrame,
                          text_column: str = 'Наименование',
                          id_column: str = 'ID',
                          confidence_threshold: float = 0.1,
                          batch_size: int = 5000) -> pd.DataFrame:
        """
        Классификация всего DataFrame
        
        Args:
            df: DataFrame с данными для классификации
            text_column: название колонки с текстом
            id_column: название колонки с ID
            confidence_threshold: порог уверенности
            batch_size: размер батча для обработки больших файлов
        
        Returns:
            DataFrame с добавленными колонками классификации
        """
        
        total_rows = len(df)
        logger.info(f"Начало классификации {total_rows:,} записей...")
        
        # Копируем DataFrame
        result_df = df.copy()
        
        # Предобработка текстов
        logger.info("Предобработка текстов...")
        result_df['text_processed'] = result_df[text_column].astype(str).apply(
            self.preprocessor.process_text
        )
        
        # Проверяем пустые тексты
        empty_mask = result_df['text_processed'] == ''
        empty_count = empty_mask.sum()
        if empty_count > 0:
            logger.warning(f"Найдено {empty_count} пустых текстов после обработки")
        
        # Классификация по батчам (для больших файлов)
        all_predictions = []
        all_confidences = []
        all_second_guesses = []
        all_needs_review = []
        all_probabilities = []
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_texts = result_df['text_processed'].iloc[start_idx:end_idx]
            
            logger.info(f"Обработка батча {start_idx//batch_size + 1}: "
                       f"записи {start_idx:,}-{end_idx:,}")
            
            # Предсказание
            predictions_batch = self.classifier.predict_with_confidence(batch_texts)
            
            all_predictions.extend(predictions_batch['predicted_class'].tolist())
            all_confidences.extend(predictions_batch['confidence'].tolist())
            all_second_guesses.extend(predictions_batch['second_guess'].tolist())
            all_probabilities.extend(predictions_batch['class_probabilities'].tolist())
            
            # Определяем необходимость экспертной проверки
            # Определяем необходимость экспертной проверки
            # Используем более реалистичные пороги для 945 классов
            needs_review = predictions_batch['confidence'] < 0.1  # Только <10% считаем неуверенными
            
            all_needs_review.extend(needs_review.tolist())
        
        # Добавляем результаты
        result_df['predicted_class'] = all_predictions
        result_df['confidence'] = all_confidences
        result_df['second_guess'] = all_second_guesses
        result_df['needs_review'] = all_needs_review
        result_df['class_probabilities'] = all_probabilities
        
        # Для пустых текстов ставим специальные значения
        if empty_count > 0:
            result_df.loc[empty_mask, 'predicted_class'] = 'EMPTY_TEXT'
            result_df.loc[empty_mask, 'confidence'] = 0.0
            result_df.loc[empty_mask, 'needs_review'] = True
        
        # Добавляем статус
        result_df['classification_status'] = result_df['needs_review'].apply(
            lambda x: 'NEEDS_REVIEW' if x else 'CONFIDENT'
        )
        
        # Статистика
        self._print_statistics(result_df)
        
        return result_df
    
    def _print_statistics(self, df: pd.DataFrame):
        """Вывод статистики классификации"""
        
        total = len(df)
        confident = (df['classification_status'] == 'CONFIDENT').sum()
        needs_review = (df['classification_status'] == 'NEEDS_REVIEW').sum()
        empty = (df['predicted_class'] == 'EMPTY_TEXT').sum()
        
        logger.info("\n" + "="*50)
        logger.info("СТАТИСТИКА КЛАССИФИКАЦИИ")
        logger.info("="*50)
        logger.info(f"Всего записей: {total:,}")
        logger.info(f"Уверенных предсказаний: {confident:,} ({confident/total*100:.1f}%)")
        logger.info(f"Требуют проверки: {needs_review:,} ({needs_review/total*100:.1f}%)")
        logger.info(f"Пустых текстов: {empty:,}")
        
        # Распределение по классам
        logger.info("\nРаспределение по классам (топ-10):")
        class_dist = df['predicted_class'].value_counts().head(10)
        for class_name, count in class_dist.items():
            if class_name != 'EMPTY_TEXT':
                pct = count / total * 100
                logger.info(f"  {class_name}: {count:,} ({pct:.1f}%)")
        
        # Средняя уверенность
        avg_conf = df[df['predicted_class'] != 'EMPTY_TEXT']['confidence'].mean()
        logger.info(f"\nСредняя уверенность: {avg_conf:.3f}")
    
    def save_results(self,
                     df: pd.DataFrame,
                     output_path: Path,
                     include_probabilities: bool = False,
                     create_review_file: bool = True) -> Tuple[Path, Optional[Path]]:
        """
        Сохранение результатов классификации
        
        Returns:
            Путь к основному файлу и путь к файлу для проверки (если создан)
        """
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Подготовка колонок для сохранения
        export_columns = [
            'ID', 'Наименование', 'predicted_class', 'confidence',
            'second_guess', 'classification_status'
        ]
        
        # Добавляем существующие колонки
        available_columns = [c for c in export_columns if c in df.columns]
        export_df = df[available_columns].copy()
        
        # Округляем confidence для читаемости
        export_df['confidence'] = export_df['confidence'].round(4)
        
        # Сохраняем основной файл
        save_excel_safe(export_df, output_path)
        logger.info(f"Результаты сохранены: {output_path}")
        
        review_path = None
        
        # Создаем отдельный файл для проверки
        if create_review_file:
            review_df = df[df['needs_review'] == True].copy()
            
            if len(review_df) > 0:
                review_columns = [
                    'ID', 'Наименование', 'predicted_class', 'confidence',
                    'second_guess', 'class_probabilities'
                ]
                review_available = [c for c in review_columns if c in review_df.columns]
                review_export = review_df[review_available].copy()
                review_export['confidence'] = review_export['confidence'].round(4)
                
                review_path = output_path.parent / f"{output_path.stem}_needs_review.xlsx"
                save_excel_safe(review_export, review_path)
                logger.info(f"Записи для проверки сохранены: {review_path} ({len(review_export)} шт.)")
        
        return output_path, review_path
    
    def export_for_expert_review(self,
                                 df: pd.DataFrame,
                                 output_path: Path,
                                 n_samples: int = 500,
                                 priority: str = 'low_confidence') -> Path:
        """
        Экспорт записей для экспертной проверки с приоритизацией
        
        Args:
            df: DataFrame с предсказаниями
            output_path: путь для сохранения
            n_samples: количество записей
            priority: стратегия отбора ('low_confidence', 'diverse', 'mixed')
        """
        
        review_df = df[df['needs_review'] == True].copy()
        
        if priority == 'low_confidence':
            # Самые неуверенные
            selected = review_df.nsmallest(n_samples, 'confidence')
        elif priority == 'diverse':
            # По одному из каждого класса с низкой уверенностью
            selected_list = []
            for class_name in review_df['predicted_class'].unique():
                class_samples = review_df[review_df['predicted_class'] == class_name]
                if not class_samples.empty:
                    selected_list.append(class_samples.nsmallest(1, 'confidence'))
            selected = pd.concat(selected_list, ignore_index=True)
            
            # Добираем до n_samples самыми неуверенными
            if len(selected) < n_samples:
                remaining = review_df[~review_df.index.isin(selected.index)]
                additional = remaining.nsmallest(n_samples - len(selected), 'confidence')
                selected = pd.concat([selected, additional], ignore_index=True)
        else:
            # Смешанная стратегия
            n_low = int(n_samples * 0.7)
            n_diverse = n_samples - n_low
            
            low_conf = review_df.nsmallest(n_low, 'confidence')
            
            diverse_list = []
            for class_name in review_df['predicted_class'].unique():
                if class_name not in low_conf['predicted_class'].values:
                    class_samples = review_df[review_df['predicted_class'] == class_name]
                    if not class_samples.empty:
                        diverse_list.append(class_samples.nsmallest(1, 'confidence'))
            
            if diverse_list:
                diverse_df = pd.concat(diverse_list, ignore_index=True)
                selected = pd.concat([low_conf, diverse_df.head(n_diverse)], ignore_index=True)
            else:
                selected = low_conf
        
        # Добавляем колонки для эксперта
        selected['expert_decision'] = ''
        selected['expert_label'] = ''
        selected['expert_comment'] = ''
        selected['review_priority'] = 'HIGH'
        
        # Форматируем для удобства
        display_columns = [
            'ID', 'Наименование', 'predicted_class', 'confidence',
            'second_guess', 'review_priority',
            'expert_decision', 'expert_label', 'expert_comment'
        ]
        available = [c for c in display_columns if c in selected.columns]
        
        save_excel_safe(selected[available], output_path)
        logger.info(f"Файл для экспертной проверки создан: {output_path}")
        logger.info(f"  Записей: {len(selected)}")
        
        return output_path


def classify_new_data(input_path: Path,
                      output_dir: Path,
                      classifier: TextClassifier,
                      preprocessor: RussianTextPreprocessor,
                      text_column: str = 'Наименование',
                      id_column: str = 'ID',
                      confidence_threshold: float = 0.7) -> Dict[str, Path]:
    """
    Полный цикл классификации новых данных
    
    Args:
        input_path: путь к Excel файлу с данными
        output_dir: директория для результатов
        classifier: обученный классификатор
        preprocessor: препроцессор текста
        text_column: колонка с текстом
        id_column: колонка с ID
        confidence_threshold: порог уверенности
    
    Returns:
        Словарь с путями к созданным файлам
    """
    
    # Создаем предиктор
    predictor = Predictor(classifier, preprocessor)
    
    # Загружаем данные
    logger.info(f"Загрузка данных: {input_path}")
    df = load_excel_with_progress(input_path)
    
    # Проверяем наличие колонок
    if text_column not in df.columns:
        raise ValueError(f"Колонка '{text_column}' не найдена. Доступные: {df.columns.tolist()}")
    
    # Классифицируем
    result_df = predictor.predict_dataframe(
        df,
        text_column=text_column,
        id_column=id_column,
        confidence_threshold=confidence_threshold
    )
    
    # Сохраняем результаты
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"classified_{timestamp}.xlsx"
    review_path = output_dir / f"needs_review_{timestamp}.xlsx"
    
    main_path, _ = predictor.save_results(
        result_df,
        output_path,
        create_review_file=False  # Создадим отдельно
    )
    
    # Создаем файл для проверки
    if len(result_df[result_df['needs_review']]) > 0:
        predictor.export_for_expert_review(
            result_df,
            review_path,
            n_samples=min(500, len(result_df[result_df['needs_review']]))
        )
    
    return {
        'main_result': main_path,
        'review_file': review_path if review_path.exists() else None,
        'total_records': len(result_df),
        'confident_records': (result_df['classification_status'] == 'CONFIDENT').sum(),
        'needs_review_records': (result_df['classification_status'] == 'NEEDS_REVIEW').sum()
    }


if __name__ == "__main__":
    # Тестирование
    from src.train_model import TextClassifier
    from config import MODELS_DIR, UNCLASSIFIED_FILE, PROCESSED_DATA_DIR
    
    # Загружаем модель
    classifier = TextClassifier.load(MODELS_DIR)
    
    # Препроцессор
    preprocessor = RussianTextPreprocessor(use_lemmatization=False)
    
    # Классифицируем
    results = classify_new_data(
        input_path=UNCLASSIFIED_FILE,
        output_dir=PROCESSED_DATA_DIR,
        classifier=classifier,
        preprocessor=preprocessor
    )
    
    print("\nРезультаты классификации:")
    for key, value in results.items():
        print(f"  {key}: {value}")