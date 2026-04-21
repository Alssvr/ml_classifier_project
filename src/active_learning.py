# src/active_learning.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import json
import shutil

from src.utils import setup_logging, save_excel_safe
from src.train_model import TextClassifier
from src.data_preprocessing import RussianTextPreprocessor

logger = setup_logging()


class ActiveLearningManager:
    """
    Менеджер активного обучения с экспертной обратной связью
    
    Управляет циклами:
    1. Предсказание на новых данных
    2. Отбор примеров для экспертной проверки
    3. Применение исправлений
    4. Дообучение модели
    5. Оценка улучшения
    """
    
    def __init__(self,
                 models_dir: Path,
                 feedback_dir: Path,
                 experiment_name: str = "active_learning"):
        
        self.models_dir = Path(models_dir)
        self.feedback_dir = Path(feedback_dir)
        self.experiment_name = experiment_name
        
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_cycle = 0
        self.cycle_history: List[Dict] = []
        self.feedback_history: pd.DataFrame = None
        
        self._load_history()
        
        logger.info(f"Active Learning Manager инициализирован")
        logger.info(f"  Эксперимент: {experiment_name}")
        logger.info(f"  Текущий цикл: {self.current_cycle}")
    
    def _load_history(self):
        """Загрузка истории циклов"""
        
        history_file = self.feedback_dir / "cycle_history.json"
        feedback_file = self.feedback_dir / "feedback_history.csv"
        
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.cycle_history = data.get('cycles', [])
                self.current_cycle = data.get('current_cycle', 0)
                logger.info(f"Загружена история: {len(self.cycle_history)} циклов")
        
        if feedback_file.exists():
            self.feedback_history = pd.read_csv(feedback_file)
            logger.info(f"Загружена история обратной связи: {len(self.feedback_history)} записей")
        else:
            self.feedback_history = pd.DataFrame()
    
    def _save_history(self):
        """Сохранение истории циклов"""
        
        history_file = self.feedback_dir / "cycle_history.json"
        
        data = {
            'experiment_name': self.experiment_name,
            'current_cycle': self.current_cycle,
            'cycles': self.cycle_history,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        if not self.feedback_history.empty:
            feedback_file = self.feedback_dir / "feedback_history.csv"
            self.feedback_history.to_csv(feedback_file, index=False)
    
    def start_new_cycle(self,
                        classifier: TextClassifier,
                        unlabeled_data: pd.DataFrame,
                        preprocessor: RussianTextPreprocessor,
                        text_column: str = 'Наименование',
                        id_column: str = 'ID',
                        samples_for_review: int = 500) -> Tuple[pd.DataFrame, Path]:
        """
        Запуск нового цикла активного обучения
        
        1. Предсказание на неразмеченных данных
        2. Отбор наиболее информативных примеров для проверки
        3. Создание файла для экспертов
        
        Returns:
            DataFrame с отобранными примерами и путь к файлу
        """
        
        self.current_cycle += 1
        logger.info(f"Запуск цикла активного обучения #{self.current_cycle}")
        
        # Предобработка текстов
        logger.info("Предобработка текстов...")
        unlabeled_data = unlabeled_data.copy()
        unlabeled_data['text_processed'] = unlabeled_data[text_column].apply(
            preprocessor.process_text
        )
        
        # Удаляем пустые после обработки
        unlabeled_data = unlabeled_data[unlabeled_data['text_processed'] != ""]
        
        # Предсказание с вероятностями
        logger.info("Предсказание на новых данных...")
        predictions = classifier.predict_with_confidence(
            unlabeled_data['text_processed']
        )
        
        # Объединяем с исходными данными
        results_df = unlabeled_data.copy()
        results_df['predicted_class'] = predictions['predicted_class'].values
        results_df['confidence'] = predictions['confidence'].values
        results_df['is_confident'] = predictions['is_confident'].values
        results_df['second_guess'] = predictions['second_guess'].values
        
        # Отбираем примеры для экспертной проверки
        review_samples = self._select_samples_for_review(
            results_df,
            n_samples=samples_for_review
        )
        
        # Создаем файл для экспертов
        review_file = self._create_review_file(review_samples)
        
        # Сохраняем информацию о цикле
        cycle_info = {
            'cycle_number': self.current_cycle,
            'started_at': datetime.now().isoformat(),
            'model_version': classifier.version,
            'unlabeled_count': len(unlabeled_data),
            'selected_for_review': len(review_samples),
            'review_file': str(review_file),
            'status': 'awaiting_review',
            'metrics_before': classifier.metrics.copy() if classifier.metrics else {}
        }
        
        self.cycle_history.append(cycle_info)
        self._save_history()
        
        logger.info(f"Цикл #{self.current_cycle} запущен")
        logger.info(f"  Отобрано для проверки: {len(review_samples)} записей")
        logger.info(f"  Файл: {review_file}")
        
        return review_samples, review_file
    
    def _select_samples_for_review(self,
                                    predictions_df: pd.DataFrame,
                                    n_samples: int = 500) -> pd.DataFrame:
        """
        Стратегия отбора примеров для экспертной проверки
        
        Критерии:
        1. Низкая уверенность модели (uncertainty sampling)
        2. Близкие вероятности между топ-2 классами (margin sampling)
        3. Представители разных классов (diversity)
        """
        
        logger.info("Отбор примеров для экспертной проверки...")
        
        selected = []
        
        # 1. Uncertainty sampling - примеры с низкой уверенностью
        n_uncertain = int(n_samples * 0.4)
        uncertain = predictions_df[
            predictions_df['confidence'] < 0.6
        ].sort_values('confidence').head(n_uncertain)
        selected.append(uncertain)
        logger.info(f"  - Неуверенных: {len(uncertain)}")
        
        # 2. Margin sampling - близкие вероятности между классами
        n_margin = int(n_samples * 0.3)
        
        # Вычисляем разницу между вероятностями топ-2 классов
        def get_margin(row):
            if isinstance(row['class_probabilities'], dict):
                probs = sorted(row['class_probabilities'].values(), reverse=True)
                if len(probs) >= 2:
                    return probs[0] - probs[1]
            return 1.0
        
        predictions_df['margin'] = predictions_df.apply(get_margin, axis=1)
        margin_samples = predictions_df[
            (predictions_df['margin'] < 0.15) & 
            (predictions_df['confidence'] >= 0.6)  # Исключаем уже отобранные неуверенные
        ].sort_values('margin').head(n_margin)
        selected.append(margin_samples)
        logger.info(f"  - С близкой вероятностью классов: {len(margin_samples)}")
        
        # 3. Diversity sampling - по одному уверенному примеру из каждого класса
        n_diverse = n_samples - len(uncertain) - len(margin_samples)
        
        diverse_samples = []
        for class_name in predictions_df['predicted_class'].unique():
            class_df = predictions_df[
                (predictions_df['predicted_class'] == class_name) & 
                (predictions_df['confidence'] > 0.8)
            ]
            if not class_df.empty:
                # Берем самый уверенный пример
                sample = class_df.nlargest(1, 'confidence')
                diverse_samples.append(sample)
        
        if diverse_samples:
            diverse_df = pd.concat(diverse_samples, ignore_index=True)
            # Добираем случайными если не хватает
            if len(diverse_df) < n_diverse:
                remaining = n_diverse - len(diverse_df)
                random_samples = predictions_df[
                    ~predictions_df.index.isin(pd.concat(selected + [diverse_df]).index)
                ].sample(min(remaining, len(predictions_df)), random_state=42)
                diverse_df = pd.concat([diverse_df, random_samples])
            
            selected.append(diverse_df.head(n_diverse))
            logger.info(f"  - Разнообразных: {len(diverse_df.head(n_diverse))}")
        
        # Объединяем все отобранные
        review_df = pd.concat(selected, ignore_index=True)
        review_df = review_df.drop_duplicates(subset=['ID'])
        
        # Добавляем колонки для экспертов
        review_df['review_priority'] = review_df.apply(
            lambda r: 'HIGH' if r['confidence'] < 0.5 else ('MEDIUM' if r['margin'] < 0.1 else 'LOW'),
            axis=1
        )
        review_df['expert_decision'] = ''
        review_df['expert_label'] = ''
        review_df['expert_comment'] = ''
        review_df['reviewed_at'] = ''
        review_df['reviewed_by'] = ''
        
        # Сортируем по приоритету и уверенности
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        review_df['priority_order'] = review_df['review_priority'].map(priority_order)
        review_df = review_df.sort_values(
            ['priority_order', 'confidence']
        ).drop('priority_order', axis=1)
        
        return review_df
    
    def _create_review_file(self, review_df: pd.DataFrame) -> Path:
        """Создание Excel файла для экспертной проверки"""
        
        # Выбираем колонки для отображения
        display_columns = [
            'ID', 'Наименование', 'predicted_class', 'confidence',
            'second_guess', 'review_priority',
            'expert_decision', 'expert_label', 'expert_comment',
            'reviewed_at', 'reviewed_by'
        ]
        
        # Оставляем только существующие колонки
        available_columns = [c for c in display_columns if c in review_df.columns]
        export_df = review_df[available_columns].copy()
        
        # Создаем выпадающие списки для экспертного решения
        # (будет работать в Excel)
        export_df['expert_decision'] = ''
        
        filename = f"review_cycle_{self.current_cycle:03d}_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        filepath = self.feedback_dir / filename
        
        # Сохраняем с инструкцией на первом листе
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Лист с инструкцией
            instructions = pd.DataFrame({
                'Инструкция': [
                    '1. В колонке "expert_decision" укажите: CONFIRM (подтверждаю), CORRECT (исправить), SKIP (пропустить)',
                    '2. Если выбрали CORRECT, укажите правильный класс в колонке "expert_label"',
                    '3. При необходимости оставьте комментарий в "expert_comment"',
                    '4. После проверки сохраните файл с суффиксом "_reviewed"',
                    '',
                    f'Цикл обучения: {self.current_cycle}',
                    f'Дата создания: {datetime.now():%Y-%m-%d %H:%M:%S}',
                    f'Записей на проверку: {len(export_df)}'
                ]
            })
            instructions.to_excel(writer, sheet_name='ИНСТРУКЦИЯ', index=False)
            
            # Данные для проверки
            export_df.to_excel(writer, sheet_name='data', index=False)
            
            # Статистика по классам
            stats_df = export_df.groupby('predicted_class').agg({
                'ID': 'count',
                'confidence': 'mean'
            }).rename(columns={'ID': 'count', 'confidence': 'avg_confidence'})
            stats_df = stats_df.sort_values('count', ascending=False)
            stats_df.to_excel(writer, sheet_name='статистика')
        
        return filepath
    
    def process_expert_feedback(self,
                                reviewed_file: Path,
                                classifier: TextClassifier,
                                training_data: pd.DataFrame,
                                preprocessor: RussianTextPreprocessor,
                                text_column: str = 'Наименование',
                                label_column: str = 'Шаблон класса',
                                id_column: str = 'ID') -> Dict[str, Any]:
        """
        Обработка экспертной обратной связи и дообучение модели
        
        Returns:
            Словарь с результатами дообучения
        """
        
        logger.info(f"Обработка экспертной обратной связи: {reviewed_file}")
        
        # Загружаем проверенный файл
        reviewed_df = pd.read_excel(reviewed_file, sheet_name='data')
        
        # Фильтруем проверенные записи
        reviewed_mask = reviewed_df['expert_decision'].notna() & (reviewed_df['expert_decision'] != '')
        verified_df = reviewed_df[reviewed_mask].copy()
        
        logger.info(f"Проверено экспертом: {len(verified_df)} записей")
        
        # Анализируем решения
        decisions = verified_df['expert_decision'].value_counts()
        logger.info(f"Решения экспертов:")
        for decision, count in decisions.items():
            logger.info(f"  - {decision}: {count}")
        
        # Разделяем на подтвержденные и исправленные
        confirmed = verified_df[verified_df['expert_decision'].isin(['CONFIRM', 'OK', 'ДА', 'YES'])]
        corrected = verified_df[
            (verified_df['expert_decision'].isin(['CORRECT', 'ИСПРАВИТЬ', 'FIX'])) &
            (verified_df['expert_label'].notna())
        ]
        
        logger.info(f"Подтверждено: {len(confirmed)}, Исправлено: {len(corrected)}")
        
        # Создаем новые обучающие примеры из исправленных
        new_training_examples = []
        
        for _, row in corrected.iterrows():
            # Находим исходный текст
            original_row = reviewed_df[reviewed_df[id_column] == row[id_column]]
            if not original_row.empty:
                new_training_examples.append({
                    id_column: row[id_column],
                    text_column: original_row[text_column].values[0] if text_column in original_row.columns else '',
                    label_column: row['expert_label']
                })
        
        # Также добавляем подтвержденные как новые обучающие примеры
        # (с предсказанным классом)
        for _, row in confirmed.iterrows():
            original_row = reviewed_df[reviewed_df[id_column] == row[id_column]]
            if not original_row.empty:
                new_training_examples.append({
                    id_column: row[id_column],
                    text_column: original_row[text_column].values[0] if text_column in original_row.columns else '',
                    label_column: row['predicted_class']
                })
        
        # Создаем DataFrame с новыми примерами
        new_examples_df = pd.DataFrame(new_training_examples)
        
        if len(new_examples_df) == 0:
            logger.warning("Нет новых примеров для обучения")
            return {'new_examples': 0, 'model_updated': False}
        
        # Предобработка новых примеров
        new_examples_df['text_processed'] = new_examples_df[text_column].apply(
            preprocessor.process_text
        )
        
        # Объединяем с исходными обучающими данными
        training_data = training_data.copy()
        if 'text_processed' not in training_data.columns:
            training_data['text_processed'] = training_data[text_column].apply(
                preprocessor.process_text
            )
        
        # Добавляем новые примеры
        augmented_data = pd.concat([
            training_data[[id_column, text_column, 'text_processed', label_column]],
            new_examples_df[[id_column, text_column, 'text_processed', label_column]]
        ], ignore_index=True)
        
        # Удаляем дубликаты по ID (оставляем новые)
        augmented_data = augmented_data.drop_duplicates(subset=[id_column], keep='last')
        
        logger.info(f"Обучающая выборка увеличена: {len(training_data)} -> {len(augmented_data)} записей")
        
        # Сохраняем метрики до обучения
        metrics_before = classifier.metrics.copy() if classifier.metrics else {}
        
        # Дообучаем модель
        logger.info("Дообучение модели...")
        
        classifier.build_pipeline(
            tfidf_max_features=7000,
            tfidf_ngram_range=(1, 2),
            class_weight='balanced'
        )
        
        classifier.train(
            augmented_data['text_processed'],
            augmented_data[label_column],
            test_size=0.2
        )
        
        # Оцениваем улучшение
        metrics_after = classifier.metrics.copy()
        
        improvement = {
            'accuracy_before': metrics_before.get('test_accuracy', 0),
            'accuracy_after': metrics_after.get('test_accuracy', 0),
            'improvement': metrics_after.get('test_accuracy', 0) - metrics_before.get('test_accuracy', 0)
        }
        
        # Сохраняем новую версию модели
        new_version = f"{classifier.version.split('.')[0]}.{int(classifier.version.split('.')[-1]) + 1}.0"
        classifier.version = new_version
        classifier.save(self.models_dir)
        
        # Обновляем историю обратной связи
        feedback_records = []
        for _, row in verified_df.iterrows():
            feedback_records.append({
                'cycle': self.current_cycle,
                'id': row.get(id_column, ''),
                'original_text': row.get(text_column, ''),
                'predicted_class': row.get('predicted_class', ''),
                'confidence': row.get('confidence', 0),
                'expert_decision': row.get('expert_decision', ''),
                'expert_label': row.get('expert_label', ''),
                'reviewed_at': datetime.now().isoformat(),
                'used_for_training': row[id_column] in new_examples_df[id_column].values
            })
        
        new_feedback_df = pd.DataFrame(feedback_records)
        
        if self.feedback_history.empty:
            self.feedback_history = new_feedback_df
        else:
            self.feedback_history = pd.concat(
                [self.feedback_history, new_feedback_df],
                ignore_index=True
            )
        
        # Обновляем информацию о цикле
        if self.cycle_history:
            self.cycle_history[-1].update({
                'status': 'completed',
                'completed_at': datetime.now().isoformat(),
                'verified_count': len(verified_df),
                'confirmed_count': len(confirmed),
                'corrected_count': len(corrected),
                'new_examples_added': len(new_examples_df),
                'model_version_after': new_version,
                'metrics_after': metrics_after,
                'improvement': improvement
            })
        
        self._save_history()
        
        logger.info(f"Цикл #{self.current_cycle} завершен")
        logger.info(f"  Новая версия модели: {new_version}")
        logger.info(f"  Улучшение accuracy: {improvement['improvement']:+.4f}")
        
        return {
            'new_examples': len(new_examples_df),
            'model_updated': True,
            'new_version': new_version,
            'improvement': improvement,
            'augmented_data': augmented_data
        }
    
    def get_cycle_report(self) -> pd.DataFrame:
        """Получение отчета по всем циклам"""
        
        if not self.cycle_history:
            return pd.DataFrame()
        
        report_data = []
        for cycle in self.cycle_history:
            report_data.append({
                'cycle': cycle.get('cycle_number'),
                'started': cycle.get('started_at'),
                'status': cycle.get('status'),
                'samples_reviewed': cycle.get('verified_count', 0),
                'confirmed': cycle.get('confirmed_count', 0),
                'corrected': cycle.get('corrected_count', 0),
                'new_examples': cycle.get('new_examples_added', 0),
                'accuracy_before': cycle.get('metrics_before', {}).get('test_accuracy'),
                'accuracy_after': cycle.get('metrics_after', {}).get('test_accuracy'),
                'improvement': cycle.get('improvement', {}).get('improvement')
            })
        
        return pd.DataFrame(report_data)
    
    def export_full_classified_dataset(self,
                                       output_path: Path,
                                       include_metadata: bool = True) -> pd.DataFrame:
        """
        Экспорт полного классифицированного датасета
        включая все подтвержденные экспертами записи
        """
        
        # TODO: Собрать все данные из feedback_history
        
        logger.info("Экспорт полного датасета...")
        
        if self.feedback_history.empty:
            logger.warning("Нет данных для экспорта")
            return pd.DataFrame()
        
        # Фильтруем только проверенные и использованные для обучения
        export_df = self.feedback_history[
            self.feedback_history['used_for_training'] == True
        ].copy()
        
        if include_metadata:
            export_df['cycle'] = export_df['cycle']
            export_df['reviewed_at'] = export_df['reviewed_at']
        
        save_excel_safe(export_df, output_path)
        logger.info(f"Датасет экспортирован: {output_path} ({len(export_df)} записей)")
        
        return export_df


def run_active_learning_pipeline(
    classifier: TextClassifier,
    unlabeled_data: pd.DataFrame,
    training_data: pd.DataFrame,
    preprocessor: RussianTextPreprocessor,
    models_dir: Path,
    feedback_dir: Path,
    samples_per_cycle: int = 500
) -> ActiveLearningManager:
    """
    Запуск одного цикла активного обучения
    """
    
    # Создаем менеджер
    manager = ActiveLearningManager(models_dir, feedback_dir)
    
    # Запускаем новый цикл
    review_samples, review_file = manager.start_new_cycle(
        classifier=classifier,
        unlabeled_data=unlabeled_data,
        preprocessor=preprocessor,
        samples_for_review=samples_per_cycle
    )
    
    print("\n" + "="*60)
    print("ФАЙЛ ДЛЯ ЭКСПЕРТНОЙ ПРОВЕРКИ СОЗДАН")
    print("="*60)
    print(f"Путь к файлу: {review_file}")
    print(f"Записей на проверку: {len(review_samples)}")
    print("\nИнструкция:")
    print("1. Откройте файл в Excel")
    print("2. Заполните колонку 'expert_decision' (CONFIRM/CORRECT/SKIP)")
    print("3. При CORRECT укажите правильный класс в 'expert_label'")
    print("4. Сохраните файл с суффиксом '_reviewed'")
    print("5. Запустите process_expert_feedback() с этим файлом")
    print("="*60)
    
    return manager


if __name__ == "__main__":
    # Тестирование модуля
    from src.data_preprocessing import RussianTextPreprocessor
    from src.train_model import TextClassifier
    from config import CLASSIFIED_FILE, UNCLASSIFIED_FILE, MODELS_DIR
    
    # Загружаем данные
    df_train = pd.read_excel(CLASSIFIED_FILE, nrows=5000)
    df_unlabeled = pd.read_excel(UNCLASSIFIED_FILE, nrows=1000)
    
    # Препроцессор
    preprocessor = RussianTextPreprocessor(use_lemmatization=False)
    
    # Загружаем обученную модель (или создаем новую)
    try:
        classifier = TextClassifier.load(MODELS_DIR)
    except:
        logger.warning("Модель не найдена, создаем новую")
        classifier = TextClassifier(model_name="product_classifier", version="1.0.0")
        classifier.build_pipeline()
        
        processed = preprocessor.process_dataframe(
            df_train,
            text_column='Наименование',
            class_column='Шаблон класса'
        )
        
        classifier.train(
            processed['text_processed'],
            processed['Шаблон класса']
        )
        classifier.save(MODELS_DIR)
    
    # Запускаем активное обучение
    manager = run_active_learning_pipeline(
        classifier=classifier,
        unlabeled_data=df_unlabeled,
        training_data=df_train,
        preprocessor=preprocessor,
        models_dir=MODELS_DIR,
        feedback_dir=Path('data/feedback'),
        samples_per_cycle=100
    )
    
    print("\nИстория циклов:")
    print(manager.get_cycle_report())