# main.py
"""
Главный скрипт для запуска полного пайплайна классификации

Режимы работы:
1. train - обучение модели на размеченных данных
2. analyze - анализ качества разметки (Cleanlab)
3. predict - классификация новых данных
4. active - запуск цикла активного обучения
5. full - полный пайплайн (train + analyze + predict)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent))

from src.utils import setup_logging, load_excel_with_progress, save_excel_safe
from src.data_preprocessing import RussianTextPreprocessor
from src.feature_engineering import TechnicalTfidfVectorizer
from src.train_model import TextClassifier, train_and_evaluate_model
from src.cleanlab_analysis import analyze_label_quality, LabelNoiseDetector
from src.active_learning import ActiveLearningManager, run_active_learning_pipeline
from src.predict import classify_new_data, Predictor

import config

logger = setup_logging()


def train_mode():
    """Режим обучения модели"""
    
    print("\n" + "="*60)
    print("РЕЖИМ ОБУЧЕНИЯ МОДЕЛИ")
    print("="*60)
    
    # Проверяем наличие файла
    if not config.CLASSIFIED_FILE.exists():
        logger.error(f"Файл не найден: {config.CLASSIFIED_FILE}")
        logger.info("Поместите размеченные данные в data/raw/classified_40k.xlsx")
        return None
    
    # Загружаем данные
    logger.info(f"Загрузка данных: {config.CLASSIFIED_FILE}")
    df_train = load_excel_with_progress(config.CLASSIFIED_FILE)
    
    logger.info(f"Загружено {len(df_train):,} записей")
    logger.info(f"Колонки: {df_train.columns.tolist()}")
    logger.info(f"Классы: {df_train[config.LABEL_COLUMN].nunique()}")
    
    # Препроцессинг
    logger.info("\nПредобработка текстов...")
    preprocessor = RussianTextPreprocessor(
        use_lemmatization=config.PREPROCESSING_CONFIG['use_lemmatization'],
        custom_stopwords=config.CUSTOM_STOPWORDS
    )
    
    processed = preprocessor.process_dataframe(
        df_train,
        text_column=config.TEXT_COLUMN,
        class_column=config.LABEL_COLUMN,
        id_column=config.ID_COLUMN
    )
    
    # Сохраняем обработанные данные
    processed_path = config.PROCESSED_DATA_DIR / "train_processed.xlsx"
    save_excel_safe(processed, processed_path)
    logger.info(f"Обработанные данные сохранены: {processed_path}")
    
    # Обучаем модель
    logger.info("\nОбучение модели...")
    classifier = train_and_evaluate_model(
        processed,
        text_column=config.PROCESSED_TEXT_COLUMN,
        class_column=config.LABEL_COLUMN,
        models_dir=config.MODELS_DIR,
        version=config.MODEL_VERSION
    )
    
    # Извлекаем правила
    from src.rules_classifier import RuleBasedClassifier
    
    rule_clf = RuleBasedClassifier()
    rule_clf.auto_extract_rules(
        processed,
        text_col='text_processed',
        class_col=config.LABEL_COLUMN,
        min_confidence=0.9,
        max_rules=300
    )
    rule_clf.save(config.MODELS_DIR / 'rule_classifier.pkl')
    
    stats = rule_clf.get_stats()
    print(f"Создано правил: {stats['rules_count']}")
    print(f"Покрытие обучающей выборки: {stats['coverage']*100:.1f}%")
    
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*60)
    print(f"Модель: {classifier.model_name} v{classifier.version}")
    print(f"Accuracy: {classifier.metrics.get('test_accuracy', 'N/A')}")
    print(f"Классов: {len(classifier.class_names)}")
    
    return classifier


def analyze_mode():
    """Режим анализа качества разметки"""
    
    print("\n" + "="*60)
    print("РЕЖИМ АНАЛИЗА КАЧЕСТВА РАЗМЕТКИ")
    print("="*60)
    
    # Проверяем наличие файла
    if not config.CLASSIFIED_FILE.exists():
        logger.error(f"Файл не найден: {config.CLASSIFIED_FILE}")
        return None
    
    # Загружаем данные
    df = load_excel_with_progress(config.CLASSIFIED_FILE)
    
    # Препроцессинг
    preprocessor = RussianTextPreprocessor(
        use_lemmatization=config.PREPROCESSING_CONFIG['use_lemmatization']
    )
    
    processed = preprocessor.process_dataframe(
        df,
        text_column=config.TEXT_COLUMN,
        class_column=config.LABEL_COLUMN,
        id_column=config.ID_COLUMN
    )
    
    # Анализируем качество
    results = analyze_label_quality(
        processed,
        text_column=config.PROCESSED_TEXT_COLUMN,
        label_column=config.LABEL_COLUMN,
        id_column=config.ID_COLUMN,
        output_dir=config.FEEDBACK_DIR
    )
    
    print("\n" + "="*60)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("="*60)
    print(f"Метод: {results['method_used']}")
    print(f"Всего записей: {results['statistics']['total_records']:,}")
    print(f"Подозрительных: {results['statistics']['suspected_issues']:,} "
          f"({results['statistics']['issue_percentage']}%)")
    print(f"\nОтчет: {results['report_path']}")
    print(f"Файл для проверки: {results['review_path']}")
    
    return results


def predict_mode():
    """Режим классификации новых данных"""
    
    print("\n" + "="*60)
    print("РЕЖИМ КЛАССИФИКАЦИИ НОВЫХ ДАННЫХ")
    print("="*60)
    
    # Проверяем наличие файлов
    if not config.UNCLASSIFIED_FILE.exists():
        logger.error(f"Файл не найден: {config.UNCLASSIFIED_FILE}")
        logger.info("Поместите неразмеченные данные в data/raw/unclassified_18k.xlsx")
        return None
    
    # Загружаем модель
    try:
        classifier = TextClassifier.load(config.MODELS_DIR)
        print(f"ML модель загружена: {classifier.version}")
    except Exception as e:
        logger.error(f"Не удалось загрузить модель: {e}")
        logger.info("Сначала запустите обучение: python main.py --mode train")
        return None
        
       
    # Препроцессор
    preprocessor = RussianTextPreprocessor(
        use_lemmatization=config.PREPROCESSING_CONFIG['use_lemmatization']
    )
    
    # Используем чистый Predictor
    from src.predict import Predictor
    predictor = Predictor(classifier, preprocessor)
    
    # Загружаем данные
    df = load_excel_with_progress(config.UNCLASSIFIED_FILE)
    
    result_df = predictor.predict_dataframe(
        df,
        text_column=config.TEXT_COLUMN,
        id_column=config.ID_COLUMN,
        confidence_threshold=config.PREDICTION_CONFIG['confidence_threshold']
    )
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = config.PROCESSED_DATA_DIR / f"classified_{timestamp}.xlsx"
    
    predictor.save_results(
        result_df,
        output_path,
        create_review_file=False
    )
    
    # Статистика
    confident = (~df['needs_review']).sum() if 'needs_review' in df.columns else 0
    needs_review = df['needs_review'].sum() if 'needs_review' in df.columns else len(df)
    
    print("\n" + "="*60)
    print("КЛАССИФИКАЦИЯ ЗАВЕРШЕНА")
    print("="*60)
    print(f"Всего обработано: {len(df):,} записей")
    print(f"Уверенных: {confident:,} ({confident/len(result_df)*100:.1f}%)")
    print(f"Требуют проверки: {needs_review:,}")
    print(f"\nРезультат: {output_path}")
    
    return {
        'total': len(result_df),
        'confident': confident,
        'needs_review': needs_review,
        'output': output_path
    }
    
def active_mode():
    """Режим активного обучения"""
    
    print("\n" + "="*60)
    print("РЕЖИМ АКТИВНОГО ОБУЧЕНИЯ")
    print("="*60)
    
    # Проверяем наличие файлов
    if not config.CLASSIFIED_FILE.exists():
        logger.error(f"Файл не найден: {config.CLASSIFIED_FILE}")
        return None
    
    if not config.UNCLASSIFIED_FILE.exists():
        logger.error(f"Файл не найден: {config.UNCLASSIFIED_FILE}")
        return None
    
    # Загружаем модель
    try:
        classifier = TextClassifier.load(config.MODELS_DIR)
    except:
        logger.warning("Модель не найдена, создаем новую...")
        train_mode()
        classifier = TextClassifier.load(config.MODELS_DIR)
    
    # Загружаем данные
    df_train = load_excel_with_progress(config.CLASSIFIED_FILE)
    df_unlabeled = load_excel_with_progress(config.UNCLASSIFIED_FILE)
    
    # Препроцессор
    preprocessor = RussianTextPreprocessor(
        use_lemmatization=config.PREPROCESSING_CONFIG['use_lemmatization']
    )
    
    # Запускаем активное обучение
    manager = run_active_learning_pipeline(
        classifier=classifier,
        unlabeled_data=df_unlabeled,
        training_data=df_train,
        preprocessor=preprocessor,
        models_dir=config.MODELS_DIR,
        feedback_dir=config.FEEDBACK_DIR,
        samples_per_cycle=config.ACTIVE_LEARNING_CONFIG['samples_per_cycle']
    )
    
    # Показываем историю
    history = manager.get_cycle_report()
    if not history.empty:
        print("\nИстория циклов:")
        print(history.to_string(index=False))
    
    return manager


def full_pipeline():
    """Полный пайплайн"""
    
    print("\n" + "="*60)
    print("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА")
    print("="*60)
    
    start_time = datetime.now()
    
    # 1. Обучение
    print("\n[1/4] Обучение модели...")
    classifier = train_mode()
    
    if classifier is None:
        logger.error("Обучение не выполнено")
        return None
    
    # 2. Анализ качества
    print("\n[2/4] Анализ качества разметки...")
    analysis_results = analyze_mode()
    
    # 3. Классификация
    print("\n[3/4] Классификация новых данных...")
    predict_results = predict_mode()
    
    # 4. Создание файла для проверки
    print("\n[4/4] Создание файла для экспертной проверки...")
    
    if predict_results and predict_results['review_file']:
        print(f"Файл для проверки создан: {predict_results['review_file']}")
    
    elapsed = datetime.now() - start_time
    
    print("\n" + "="*60)
    print("ПАЙПЛАЙН ЗАВЕРШЕН")
    print("="*60)
    print(f"Время выполнения: {elapsed}")
    print(f"Модель: {config.MODEL_NAME} v{config.MODEL_VERSION}")
    print(f"Результаты в: {config.PROCESSED_DATA_DIR}")
    
    return {
        'classifier': classifier,
        'analysis': analysis_results,
        'predictions': predict_results,
        'elapsed_time': elapsed
    }


def main():
    """Главная функция"""
    
    parser = argparse.ArgumentParser(
        description='Классификация текстов на основе ML'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'analyze', 'predict', 'active', 'full'],
        default='full',
        help='Режим работы'
    )
    
    parser.add_argument(
        '--config',
        action='store_true',
        help='Показать конфигурацию'
    )
    
    args = parser.parse_args()
    
    # Показываем конфигурацию
    if args.config:
        config.print_config()
        return
    
    print("\n" + "="*60)
    print("КЛАССИФИКАТОР ТЕКСТОВ")
    print("="*60)
    print(f"Время запуска: {datetime.now():%Y-%m-%d %H:%M:%S}")
    
    # Запускаем выбранный режим
    if args.mode == 'train':
        train_mode()
    elif args.mode == 'analyze':
        analyze_mode()
    elif args.mode == 'predict':
        predict_mode()
    elif args.mode == 'active':
        active_mode()
    elif args.mode == 'full':
        full_pipeline()
    
    print("\nГотово!")


if __name__ == "__main__":
    main()