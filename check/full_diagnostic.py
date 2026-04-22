# full_diagnostic.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("ПОЛНАЯ ДИАГНОСТИКА МОДЕЛИ")
print("="*60)

# 1. Загружаем модель
model_path = Path('models/product_classifier_v1_1_0.pkl')
print(f"\n1. Загрузка модели: {model_path}")
print(f"   Файл существует: {model_path.exists()}")
print(f"   Размер файла: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

model_data = joblib.load(model_path)

print(f"\n2. Содержимое модели:")
for key in model_data.keys():
    value = model_data[key]
    if hasattr(value, '__len__'):
        print(f"   - {key}: {type(value).__name__} (длина: {len(value)})")
    else:
        print(f"   - {key}: {type(value).__name__}")

# 3. Проверяем пайплайн
pipeline = model_data['pipeline']
print(f"\n3. Пайплайн:")
print(f"   Шаги: {pipeline.named_steps.keys()}")
print(f"   Векторизатор: {type(pipeline.named_steps['vectorizer']).__name__}")
print(f"   Классификатор: {type(pipeline.named_steps['classifier']).__name__}")

# 4. Проверяем векторизатор
vectorizer = pipeline.named_steps['vectorizer']
print(f"\n4. Векторизатор:")
try:
    vocab_size = len(vectorizer.vocabulary_)
    print(f"   Размер словаря: {vocab_size}")
    print(f"   Примеры слов: {list(vectorizer.vocabulary_.keys())[:10]}")
except Exception as e:
    print(f"   ОШИБКА: {e}")

# 5. Тестовая векторизация
print(f"\n5. Тестовая векторизация:")
test_text = "распределитель переключения режимов"
print(f"   Текст: '{test_text}'")

try:
    vec_result = vectorizer.transform([test_text])
    print(f"   Форма матрицы: {vec_result.shape}")
    print(f"   Ненулевых элементов: {vec_result.nnz}")
    if vec_result.nnz > 0:
        # Показываем топ-5 признаков
        row = vec_result.toarray()[0]
        top_indices = row.argsort()[-5:][::-1]
        feature_names = vectorizer.get_feature_names_out()
        print(f"   Топ-5 признаков:")
        for idx in top_indices:
            if row[idx] > 0:
                print(f"     - {feature_names[idx]}: {row[idx]:.4f}")
    else:
        print("   ❌ Вектор пустой! Текст не содержит известных слов.")
except Exception as e:
    print(f"   ОШИБКА: {e}")

# 6. Тестовое предсказание через пайплайн
print(f"\n6. Тестовое предсказание:")
try:
    proba = pipeline.predict_proba([test_text])[0]
    class_names = model_data['class_names']
    
    top_indices = proba.argsort()[-5:][::-1]
    print(f"   Топ-5 предсказаний:")
    for i, idx in enumerate(top_indices, 1):
        print(f"     {i}. {class_names[idx]}: {proba[idx]:.4f}")
    
    max_prob = proba.max()
    pred_class = class_names[proba.argmax()]
    print(f"\n   Итоговое предсказание: {pred_class}")
    print(f"   Уверенность: {max_prob:.4f}")
    
except Exception as e:
    print(f"   ОШИБКА: {e}")
    import traceback
    traceback.print_exc()

# 7. Проверяем препроцессор
print(f"\n7. Проверка препроцессора:")
from src.data_preprocessing import RussianTextPreprocessor
import config

preprocessor = RussianTextPreprocessor(
    use_lemmatization=config.PREPROCESSING_CONFIG['use_lemmatization']
)

original = "Распределитель переключения режимов"
processed = preprocessor.process_text(original)

print(f"   Оригинал: '{original}'")
print(f"   Обработано: '{processed}'")

# 8. Предсказание через препроцессор + пайплайн
print(f"\n8. Предсказание с препроцессингом:")
try:
    proba2 = pipeline.predict_proba([processed])[0]
    top_indices2 = proba2.argsort()[-5:][::-1]
    print(f"   Топ-5 предсказаний:")
    for i, idx in enumerate(top_indices2, 1):
        print(f"     {i}. {class_names[idx]}: {proba2[idx]:.4f}")
except Exception as e:
    print(f"   ОШИБКА: {e}")

print("\n" + "="*60)