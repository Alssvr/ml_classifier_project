# check_probas.py
import pandas as pd
import joblib
from src.data_preprocessing import RussianTextPreprocessor
import config

# Загрузите модель
model_data = joblib.load('models/product_classifier_v1_1_0.pkl')
classifier = model_data['pipeline']
class_names = model_data['class_names']

preprocessor = RussianTextPreprocessor(use_lemmatization=True)

# Тестовые примеры из ОБУЧАЮЩЕЙ выборки (должны быть уверенными)
train_df = pd.read_excel(config.CLASSIFIED_FILE, nrows=5)

print("="*50)
print("ТЕСТ НА ОБУЧАЮЩИХ ДАННЫХ")
print("="*50)

for _, row in train_df.iterrows():
    text = str(row['Наименование'])
    processed = preprocessor.process_text(text)
    
    proba = classifier.predict_proba([processed])[0]
    max_prob = max(proba)
    pred_class = class_names[proba.argmax()]
    
    print(f"\nТекст: {text[:50]}...")
    print(f"Истинный класс: {row['Шаблон класса']}")
    print(f"Предсказано: {pred_class}")
    print(f"Уверенность: {max_prob:.4f}")