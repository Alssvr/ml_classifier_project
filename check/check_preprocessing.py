# check_preprocessing.py
import pandas as pd
from src.data_preprocessing import RussianTextPreprocessor
import config

# Загрузите немного данных
df = pd.read_excel(config.UNCLASSIFIED_FILE, nrows=10)

preprocessor = RussianTextPreprocessor(
    use_lemmatization=config.PREPROCESSING_CONFIG['use_lemmatization']
)

print("="*50)
print("ПРОВЕРКА ПРЕДОБРАБОТКИ")
print("="*50)

for _, row in df.iterrows():
    original = str(row['Наименование'])[:60]
    processed = preprocessor.process_text(original)
    print(f"\nОригинал: {original}")
    print(f"Обработано: {processed}")
    print(f"Длина: {len(original.split())} -> {len(processed.split())} слов")