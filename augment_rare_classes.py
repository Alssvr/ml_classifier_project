# augment_rare_classes.py
import pandas as pd
import random
from pathlib import Path
import config

def augment_text(text: str, n_variants: int = 5) -> list:
    """Генерирует варианты текста"""
    variants = []
    text_lower = text.lower()
    
    # 1. Замена ё ↔ е
    if 'ё' in text_lower:
        variants.append(text.replace('ё', 'е').replace('Ё', 'Е'))
    if 'е' in text_lower and 'ё' not in text_lower:
        variants.append(text.replace('е', 'ё').replace('Е', 'Ё'))
    
    # 2. Перестановка слов
    words = text.split()
    if len(words) >= 3:
        random.shuffle(words)
        variants.append(' '.join(words))
    
    # 3. Изменение чисел (если есть цифры)
    import re
    numbers = re.findall(r'\d+[.,]?\d*', text)
    if numbers:
        for num in numbers:
            try:
                val = float(num.replace(',', '.'))
                new_val = val * random.choice([0.9, 1.1, 1.2, 0.8])
                new_num = f"{new_val:.1f}".replace('.', ',') if ',' in num else f"{new_val:.1f}"
                variants.append(text.replace(num, new_num))
            except:
                pass
    
    # 4. Удаление/добавление пробелов
    variants.append(text.replace(' ', ''))
    
    # 5. Изменение регистра
    variants.append(text.upper())
    variants.append(text.lower())
    variants.append(text.title())
    
    # Оставляем уникальные и не совпадающие с оригиналом
    unique_variants = list(set([v for v in variants if v != text]))
    
    return unique_variants[:n_variants]


def augment_rare_classes(df: pd.DataFrame, 
                         min_samples: int = 3,
                         target_samples: int = 7) -> pd.DataFrame:
    """Добавляет синтетические примеры для редких классов"""
    
    class_counts = df[config.LABEL_COLUMN].value_counts()
    rare_classes = class_counts[class_counts < min_samples].index
    
    print(f"Найдено {len(rare_classes)} классов с < {min_samples} примерами")
    
    new_rows = []
    
    for class_name in rare_classes:
        class_df = df[df[config.LABEL_COLUMN] == class_name]
        current_count = len(class_df)
        needed = target_samples - current_count
        
        if needed <= 0:
            continue
        
        print(f"  {class_name}: {current_count} → {target_samples} (+{needed})")
        
        for _, row in class_df.iterrows():
            original_text = row[config.TEXT_COLUMN]
            variants = augment_text(original_text, n_variants=needed)
            
            for variant in variants:
                new_id = f"{row[config.ID_COLUMN]}_aug_{len(new_rows)}"
                new_rows.append({
                    config.ID_COLUMN: new_id,
                    config.TEXT_COLUMN: variant,
                    config.LABEL_COLUMN: class_name,
                    'is_augmented': True
                })
                
                if len(new_rows) >= needed * len(class_df):
                    break
            if len(new_rows) >= needed * len(class_df):
                break
    
    if new_rows:
        augmented_df = pd.DataFrame(new_rows)
        result_df = pd.concat([df, augmented_df], ignore_index=True)
        print(f"\nДобавлено {len(new_rows)} синтетических примеров")
        print(f"Итоговый размер: {len(df)} → {len(result_df)} записей")
        return result_df
    
    return df


if __name__ == "__main__":
    df = pd.read_excel(config.CLASSIFIED_FILE)
    print(f"Исходный размер: {len(df)} записей")
    
    augmented_df = augment_rare_classes(df, min_samples=3, target_samples=7)
    
    output_path = config.RAW_DATA_DIR / "classified_augmented.xlsx"
    augmented_df.to_excel(output_path, index=False)
    print(f"\nСохранено: {output_path}")