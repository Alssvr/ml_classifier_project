# model_info.py
import joblib
from pathlib import Path
import json

# Загружаем модель
model_path = Path('models/product_classifier_v1_1_0.pkl')
if not model_path.exists():
    # Ищем любую модель
    models = list(Path('models').glob('product_classifier*.pkl'))
    if models:
        model_path = models[0]
    else:
        print("Модель не найдена!")
        exit()

model_data = joblib.load(model_path)
metrics = model_data['metrics']

print("="*60)
print("ИНФОРМАЦИЯ О МОДЕЛИ")
print("="*60)
print(f"Модель: {model_data['model_name']} v{model_data['version']}")
print(f"Дата обучения: {model_data['training_date']}")
print(f"Accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}")
print(f"Классов всего: {metrics.get('n_classes_total', 'N/A')}")
print(f"Классов в обучении: {metrics.get('n_classes_in_train', 'N/A')}")
print(f"Обучающих записей: {metrics.get('train_size', 'N/A'):,}")
print(f"Тестовых записей: {metrics.get('test_size', 'N/A'):,}")
print(f"Валидных тестовых: {metrics.get('valid_test_size', 'N/A'):,}")

rare = model_data.get('rare_classes_info', {})
if rare:
    print(f"Редких классов: {rare.get('count', 0)}")
print("="*60)

# Альтернативно - прочитать из metadata.json
meta_path = Path('models/model_metadata.json')
if meta_path.exists():
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    print("\nВсе версии моделей:")
    for m in meta.get('models', []):
        acc = m.get('metrics', {}).get('test_accuracy', 'N/A')
        print(f"  v{m['version']}: accuracy={acc:.4f}, записей={m['metrics'].get('train_size', 'N/A')}")