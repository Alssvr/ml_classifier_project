# test_model.py (обновленный)
import joblib

model_data = joblib.load('models/product_classifier_v1_1_0.pkl')

print("Ключи в модели:", list(model_data.keys()))
print("Тип векторизатора:", type(model_data['vectorizer']))
print("Количество классов:", len(model_data['class_names']))

# Проверяем, что векторизатор обучен
vectorizer = model_data['vectorizer']
print("Размер словаря:", len(vectorizer.get_feature_names_out()))