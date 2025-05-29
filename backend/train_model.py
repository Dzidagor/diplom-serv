import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import joblib
import os

def prepare_training_data(X_days, y):
    """
    Подготовка данных для обучения модели.
    X_days: DataFrame с данными за первые 7 дней
    y: DataFrame с данными за дни 2-30
    """
    all_X = []
    all_y = []
    
    # Для каждого k от 1 до 7 создаем обучающие примеры
    for k in range(1, 8):
        X_k = X_days.iloc[:, :k]  # Берем первые k дней
        y_k = y.iloc[:, k:]       # Берем оставшиеся дни после k
        
        # Добавляем примеры в общий список
        all_X.extend(X_k.values)
        all_y.extend(y_k.values)
    
    return np.array(all_X), np.array(all_y)

def train_and_save_model(X_days, y, model_path='model/streaming_model.joblib'):
    """
    Обучение модели и сохранение её в файл
    """
    # Подготовка данных
    X, y = prepare_training_data(X_days, y)
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=30
    )
    
    # Создание и обучение модели с параметрами из оригинального скрипта
    model = ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        random_state=30,
        max_iter=10000
    )
    
    # Обучение модели
    model.fit(X_train, y_train)
    
    # Оценка качества модели
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Train R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")
    
    # Создание директории для модели, если её нет
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Сохранение модели
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # Здесь нужно загрузить ваши данные
    # Пример загрузки (замените на реальный путь к вашим данным):
    # X_days = pd.read_csv("path_to_your_X_days_data.csv")
    # y = pd.read_csv("path_to_your_y_data.csv")
    
    print("Please provide the paths to your data files to train the model") 