from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

def load_model(days_count):
    """Загрузка модели в зависимости от количества дней"""
    model_path = f'model/model_{days_count}days.joblib'
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

def validate_input_data(data):
    """Валидация входных данных"""
    if not data:
        return False, "Данные не предоставлены"
    
    days = []
    for i in range(1, 8):
        value = data.get(f'day{i}')
        if value is not None:
            if not isinstance(value, (int, float)) or value < 0:
                return False, "Значения должны быть неотрицательными числами"
            days.append(value)
        else:
            break
    
    if not days:
        return False, "Необходимо предоставить данные хотя бы за один день"
    
    return True, days

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Валидация входных данных
        is_valid, result = validate_input_data(request.json)
        if not is_valid:
            return jsonify({'error': result}), 400
        
        input_days = result
        days_count = len(input_days)
        
        # Загрузка соответствующей модели
        model = load_model(days_count)
        if model is None:
            return jsonify({'error': f'Модель для {days_count} дней не найдена'}), 400
        
        # Подготовка данных для предсказания
        X = np.array(input_days).reshape(1, -1)
        
        # Получение предсказаний
        predictions = model.predict(X)[0]
        
        # Формирование полного временного ряда
        full_timeline = input_days + predictions.tolist()
        
        return jsonify({
            'predictions': full_timeline,
            'days_used': days_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 