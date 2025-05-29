from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)


def load_model(days_count):
    """Загрузка модели в зависимости от количества дней"""
    model_path = f'model/model_{days_count}days.joblib'
    logger.debug(f"Пытаемся загрузить модель: {model_path}")
    if not os.path.exists(model_path):
        logger.error(f"Модель не найдена: {model_path}")
        return None

    try:
        model_data = joblib.load(model_path)
        logger.debug(f"Модель успешно загружена: {type(model_data)}")
        return model_data
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        return None


def validate_input_data(data):
    """Валидация входных данных"""
    logger.debug(f"Получены данные для валидации: {data}")
    if not data:
        return False, "Данные не предоставлены"

    days = []
    for i in range(1, 8):
        value = data.get(f'day{i}')
        if value is not None:
            if not isinstance(value, (int, float)) or value < 0:
                return False, "Значения должны быть неотрицательными числами"
            days.append(float(value))
        else:
            break

    logger.debug(f"Валидированные данные: {days}")
    if not days:
        return False, "Необходимо предоставить данные хотя бы за один день"

    return True, days


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        logger.debug(f"Получен POST запрос с данными: {request.json}")

        # Валидация входных данных
        is_valid, result = validate_input_data(request.json)
        if not is_valid:
            logger.error(f"Ошибка валидации: {result}")
            return jsonify({'error': result}), 400

        input_days = result
        days_count = len(input_days)
        logger.debug(f"Количество дней: {days_count}, входные данные: {input_days}")

        # Загрузка соответствующей модели
        model_data = load_model(days_count)
        if model_data is None:
            return jsonify({'error': f'Модель для {days_count} дней не найдена'}), 400

        model = model_data['model']
        scaler = model_data['scaler']

        # Подготовка данных для предсказания
        X = np.array(input_days).reshape(1, -1)
        X_scaled = scaler.transform(X)
        logger.debug(f"Подготовленные данные для предсказания: {X_scaled}")

        try:
            # Получение предсказаний
            predictions = model.predict(X_scaled)
            logger.debug(f"Получены предсказания: {predictions}")

            # Формирование полного временного ряда
            full_timeline = input_days + predictions[0].tolist()
            logger.debug(f"Полный временной ряд: {full_timeline}")

            return jsonify({
                'predictions': full_timeline,
                'days_used': days_count
            })

        except Exception as e:
            logger.error(f"Ошибка при выполнении предсказания: {str(e)}")
            return jsonify({'error': f'Ошибка при выполнении предсказания: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Общая ошибка: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 