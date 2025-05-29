from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import os
import logging
import sklearn

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Выводим версии при запуске
logger.info(f"Python version: {'.'.join(map(str, os.sys.version_info[:3]))}")
logger.info(f"NumPy version: {np.__version__}")
logger.info(f"Scikit-learn version: {sklearn.__version__}")
logger.info(f"Joblib version: {joblib.__version__}")

def load_model(days_count):
    """Загрузка модели в зависимости от количества дней"""
    model_path = f'model/model_{days_count}days.joblib'
    logger.debug(f"Пытаемся загрузить модель: {model_path}")
    
    # Проверяем абсолютный путь к файлу
    abs_path = os.path.abspath(model_path)
    logger.debug(f"Абсолютный путь к модели: {abs_path}")
    
    # Проверяем существование файла
    if not os.path.exists(model_path):
        logger.error(f"Модель не найдена: {model_path}")
        # Проверяем содержимое директории
        model_dir = os.path.dirname(model_path)
        if os.path.exists(model_dir):
            logger.debug(f"Содержимое директории {model_dir}:")
            for file in os.listdir(model_dir):
                logger.debug(f"- {file}")
        else:
            logger.error(f"Директория {model_dir} не существует")
        return None
    
    try:
        # Проверяем размер файла
        file_size = os.path.getsize(model_path)
        logger.debug(f"Размер файла модели: {file_size} байт")
        
        # Выводим текущие версии библиотек
        logger.debug("Текущие версии библиотек:")
        logger.debug(f"Python: {'.'.join(map(str, os.sys.version_info[:3]))}")
        logger.debug(f"NumPy: {np.__version__}")
        logger.debug(f"Scikit-learn: {sklearn.__version__}")
        logger.debug(f"Joblib: {joblib.__version__}")
        
        # Загружаем модель
        model_data = joblib.load(model_path)
        logger.debug(f"Тип загруженных данных: {type(model_data)}")
        
        if isinstance(model_data, dict):
            logger.debug(f"Ключи в model_data: {model_data.keys()}")
            if 'metadata' in model_data:
                logger.debug("Версии при сохранении модели:")
                for k, v in model_data['metadata'].items():
                    logger.debug(f"- {k}: {v}")
        else:
            logger.warning("model_data не является словарем")
        
        return model_data
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {str(e)}")
        logger.exception("Полный стек ошибки:")
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
        
        try:
            model = model_data['model']
            scaler = model_data['scaler']
        except KeyError as e:
            logger.error(f"Отсутствует ключ в model_data: {e}")
            return jsonify({'error': 'Некорректная структура модели'}), 500
        
        # Подготовка данных для предсказания
        X = np.array(input_days).reshape(1, -1)
        try:
            X_scaled = scaler.transform(X)
            logger.debug(f"Подготовленные данные для предсказания: {X_scaled}")
        except Exception as e:
            logger.error(f"Ошибка при нормализации данных: {str(e)}")
            return jsonify({'error': 'Ошибка при подготовке данных'}), 500
        
        try:
            # Получение предсказаний
            predictions = model.predict(X_scaled)
            logger.debug(f"Получены предсказания: {predictions}")
            
            # Округляем все значения до целых чисел
            input_days = [round(x) for x in input_days]
            predictions = np.round(predictions[0]).astype(int).tolist()
            
            # Формирование полного временного ряда
            full_timeline = input_days + predictions
            logger.debug(f"Полный временной ряд: {full_timeline}")
            
            return jsonify({
                'predictions': full_timeline,
                'days_used': days_count
            })
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении предсказания: {str(e)}")
            logger.exception("Полный стек ошибки:")
            return jsonify({'error': f'Ошибка при выполнении предсказания: {str(e)}'}), 500
        
    except Exception as e:
        logger.error(f"Общая ошибка: {str(e)}")
        logger.exception("Полный стек ошибки:")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 