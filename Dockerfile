FROM python:3.9-slim

WORKDIR /app

# Установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создаем отдельный слой для pip cache
RUN mkdir -p /root/.cache/pip && chmod -R 777 /root/.cache/pip

# Копирование файлов проекта
COPY . .

# Открываем порт
EXPOSE 5000

# Запуск приложения
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "backend.app:app", "--workers", "4"] 