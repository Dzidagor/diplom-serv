FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем requirements.txt и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создаем отдельный слой для pip cache
RUN mkdir -p /root/.cache/pip && chmod -R 777 /root/.cache/pip

# Копируем все файлы приложения
COPY . .

# Создаем директорию для моделей, если её нет
RUN mkdir -p model

# Проверяем наличие моделей после копирования
RUN ls -la model/

# Открываем порт
EXPOSE 5000

# Запуск приложения
ENV FLASK_APP=backend/app.py
ENV FLASK_ENV=production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "backend.app:app"] 