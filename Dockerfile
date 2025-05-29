FROM python:3.9-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    python3-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

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