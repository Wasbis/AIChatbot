# Gunakan image Python yang ringan
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Matikan output buffering Python biar semua print() langsung muncul di docker logs
ENV PYTHONUNBUFFERED=1

# Copy file requirements dulu biar bisa di-cache sama Docker
COPY requirements.txt .

# Install dependencies menggunakan BuildKit Cache Mount
# Ini bakal jauh lebih cepet pas kamu nambah library baru
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

RUN python -m playwright install chromium --with-deps

# Copy seluruh source code
COPY . .

# Expose port yang dipakai FastAPI
EXPOSE 8001

# Tambahkan CMD (biar container tahu apa yang harus dijalankan saat start)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]