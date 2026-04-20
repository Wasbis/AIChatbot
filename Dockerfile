# Gunakan image Python yang ringan
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy file requirements dulu biar bisa di-cache sama Docker
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh source code
COPY . .

# Expose port yang dipakai FastAPI (misal: 8001)
EXPOSE 8001

# Command untuk jalanin server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]