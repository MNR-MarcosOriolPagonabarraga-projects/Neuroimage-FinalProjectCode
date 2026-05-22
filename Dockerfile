FROM ml_base:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libxkbcommon0 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir  -r requirements.txt

COPY . .

