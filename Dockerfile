FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements_inference.txt .
RUN pip install --no-cache-dir -r requirements_inference.txt

COPY inference_server.py .
COPY best.pt ./models/baseline_v1/weights/best.pt

ENV MODEL_PATH=./models/baseline_v1/weights/best.pt

EXPOSE 7860

CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "7860"]
