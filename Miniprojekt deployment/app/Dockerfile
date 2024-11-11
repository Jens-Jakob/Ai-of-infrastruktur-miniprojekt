FROM python:3.11

WORKDIR /app

COPY requirements.txt .

COPY models/autoencoder.pth .

COPY models/kmeans.pkl .

COPY models/scaler.pkl .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
