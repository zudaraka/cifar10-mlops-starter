FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MLFLOW_TRACKING_URI=file:/app/mlflow_tracking
EXPOSE 8000
CMD ["uvicorn", "src.predict_service:app", "--host", "0.0.0.0", "--port", "8000"]
