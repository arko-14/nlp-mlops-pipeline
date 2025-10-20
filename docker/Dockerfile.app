FROM python:3.12-slim
WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
# models will be mounted at /app/models via docker-compose
WORKDIR /app/app
EXPOSE 7860 9000
CMD ["python", "app.py"]
