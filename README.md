
Observa — Model Inference & Monitoring

Observa is an end-to-end NLP + MLOps pipeline that demonstrates fine-tuned model inference, real-time monitoring, and experiment tracking — all integrated into one elegant web interface built with Gradio, FastAPI, Prometheus, Grafana, and MLflow.


Key Highlights


DVC Model versioning

🧩 Fine-tuned DistilBERT model for text classification

🖥️ Gradio web interface (with FastAPI backend)

📈 Prometheus metrics (/metrics endpoint)

📊 Grafana dashboards for visualization

🧪 MLflow experiment tracking (loss, accuracy, artifacts)

🐳 Dockerized for quick local or cloud deployment

🔐 Ready for OAuth integration and alerting (future scope)

Project Overview


          ┌──────────────────────────┐
          │       Gradio UI          │
          │  (Model Inference Tab)   │
          └──────────┬───────────────┘
                     │
                     ▼
             ┌──────────────────┐
             │   FastAPI App    │
             │ exposes /metrics │
             └──────────────────┘
                     │
     ┌───────────────┼────────────────────────┐
     ▼               ▼                        ▼
 Prometheus      MLflow Server             Grafana
 (collects)      (tracks runs)             (visualizes)

```
```
Folder Structure

```

nlp-mlops-pipeline/
│
├── .gitignore
├── docker-compose.yml
├── README.md
├── requirements.txt
│
├── app/
│   ├── app.py                     # Gradio + FastAPI app (Prometheus metrics + iframe embedding)
│   ├── inference.py               # Hugging Face model + predict_with_threshold()
│   ├── verify_model.py            # Utility to verify config.json and model files
│   ├── __init__.py
│
├── docker/
│   ├── Dockerfile.app             # App container (Gradio + FastAPI)
│   ├── Dockerfile.monitoring      # Prometheus + Grafana container
│   ├── grafana-data/              # Grafana local data
│   ├── prometheus-data/           # Prometheus local data
│   └── prometheus.render.yml      # Prometheus scrape configuration
│
├── monitoring/
│   ├── grafana/                   # Grafana dashboards and configs
│   │   ├── dashboards/
│   │   │   └── nlp-observa-dashboard.json
│   │   ├── datasources/
│   │   │   └── prometheus.yml
│   │   └── grafana.ini
│   └── prometheus.yml             # Prometheus configuration (targets: app:7860/metrics)
│
├── training/
│   ├── train.py                   # Model training with MLflow tracking
│   ├── evaluate.py                # Evaluation script
│   ├── params.yaml                # Configurable hyperparameters
│   └── data_prep.py               # Preprocessing logic
│
├── mlflow/
│   ├── mlflow.db                  # Local tracking DB (if SQLite backend)
│   └── artifacts/                 # Stored MLflow runs and metrics
│
├── reports/
│   ├── metrics.json               # Evaluation metrics summary


```
    
⚙️ Local Setup

1️⃣ Create Environment

```
python -m venv .venv
. .venv/Scripts/activate      # On Windows
# or source .venv/bin/activate (on Linux/Mac)
pip install -r app/requirements.txt

```

2️⃣ Run the Application

```
python app/app.py


```

3️⃣ (Optional) Run MLflow UI

```

mlflow ui --host 0.0.0.0 --port 5000


```

4️⃣ (Optional) Run Prometheus and Grafana (Docker)


```

cd docker
docker compose up --build


```








