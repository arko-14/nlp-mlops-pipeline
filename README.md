# Observa — Model Inference & Monitoring

Observa is an end-to-end NLP + MLOps pipeline that demonstrates fine-tuned model inference, real-time monitoring, and experiment tracking.


Key Highlights 


        DVC Model versioning

    🧩 Fine-tuned DistilBERT model for text     classification

    🖥️ Gradio web interface (with FastAPI backend)

    📈 Prometheus metrics (/metrics endpoint)

    📊 Grafana dashboards for visualization

    🧪 MLflow experiment tracking (loss, accuracy, artifacts)

    🐳 Dockerized for quick local or cloud deployment

🔐 Ready for OAuth integration and alerting (future scope)
```
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
Folder Structure

```

nlp-mlops-pipeline/
│
├── .gitignore
├── .dvc/                         # DVC internal metadata (auto-created)
├── data/                         # Raw & processed datasets (tracked via DVC)
│   ├── raw/
│   │   └── train.csv
│   │   └── test.csv
│   ├── processed/
│   │   └── clean_text.csv
│   └── README.md
│
├── docker-compose.yml
├── README.md
├── requirements.txt
├── dvc.yaml                      # DVC pipeline definition (stages: preprocess → train → eval)
├── params.yaml                   # Shared parameters (used by train.py + DVC)
│
├── app/
│   ├── app.py                    # Gradio + FastAPI + Prometheus UI
│   ├── inference.py              # Model inference logic
│   ├── verify_model.py           # Config validation helper
│   ├── __init__.py
│
├── docker/
│   ├── Dockerfile.app            # For app container
│   ├── Dockerfile.monitoring     # For Prometheus + Grafana
│   ├── grafana-data/
│   ├── prometheus-data/
│   └── prometheus.render.yml
│
├── monitoring/
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   └── observa-dashboard.json
│   │   ├── datasources/
│   │   │   └── prometheus.yml
│   │   └── grafana.ini
│   └── prometheus.yml
│
├── training/
│   ├── train.py                  # Logs metrics to MLflow
│   ├── evaluate.py
│   ├── data_prep.py              # Preprocessing stage (linked in dvc.yaml)
│   └── metrics.json
│
├── mlflow/
│   ├── mlflow.db
│   └── artifacts/
│
├── reports/
│   ├── metrics.json
│   
│
└──fix_config.py
    

### 🧠 Inference UI
![Gradio Interface](assets/ui_preview.png)


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



📈 Grafana Setup

1.Open Grafana at http://localhost:3000

2.Add a Prometheus data source → URL: http://prometheus:9090

3.Import or create panels with queries like:





⚙️ Typical DVC Commands


Initialize DVC inside your repo:

```
dvc init

```

Track your dataset:

```

dvc add data/raw/train.csv
git add data/.gitignore data/raw/train.csv.dvc
git commit -m "Track dataset with DVC"

```

Define and run the pipeline:

```

dvc repro


```

Visualize your pipeline:

```
dvc dag
```



🧪 MLflow Experiment Tracking


Training scripts automatically log metrics and artifacts to MLflow.
If you want to use your own tracking server:

```
export MLFLOW_TRACKING_URI=http://localhost:5000

```

🧮 Sample Output

Input:

```

UK markets rebound after inflation report

```
Output:

```
Prediction: Business (id 2)
Confidence: 95.38%
Latency: 1011 ms


```
🔧 Environment Variables

📦 Docker Deployment

```
docker compose up --build

```

Then visit:

App → localhost:7860

Prometheus → localhost:9090

Grafana → localhost:3000

MLflow → localhost:5000




```

🔭 Future Enhancements

Add OAuth-based login for dashboards

Integrate DVC for dataset versioning

Automate retraining and redeploy via CI/CD

Include Prometheus alerts for latency/error spikes

Add SHAP or LIME explainability inside Gradio UI

Deploy to cloud (Render / Oracle / AWS free tiers)

```


🧾 License

MIT License © 2025 — Sandipan

Free to use and modify for educational and personal projects.












<img width="960" height="540" alt="image" src="https://github.com/user-attachments/assets/95915fab-b985-4d0c-b9da-62c0ea1575c6" />



