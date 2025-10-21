import os, time
import gradio as gr # type: ignore
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from inference import predict_with_threshold

# ------------------- Prometheus metrics -------------------
REQS = Counter("inference_requests_total", "Total inference requests")
LATENCY = Histogram("inference_latency_seconds", "Inference latency (s)")

metrics_app = FastAPI()
@metrics_app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ------------------- Gradio Inference UI -------------------
def serve(text: str):
    t0 = time.time()
    out = predict_with_threshold(text)
    dur = time.time() - t0
    REQS.inc()
    LATENCY.observe(dur)

    label = out.get("label", "Unknown")
    conf = float(out.get("confidence", 0.0))
    pid  = int(out.get("label_id", -1))
    status = out.get("status", "OK")
    latency_ms = int(dur * 1000)
    if status == "REJECTED":
        return f"❌ Low confidence — REJECTED.\nGuess: {label} (id {pid})\nConfidence: {conf:.2%}\nLatency: {latency_ms} ms"
    return f"✅ Prediction: {label} (id {pid})\nConfidence: {conf:.2%}\nLatency: {latency_ms} ms"

# ------------------- External Dashboards -------------------
GRAFANA_IFRAME = os.getenv(
    "GRAFANA_IFRAME",
    "http://localhost:3000/d/YOUR_DASH_UID/nlp-inference-metrics?orgId=1&kiosk"
)
MLFLOW_IFRAME = os.getenv(
    "MLFLOW_IFRAME",
    "http://localhost:5000"
)

# ------------------- Full Gradio Interface -------------------
with gr.Blocks(title="Traceflow-monitoring") as demo:
    with gr.Tab("Predict"):
        gr.Markdown("### 🧠 Text Classifier — Enter any sentence:")
        inp = gr.Textbox(label="Input text", placeholder="Type here...")
        out = gr.Textbox(label="Model output")
        btn = gr.Button("Submit")
        btn.click(serve, inp, out)

    with gr.Tab("Monitoring"):
        gr.HTML(
            f'<iframe src="{GRAFANA_IFRAME}" '
            'width="100%" height="900" style="border:none;"></iframe>'
        )

    with gr.Tab("MLflow"):
        gr.HTML(
            f'<iframe src="{MLFLOW_IFRAME}" '
            'width="100%" height="900" style="border:none;"></iframe>'
        )

#demo.mount("/metrics", app=metrics_app)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
