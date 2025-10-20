import os, time, logging
import gradio as gr # type: ignore
from prometheus_client import Counter, Histogram, start_http_server
from logging.handlers import TimedRotatingFileHandler
from inference import predict_with_threshold

# ---------- App logging (rotates daily) ----------
os.makedirs("logs", exist_ok=True)
handler = TimedRotatingFileHandler("logs/app.log", when="D", interval=1, backupCount=7, encoding="utf-8")
logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("app")

# ---------- Metrics endpoint ----------
start_http_server(9000)
REQS = Counter("inference_requests_total", "Total inference requests")
LAT  = Histogram("inference_latency_seconds", "Latency per request (s)")

CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.5"))
GRAFANA_IFRAME = os.environ.get("GRAFANA_IFRAME", "")  # full iframe URL for your dashboard

@LAT.time()
def serve(text):
    REQS.inc()
    t0 = time.time()
    conf, pred_id, label = predict_with_threshold(text)
    latency_ms = int((time.time() - t0) * 1000)

    if conf < CONF_THRESH:
        msg = (f"**Prediction:** Rejected (low confidence)\n\n"
               f"**Most likely:** {label} (id {pred_id})\n"
               f"**Confidence:** {conf:.2%}\n"
               f"**Latency:** {latency_ms} ms")
        log.info(f"REJECTED | label={label} id={pred_id} conf={conf:.3f} latency_ms={latency_ms}")
        return msg

    msg = (f"**Prediction:** {label} (id {pred_id})\n"
           f"**Confidence:** {conf:.2%}\n"
           f"**Latency:** {latency_ms} ms")
    log.info(f"OK | label={label} id={pred_id} conf={conf:.3f} latency_ms={latency_ms}")
    return msg

def tail_logs():
    try:
        with open("logs/app.log", "r", encoding="utf-8") as f:
            return "".join(f.readlines()[-200:])
    except FileNotFoundError:
        return "(no logs yet)"
        
metrics_app = fastapi.FastAPI()
@metrics_app.get("/metrics")
def _metrics():
    return fastapi.responses.PlainTextResponse(pc.generate_latest().decode("utf-8"))
# later
demo = gr.Blocks(title="…")
demo.mount("/metrics", metrics_app)    

with gr.Blocks(title="Observa — Model + Monitoring") as demo:
    gr.Markdown("# 🧠 Observa — Model Inference & Monitoring")
    with gr.Tab("Model"):
        input_box = gr.Textbox(lines=4, label="Input text")
        output_md = gr.Markdown()
        gr.Button("Predict").click(serve, inputs=input_box, outputs=output_md)

    with gr.Tab("Monitoring"):
        gr.Markdown("### Live Metrics (Grafana)")
        if GRAFANA_IFRAME:
            gr.HTML(f'<iframe src="{GRAFANA_IFRAME}" width="100%" height="850" style="border:0;"></iframe>')
        else:
            gr.Markdown("> Set `GRAFANA_IFRAME` env to embed your Grafana dashboard here.")

    with gr.Tab("Logs"):
        logs_box = gr.Textbox(label="Recent logs", lines=14, interactive=False)
        gr.Button("Refresh").click(fn=tail_logs, outputs=logs_box)

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port)

