import gradio as gr
from prometheus_client import Counter, Histogram, start_http_server
import time
from inference import predict_with_threshold

# Start metrics endpoint
start_http_server(9000)
REQS = Counter("inference_requests_total", "Total inference requests")
LAT  = Histogram("inference_latency_seconds", "Latency per request (s)")

@LAT.time()
def serve(text, max_new_tokens=0):  # max_new_tokens kept for a consistent UI
    REQS.inc()
    t0 = time.time()
    out = predict_with_threshold(text)
    out["latency_ms"] = int((time.time() - t0) * 1000)
    return out

demo = gr.Interface(
    fn=serve,
    inputs=[gr.Textbox(lines=4, label="Input text"), gr.Slider(0, 0, 0, step=1, label="(unused)")],
    outputs="json",
    title="Fine-tuned Classifier (with Confidence Threshold)",
    description="Returns REJECTED if confidence < threshold."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
