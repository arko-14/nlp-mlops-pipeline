import os, json, time, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download

MODEL_ID = os.getenv("MODEL_ID", "sandipan14/fine-grained")
LOCAL_DIR = os.getenv("MODEL_DIR", "models/fine_tuned")  
HF_TOKEN  = os.getenv("HF_TOKEN")

def find_model_dir(root: str) -> str:
    for dirpath, _, files in os.walk(root):
        f = set(files)
        if "config.json" in f and ("model.safetensors" in f or "pytorch_model.bin" in f):
            return dirpath
    return root

def get_model_path():
    if os.path.exists(LOCAL_DIR):
        return find_model_dir(LOCAL_DIR)
    snap = snapshot_download(MODEL_ID, token=HF_TOKEN)
    return find_model_dir(snap)

model_path = get_model_path()
tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
mdl = AutoModelForSequenceClassification.from_pretrained(
    model_path, low_cpu_mem_usage=True, torch_dtype=torch.float32
).eval()



_env_labels = os.getenv("CLASS_LABELS")
ENV_LABELS = [s.strip() for s in _env_labels.split(",")] if _env_labels else None
DEFAULT_AGNEWS = ["World", "Sports", "Business", "Sci/Tech"]

tok = AutoTokenizer.from_pretrained(model_path)
mdl = AutoModelForSequenceClassification.from_pretrained(model_path)

# ----- normalize id2label so we never show LABEL_2 -----
id2label = getattr(mdl.config, "id2label", None) or {}
# choose source of human names
if ENV_LABELS is not None and len(ENV_LABELS) == mdl.config.num_labels:
    names = ENV_LABELS
else:
    names = DEFAULT_AGNEWS[: mdl.config.num_labels]

# if config has no mapping OR looks like "LABEL_*", override with human names
if not id2label or all(str(v).upper().startswith("LABEL_") for v in id2label.values()):
    mdl.config.id2label = {i: names[i] for i in range(mdl.config.num_labels)}
    mdl.config.label2id = {v: k for k, v in mdl.config.id2label.items()}
ID2LABEL = mdl.config.id2label

CONF_THRESH = 0.5  # Default confidence threshold, adjust as needed

def predict_with_threshold(text: str):
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = mdl(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        conf, pred = torch.max(probs, dim=1)
    conf = float(conf)
    pred_id = int(pred)
    label = ID2LABEL.get(pred_id, f"Class {pred_id}")
    if conf < CONF_THRESH:
        return {"status": "REJECTED", "label_id": pred_id, "label": label, "confidence": conf}
    return {"status": "OK", "label_id": pred_id, "label": label, "confidence": conf}
