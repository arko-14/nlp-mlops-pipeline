from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch, os

MODEL_DIR = os.environ.get("MODEL_DIR", "models/fine_tuned")
CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.5"))

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

def predict_with_threshold(text: str):
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True)
    logits = mdl(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    conf, pred = torch.max(probs, dim=1)
    if conf.item() < CONF_THRESH:
        return {"label": "REJECTED", "confidence": float(conf)}
    return {"label": int(pred), "confidence": float(conf)}
