from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os, torch

MODEL_ID  = os.getenv("MODEL_ID", "sandipan14/fine-grained")
CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.5"))

# Optional: set your own labels via env, e.g. CLASS_LABELS="World,Sports,Business,Sci/Tech"
_env_labels = os.getenv("CLASS_LABELS")
ENV_LABELS = [s.strip() for s in _env_labels.split(",")] if _env_labels else None
DEFAULT_AGNEWS = ["World", "Sports", "Business", "Sci/Tech"]

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

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
# local helper mapping
ID2LABEL = mdl.config.id2label

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
