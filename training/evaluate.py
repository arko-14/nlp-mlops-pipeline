# training/evaluate.py
import argparse, json, os
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(a.model_dir)

    # Use alias to avoid name shadowing by any local "pipeline" module/file
    clf = hf_pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        device=-1,
        truncation=True,
        return_all_scores=False
    )

    test = pd.read_csv(os.path.join(a.data_dir, "test.csv"))
    texts = test["text"].astype(str).tolist()
    y_true = test["label"].tolist()

    # Map outputs like "LABEL_2" → 2 (or a named label if you trained with names)
    preds = []
    id2label = mdl.config.id2label
    label2id = {v: k for k, v in id2label.items()}
    for t in texts:
        out = clf(t)[0]["label"]  # e.g., "LABEL_2" or a custom name
        if out.startswith("LABEL_"):
            preds.append(int(out.split("_")[-1]))
        else:
            preds.append(int(label2id[out]))

    rep = classification_report(y_true, preds, output_dict=True)
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w") as f:
        json.dump(rep, f, indent=2)

    print("[OK] Wrote report:", a.report)
