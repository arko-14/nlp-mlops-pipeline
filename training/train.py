import argparse, os, yaml, mlflow, numpy as np, pandas as pd, torch # type: ignore
from datasets import Dataset # type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
from training.utils import set_seed

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def build_ds(data_dir, tokenizer, max_len):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_len)

    train_ds = Dataset.from_pandas(train).map(tok, batched=True)
    test_ds = Dataset.from_pandas(test).map(tok, batched=True)

    keep = ["input_ids", "attention_mask", "label"]
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])
    test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in keep])

    train_ds.set_format(type="torch", columns=keep)
    test_ds.set_format(type="torch", columns=keep)
    return train_ds, test_ds
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    P = load_params()
    set_seed(P["train"]["seed"])
    mlflow.set_experiment("finetune_experiments")

    tok = AutoTokenizer.from_pretrained(P["train"]["model_name"])
    train_ds, test_ds = build_ds(a.data_dir, tok, P["data"]["max_len"])

    model = AutoModelForSequenceClassification.from_pretrained(
        P["train"]["model_name"], num_labels=P["train"]["num_labels"]
    )


    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": accuracy_score(p.label_ids, preds),
                "f1": f1_score(p.label_ids, preds, average="macro")}

    args_tr = TrainingArguments(
        output_dir=a.out_dir,
        learning_rate=P["train"]["learning_rate"],
        per_device_train_batch_size=P["train"]["batch_size"],
        per_device_eval_batch_size=P["train"]["batch_size"],
        num_train_epochs=P["train"]["epochs"],
        evaluation_strategy="epoch",
        logging_steps=50,
        no_cuda=True,  # CPU training
        save_strategy="epoch",
        report_to=[],
    )

    with mlflow.start_run():
        mlflow.log_params({**P["train"], "data_dir": a.data_dir, "out_dir": a.out_dir})

        trainer = Trainer(
            model=model, args=args_tr,
            train_dataset=train_ds.select(range(P["train"]["train_subset"])),
            eval_dataset=test_ds.select(range(P["train"]["eval_subset"])),
            compute_metrics=metrics
        )
        trainer.train()
        mlflow.log_metrics(trainer.evaluate())

        trainer.save_model(a.out_dir)
        tok.save_pretrained(a.out_dir)

        # Confidence-threshold demo:
        text = "I have fever and cough for two days."
        inputs = tok(text, return_tensors="pt", truncation=True, padding=True)
        probs = torch.softmax(model(**inputs).logits, dim=-1)
        conf, pred = torch.max(probs, dim=1)
        print("DEMO:", ("REJECT (low confidence)" if conf.item() < 0.5
               else f"PRED={pred.item()} CONF={conf.item():.2f}"))

