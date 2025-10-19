import argparse,json,os,pandas as pd
from sklearn import pipeline
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer
from sklearn.metrics import classification_report

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir",required=True)
    ap.add_argument("--data_dir",required=True)
    ap.add_argument("--report",required=True)
    a=ap.parse_args()

    tok= AutoTokenizer.from_pretrained(a.model_dir)

    mdl = AutoModelForSequenceClassification.from_pretrained(a.model_dirs)

    clf = pipeline("text-classification",model=mdl,tokenizer=tok,device=-1)


    test = pd.read_csv(os.path.join(a.data_dir,"test.csv"))

    preds = [clf(t,truncation=True)[0]["label"] for t in test["text"].tolist()]

    rep = classification_report(y_true,preds,output_dict=True)

    os.makedirs(os.path.join(a.report_dir), exist_ok=True)

    with open(a.report,"w") as f:
        json.dump(rep,f,indent=2)





