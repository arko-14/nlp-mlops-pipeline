# fix_config.py
from transformers import AutoConfig
labels = ["World","Sports","Business","Sci/Tech"]  # adjust if you changed classes

cfg = AutoConfig.from_pretrained(
    "distilbert-base-uncased",         # base architecture you fine-tuned
    num_labels=len(labels),
    id2label={i: l for i, l in enumerate(labels)},
    label2id={l: i for i, l in enumerate(labels)}
)

cfg.save_pretrained("fixed_config")    # writes fixed_config/config.json
print("Wrote fixed_config/config.json")
