import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# ----------------------------
# CONFIG
# ----------------------------
CSV_PATH = "messages.csv"
TEXT_COL = "message_text"
LABEL_COL = "contains_trigger"

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128

EPOCHS = 1
BATCH_SIZE = 8
LR = 2e-5

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(CSV_PATH)

print("Columns:", df.columns)

# Clean text
df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

# Label
df[LABEL_COL] = df[LABEL_COL].fillna(0).astype(int)

# ----------------------------
# SPLIT
# ----------------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.4,
    random_state=42,
    stratify=df[LABEL_COL]
)

# ----------------------------
# TOKENIZER
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=MAX_LEN
    )

train_enc = tokenize(train_df[TEXT_COL].tolist())
test_enc = tokenize(test_df[TEXT_COL].tolist())

# ----------------------------
# DATASET CLASS
# ----------------------------
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.enc = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_enc, train_df[LABEL_COL].tolist())
test_dataset = Dataset(test_enc, test_df[LABEL_COL].tolist())

# ----------------------------
# MODEL
# ----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# ----------------------------
# TRAINING
# ----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    report_to="none"
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# ----------------------------
# TRAIN
# ----------------------------
trainer.train()

# ----------------------------
# EVALUATE
# ----------------------------
print("\nFinal Results:")
print(trainer.evaluate())






