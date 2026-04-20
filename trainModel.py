import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load dataset
with open("intent_model/train_data/dataset.json", "r") as f:
    data = json.load(f)

# Label mapping
label_map = {
    "QA": 0,
    "SUMMARIZE": 1,
    "QUIZ": 2,
    "COMPLIANCE": 3
}

# Convert labels to numeric
for item in data:
    item["label"] = label_map[item["label"]]

dataset = Dataset.from_list(data)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

dataset = dataset.map(tokenize)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4
)

# Training config
training_args = TrainingArguments(
    output_dir="./intent_model",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

# Save model
trainer.save_model("./intent_model")
tokenizer.save_pretrained("./intent_model")

print("✅ Training complete. Model saved.")