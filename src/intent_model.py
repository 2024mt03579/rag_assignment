from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_path = "./intent_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

labels = ["QA", "SUMMARIZE", "QUIZ", "COMPLIANCE"]


def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    outputs = model(**inputs)
    logits = outputs.logits

    predicted_class_id = torch.argmax(logits, dim=1).item()

    return labels[predicted_class_id]