import json
import random

labels = ["QA", "SUMMARIZE", "QUIZ", "COMPLIANCE"]

qa_templates = [
    "What is {}?",
    "Explain {}",
    "Define {}",
    "Give details about {}"
]

summarize_templates = [
    "Summarize {}",
    "Give a summary of {}",
    "Shorten this {}",
    "Explain briefly {}"
]

quiz_templates = [
    "Generate quiz on {}",
    "Create questions about {}",
    "Make test questions for {}",
    "Give 3 quiz questions on {}"
]

compliance_templates = [
    "Is this GDPR compliant: {}",
    "Check HIPAA compliance: {}",
    "Does this violate GDPR: {}",
    "Evaluate compliance for {}"
]

topics = [
    "diabetes", "heart disease", "HIPAA", "GDPR",
    "patient data", "nutrition", "medical records",
    "privacy laws", "healthcare systems", "insurance"
]

data = []

for _ in range(75):  # 75 * 4 = 300 samples
    topic = random.choice(topics)

    data.append({"text": random.choice(qa_templates).format(topic), "label": "QA"})
    data.append({"text": random.choice(summarize_templates).format(topic), "label": "SUMMARIZE"})
    data.append({"text": random.choice(quiz_templates).format(topic), "label": "QUIZ"})
    data.append({"text": random.choice(compliance_templates).format(topic), "label": "COMPLIANCE"})

# Save file
with open("intent_model/train_data/dataset.json", "w") as f:
    json.dump(data, f, indent=2)

print("Dataset generated with", len(data), "samples")