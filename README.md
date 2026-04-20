# API Driven Cloud Native Solutions - Assignment II

## 🩺 AI Healthcare Assistant (RAG + LLM + Fine-Tuning)

This project implements an AI-powered healthcare assistant using:

- OpenAI GPT (gpt-3.5-turbo) for generative tasks
- Retrieval-Augmented Generation (RAG) using PDF documents
- Fine-tuned Hugging Face model (DistilBERT) for intent classification

---

## 📦 Features

- Question Answering (RAG-based)
- Text Summarization
- Quiz Generation
- Compliance Checking (HIPAA / GDPR)
- Intent Classification (Fine-tuned model)

---

## ⚙️ Setup Instructions

### 1. Create Virtual Environment

```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your_api_key_here"
```

> Note: Required for LLM-based tasks such as QA, summarization, quiz generation, and compliance checking.

---

## 🚀 Run Application

```bash
streamlit run src/app.py
```

---

## 📄 Documents for RAG

Place all healthcare-related PDF documents inside the `documents/` directory.

Example:

```
documents/
├── hipaa.pdf
├── gdpr.pdf
├── healthcare_notes.pdf
```

> Ensure PDFs contain selectable text (not scanned images).

---

## 🧠 Fine-Tuning (Intent Classification)

### Step 1: Generate Dataset

```bash
python generate_dataset.py
```

### Step 2: Train Model

```bash
python intent_model/trainModel.py
```

After training, model files will be saved in:

```
intent_model/
```

---

## 🧠 Architecture Overview

- RAG Pipeline: Retrieves document chunks using FAISS
- OpenAI GPT: Generates responses
- Fine-tuned DistilBERT: Classifies user intent

---

## 🔍 Example Queries

- What is HIPAA?
- Summarize Privacy in GDPR
- Generate quiz on HIPAA
- Check GDPR compliance for this patient record

---

## ⚠️ Notes

- OpenAI API key is required for generative tasks
- Intent classification runs locally (no API required)
- Ensure PDFs are text-based

---

## 🧰 Technologies Used

- Streamlit
- LangChain
- FAISS
- OpenAI API
- Hugging Face Transformers
- PyTorch

---

## 🧾 Assignment Coverage

- Domain: Healthcare  
- Category: Natural Language Processing (NLP)

### Subtasks:
- Question Answering (RAG-based)
- Text Generation
- Text Classification

### Fine-Tuning:
- DistilBERT trained on ~300 custom samples

### RAG:
- Implemented using healthcare PDF documents