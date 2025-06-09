# 🧠 Sentiment Analysis from Image using Flask + Tesseract + Transformers

This app extracts text from uploaded images and classifies its toxicity using a Hugging Face transformer model.

---

## 🚀 Features

- OCR via Tesseract
- Toxicity detection using `unitary/unbiased-toxic-roberta`
- Flask REST API
- Docker container support

---

## 📦 How to Run

### 🐳 Docker (Recommended)

```bash
docker build -t sentiment-app .
docker run -p 5000:5000 sentiment-app
#Then open: http://localhost:5000
🧪 API Endpoint
POST /upload

Form Field: image (upload an image file)

Returns:

json
Copy
Edit
{
  "extracted_text": "sample",
  "severity": "mild",
  "scores": {
    "toxic": 0.23,
    "insult": 0.48,
    ...
  }
}
📁 Files
app.py — Main Flask app

requirements.txt — Dependencies

Dockerfile — Container setup

README.md — Project guide