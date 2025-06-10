from flask import Flask, render_template, request, jsonify
import pytesseract
from PIL import Image
import io
from transformers import pipeline
import os

app = Flask(__name__)

# 🧠 Only set this path if running locally on Windows
if os.name == 'nt':  # 'nt' means Windows
    pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Load the Hugging Face toxicity classifier once
classifier = pipeline("text-classification", model="unitary/unbiased-toxic-roberta", return_all_scores=True)

def get_severity(scores):
    bullying_labels = ["toxic", "insult", "severe_toxic", "threat", "obscene"]
    bullying_scores = [scores[label] for label in bullying_labels if label in scores]

    if not bullying_scores:
        return "safe"
    
    max_score = max(bullying_scores)
    if max_score < 0.3:
        return "safe"
    elif max_score < 0.5:
        return "mild"
    elif max_score < 0.7:
        return "insult"
    else:
        return "severe"

@app.route('/')
def index():
    return render_template("index.html")  # You'll need an index.html file

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['image']

    try:
        image = Image.open(io.BytesIO(file.read()))
        extracted_text = pytesseract.image_to_string(image)

        result = classifier(extracted_text)
        scores = {res["label"]: res["score"] for res in result[0]}
        severity = get_severity(scores)

        return jsonify({
            "extracted_text": extracted_text,
            "severity": severity,
            "scores": scores
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Make it accessible both locally and in Docker/GCP
    app.run(host='0.0.0.0', port=5000, debug=True)
