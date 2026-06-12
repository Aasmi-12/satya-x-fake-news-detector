from flask import Flask, request, jsonify
from transformers import pipeline
from newspaper import Article

app = Flask(__name__)

# Fake news detection model
classifier = pipeline("text-classification", model="facebook/bart-large-mnli")

def extract_text_from_url(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def analyze_text(text):
    # Simple labels
    labels = ["real news", "fake news"]

    result = classifier(text[:1000], labels)
    
    score = result['scores'][0]
    label = result['labels'][0]

    explanation = []
    
    if "fake" in label.lower():
        explanation.append("Content shows patterns similar to misinformation.")
    else:
        explanation.append("Content appears closer to factual reporting.")

    return {
        "prediction": label,
        "confidence": round(score * 100, 2),
        "explanation": explanation
    }

@app.route('/analyze_text', methods=['POST'])
def analyze_text_api():
    text = request.json.get("text")
    result = analyze_text(text)
    return jsonify(result)

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    url = request.json.get("url")
    text = extract_text_from_url(url)
    result = analyze_text(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)
