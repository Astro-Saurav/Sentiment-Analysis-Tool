# app.py

from flask import Flask, request, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = Flask(__name__)

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('./bert_model')
model = BertForSequenceClassification.from_pretrained('./bert_model')

# Define the label mapping
label_mapping = {0: 'positive', 1: 'negative', 2: 'neutral'}

def preprocess_and_predict(text):
    encodings = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encodings)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return label_mapping[prediction]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    sentiment = preprocess_and_predict(text)
    return render_template('index.html', prediction=sentiment, text=text)

if __name__ == '__main__':
    app.run(debug=True)
