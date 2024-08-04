# Sentiment Analysis Tool

This project is a Sentiment Analysis Tool built using BERT (Bidirectional Encoder Representations from Transformers) and Flask. It classifies text into positive, negative, or neutral sentiments. The model is trained on a dataset and served through a web interface built with Flask.

## Features

- **State-of-the-art NLP**: Uses BERT for high accuracy in sentiment classification.
- **Custom Dataset**: Trained on a custom dataset for sentiment analysis.
- **Web Interface**: Provides an easy-to-use web interface for text sentiment prediction.
- **Deployable**: Can be deployed on platforms like Heroku for public access.

## Project Structure
**
my_sentiment_app/
├── app.py
├── train_model.py
├── sentiment_data.csv
├── bert_model/
│ ├── config.json
│ ├── pytorch_model.bin
│ ├── tokenizer_config.json
│ ├── vocab.txt
├── requirements.txt
├── Procfile
└── templates/
└── index.html **
