# Sentiment Analysis Tool

This project is a Sentiment Analysis Tool built using BERT (Bidirectional Encoder Representations from Transformers) and Flask. It classifies text into positive, negative, or neutral sentiments. The model is trained on a dataset and served through a web interface built with Flask.

![Screenshot 2024-08-04 161836](https://github.com/user-attachments/assets/13b63187-9180-4d7a-ac94-b8901b63421b)

![Screenshot 2024-08-04 161934](https://github.com/user-attachments/assets/1753d77c-219d-451b-b852-5bcd4cfa70a6)

## Features

- **State-of-the-art NLP**: Uses BERT for high accuracy in sentiment classification.
- **Custom Dataset**: Trained on a custom dataset for sentiment analysis.
- **Web Interface**: Provides an easy-to-use web interface for text sentiment prediction.
- **Deployable**: Can be deployed on platforms like Heroku for public access.

## Project Structure

```plaintext
my_sentiment_app/
├── app.py
├── train_model.py
├── sentiment_data.csv
├── bert_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── vocab.txt
├── requirements.txt
├── Procfile
└── templates/
    └── index.html
```

## Setup

### Prerequisites

- Python 3.7 or higher
- Pip (Python package installer)
- Git
- Heroku CLI (for deployment)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Astro-Saurav/Sentiment-Analysis-Tool 
   cd Sentiment-Analysis-Tool
   ```

3. Create and activate a virtual environment (optional but recommended):
   `python -m venv venv`
   `source venv/bin/activate`  # On Windows use `venv\Scripts\activate`

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

6. Train the model:
   Ensure your `sentiment_data.csv` file is in the project directory. Then, run:
   `python train_model.py`
   This script will train the BERT model and save it in the `bert_model` directory.

7. Run the Flask app:
   `python app.py`
   Open your browser and go to `http://127.0.0.1:5000/` to access the sentiment analysis tool.

## Contributing

Contributions are welcome! Feel free to fork this repository, create a feature branch, and submit a pull request. Any improvements and suggestions are appreciated.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Astro-Saurav/Sentiment-Analysis-Tool/blob/main/LICENSE) file for details.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the BERT model.
- [Flask](https://flask.palletsprojects.com/) for the web framework.

## Contact

If you have any questions, feel free to reach out via GitHub issues or contact me directly [email](mailto:0501saurav@gmail.com).

---

Happy coding!
