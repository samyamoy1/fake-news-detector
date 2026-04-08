# Fake News Detector

A web application that uses traditional Machine Learning (NOT AI) to detect fake news articles.

## How It Works

This project uses:
- **TF-IDF** - Converts text into numbers based on word importance
- **Logistic Regression** - A simple ML algorithm for classification

This is traditional Machine Learning, NOT deep learning or AI.

## Files

- `train_model.py` - Script to train the ML model
- `app.py` - Flask web application
- `templates/index.html` - User interface
- `model.pkl` - Trained model (generated after running train_model.py)

## Setup

1. Install required libraries:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Run the web app:
```bash
python app.py
```

4. Open in browser:
```
http://127.0.0.1:5000
```

## How to Use

1. Enter a news headline
2. Enter article text (optional but recommended)
3. Click "Analyze News"
4. View the result - FAKE or REAL with confidence percentage

## Requirements

- Python 3.x
- Flask
- scikit-learn
- pandas
- nltk
