"""
Fake News Detector - Using Traditional Machine Learning
This uses TF-IDF and Logistic Regression (NOT AI/Deep Learning)
"""

import os
import re
import pickle
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (stopwords and lemmatizer)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Get English stopwords (words like "the", "is", "at" that don't add meaning)
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    """Clean and normalize text for the model"""
    # Handle empty or non-string input
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove numbers and special characters (keep only letters)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Split into words
    words = text.split()
    
    # Remove stopwords and apply lemmatization (convert words to base form)
    # Example: "running" -> "run", "children" -> "child"
    words = [LEMMATIZER.lemmatize(word) for word in words if word not in STOPWORDS]
    
    return ' '.join(words)


def create_training_data():
    """Create sample labeled data for training the model"""
    
    # Each item is a complete example: headline + text = fake or real
    fake_examples = [
        ("BREAKING: Scientists discover miracle cure for cancer", "Shocking discovery reveals hidden cure pharmaceutical companies don't want you to know about"),
        ("SHOCKING: Government hides alien contact evidence", "Top secret documents leaked show aliens visited Earth decades ago"),
        ("YOU WON'T BELIEVE: Celebrity secret exposed", "Famous actor reveals shocking secret that media tried to hide"),
        ("OMGB: Banks about to collapse - get your money out now", "Insider reveals major banks will fail within days"),
        ("EXCLUSIVE: New world order conspiracy revealed", "Secret society plans to control world population"),
        ("URGENT: Massive asteroid heading toward Earth", "NASA hiding truth about asteroid that will hit Earth"),
        ("INSIDE STORY: Big pharma's secret cancer cure", "Drug companies hide cure to keep selling medicine"),
        ("WATCH: Politician caught in massive scandal", "Undercover video reveals politician's dark secret"),
        ("CONFIRMED: 5G causes coronavirus", "Scientists prove link between 5G towers and virus"),
        ("ALERT: World war 3 about to start", "Military sources confirm war is imminent"),
        ("BREAKING: Scientists discover miracle cure for diabetes", "Big pharma hides one simple cure that works instantly"),
        ("SHOCKING: Government tracks citizens through phones", "Secret program monitors all your activities"),
        ("EXPOSED: Celebrity hidden wealth scheme", "Rich celebrities hiding money from taxes"),
        ("WARNING: Food companies poison your food", "Hidden ingredients causing cancer in everyone"),
        ("ALERT: Your bank account will be frozen", "New rule takes your money starting tomorrow"),
    ]
    
    real_examples = [
        ("Scientists discover new treatment for disease", "Researchers published findings in medical journal after years of study"),
        ("Government announces new economic policy", "Central bank releases statement about interest rates"),
        ("Celebrity announces new film project", "Actor reveals upcoming movie plans at press conference"),
        ("Bank reports quarterly earnings", "Financial institution reports profit increase this quarter"),
        ("Researchers publish study on climate change", "Scientists analyze temperature data from past decades"),
        ("New asteroid discovered by astronomers", "Space agency confirms new asteroid in solar system"),
        ("Pharmaceutical company announces trial results", "Company shares results of drug testing phase"),
        ("Politician responds to voter concerns", "Official addresses public questions at town hall"),
        ("Health officials report on virus spread", "Department of health releases weekly statistics"),
        ("Countries sign international agreement", "Nations reach deal on environmental protection"),
        ("Scientists develop improved method for detection", "New technique allows earlier identification of disease"),
        ("Government launches small business support", "Program provides loans to local entrepreneurs"),
        ("Stock markets close higher today", "Trading ends with gains across major indices"),
        ("Weather forecast predicts rainfall", "Meteorological department issues weekly outlook"),
        ("Company expands to new markets", "Business announces plans for international growth"),
    ]
    
    # Create dataset with unique headline-text pairs
    data = []
    
    # Add fake examples (label = 1)
    for headline, text in fake_examples:
        for _ in range(3):  # Repeat each example 3 times with slight variations
            data.append({
                'headline': headline,
                'text': text,
                'label': 1
            })
    
    # Add real examples (label = 0)
    for headline, text in real_examples:
        for _ in range(3):
            data.append({
                'headline': headline,
                'text': text,
                'label': 0
            })
    
    return pd.DataFrame(data)


def train_model():
    """Main function to train the fake news classifier"""
    print("Creating training data...")
    df = create_training_data()
    
    print(f"Total samples: {len(df)}")
    print(f"Fake samples: {(df['label'] == 1).sum()}")
    print(f"Real samples: {(df['label'] == 0).sum()}")
    
    # Combine headline and text
    df['combined'] = df['headline'] + ' ' + df['text']
    
    # Clean the text
    df['processed'] = df['combined'].apply(clean_text)
    
    # Separate features (X) and labels (y)
    X = df['processed']
    y = df['label']
    
    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training machine learning model...")
    
    # Create a pipeline:
    # 1. TF-IDF: Converts text into numbers based on word importance
    # 2. Logistic Regression: Simple ML algorithm for classification
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Check accuracy on test data
    accuracy = pipeline.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2%}")
    
    # Save the trained model to a file
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {model_path}")


# Run training when script is executed
if __name__ == "__main__":
    train_model()
