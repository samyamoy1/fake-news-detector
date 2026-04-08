"""
Fake News Detector Web App
This is a Flask web application using traditional ML (NOT AI)
"""

import os
import pickle
from flask import Flask, render_template, request, jsonify
from train_model import clean_text

# Create Flask app
app = Flask(__name__)

# Path to saved model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

# Global variable to store loaded model
model = None


def load_trained_model():
    """Load the trained ML model from file"""
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def get_model():
    """Get the model (load if not already loaded)"""
    global model
    if model is None:
        model = load_trained_model()
    return model


@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on news article"""
    # Get input from user
    data = request.get_json()
    headline = data.get('headline', '')
    text = data.get('text', '')
    
    # Validate input
    if not headline and not text:
        return jsonify({'error': 'Please provide headline or text'}), 400
    
    # Combine headline and text
    combined = headline + ' ' + text
    
    # Clean the text (same process as training)
    processed = clean_text(combined)
    
    # Get the trained model
    clf = get_model()
    
    # Make prediction (0 = Real, 1 = Fake)
    prediction = clf.predict([processed])[0]
    
    # Get prediction probabilities
    probabilities = clf.predict_proba([processed])[0]
    
    # Prepare response
    result = 'FAKE' if prediction == 1 else 'REAL'
    confidence = float(max(probabilities)) * 100
    
    return jsonify({
        'result': result,
        'confidence': round(confidence, 2),
        'fake_probability': round(probabilities[1] * 100, 2),
        'real_probability': round(probabilities[0] * 100, 2)
    })


# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5000)
