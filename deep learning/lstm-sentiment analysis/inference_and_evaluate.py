# -*- coding: utf-8 -*-
"""
Inference & Evaluation Script

Phase 3: Inference & Evaluation

* Evaluate your final model's accuracy on the unseen test set.
* Write a custom inference function that accepts any raw text string and predicts its sentiment.
* Test your model on tricky sentences (e.g., reviews containing negation like "not good") to verify it captures word order.
"""

import os
import numpy as np
import re

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model_path = 'sentiment_lstm_model.keras'
print("Loading model...")
model = load_model(model_path)
print("Model loaded successfully.")
print(f"Model loaded from {model_path}")

# Hyperparameters (must match training)
vocab_size = 10000
max_length = 250

# Load word index for preprocessing
word_index = imdb.get_word_index()


def preprocess_text(text):
    """
    Preprocess raw text for inference.
    Converts text to padded sequence matching training format.
    """
    # Clean text: remove non-alphanumeric characters
    cleaned_text = re.sub(r'[^a-zA-Z0-9 ]', '', text.lower())
    words = cleaned_text.split()
    
    # Encode words to indices
    encoded = [1]  # Start token
    for word in words:
        index = word_index.get(word, 2) + 3  # Adjust mapping offset
        if index < vocab_size:
            encoded.append(index)
        else:
            encoded.append(2)  # Unknown word
    
    # Pad sequence
    padded = pad_sequences([encoded], maxlen=max_length, padding='post', truncating='post')
    return padded


def predict_sentiment(text):
    """
    Predict sentiment for any raw text string.
    
    Args:
        text (str): Raw text input
        
    Returns:
        tuple: (sentiment_label, confidence_score, raw_probability)
            - sentiment_label: 'Positive' or 'Negative'
            - confidence_score: adjusted confidence (distance from 0.5)
            - raw_probability: model's output probability
    """
    # Preprocess text
    padded = preprocess_text(text)
    
    # Get prediction
    raw_prob = model.predict(padded, verbose=0)[0][0]
    
    # Determine sentiment
    sentiment = "Positive" if raw_prob >= 0.5 else "Negative"
    
    # Calculate confidence (how far from 0.5)
    confidence = abs(raw_prob - 0.5) * 2  # Scale to 0-1 range
    
    return sentiment, confidence, raw_prob


def evaluate_on_test_set():
    """
    Evaluate the model on the unseen test set.
    Returns accuracy and detailed metrics.
    """
    from tensorflow.keras.datasets import imdb
    
    # Load test data
    (_, _), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
    
    # Pad sequences
    x_test_padded = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')
    
    # Evaluate
    loss, accuracy = model.evaluate(x_test_padded, y_test, verbose=0)
    
    return {
        'loss': loss,
        'accuracy': accuracy
    }


def test_tricky_sentences():
    """
    Test model on tricky sentences with negation and complex sentiment patterns.
    """
    tricky_reviews = [
        # Negation patterns
        "This movie was not good.",
        "I did not enjoy this film at all.",
        "The acting was far from good.",
        "Not the worst movie I've seen, but definitely not good either.",
        
        # Mixed sentiment
        "The story was great, but the pacing was terrible.",
        "Beautiful cinematography, terrible dialogue.",
        
        # Double negation
        "I can't say I didn't enjoy it.",
        "It's not impossible to recommend, though I wouldn't say it's great.",
        
        # Sarcasm/irony
        "Oh great, another masterpiece by this director.",
        "Wow, such a predictable plot, totally surprising!",
        
        # Clear positive/negative
        "Absolutely fantastic movie! Loved every minute.",
        "Terrible waste of time. Very disappointing.",
    ]
    
    print("=" * 80)
    print("Testing Tricky Sentences")
    print("=" * 80)
    
    results = []
    for review in tricky_reviews:
        sentiment, confidence, raw_prob = predict_sentiment(review)
        results.append({
            'text': review,
            'sentiment': sentiment,
            'confidence': confidence,
            'raw_prob': raw_prob
        })
        
        print(f"\nReview: \"{review}\"")
        print(f"  Predicted: {sentiment} | Confidence: {confidence:.4f} | Probability: {raw_prob:.4f}")
    
    return results


def main():
    """Run all evaluation and inference tests."""
    print("\n" + "=" * 80)
    print("EVALUATION & INFERENCE")
    print("=" * 80 + "\n")
    
    # 1. Evaluate on test set
    print("\n--- Phase 3.1: Test Set Evaluation ---\n")
    metrics = evaluate_on_test_set()
    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    
    # 2. Test tricky sentences
    print("\n")
    test_tricky_sentences()
    
    # 3. Interactive demo
    print("\n" + "-" * 80)
    print("Interactive Demo - Enter custom reviews (or 'quit' to exit)")
    print("-" * 80)
    
    while True:
        user_input = input("\nEnter a review: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        sentiment, confidence, raw_prob = predict_sentiment(user_input)
        print(f"Predicted Sentiment: {sentiment}")
        print(f"Confidence Score: {confidence:.4f}")
        print(f"Probability: {raw_prob:.4f}")


if __name__ == "__main__":
    main()
