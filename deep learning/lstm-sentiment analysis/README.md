# LSTM Sentiment Analysis

A **Bidirectional LSTM** model for classifying movie review sentiments (Positive/Negative) using the IMDB dataset.

## Overview

This project implements a high-performance sentiment analysis model using TensorFlow/Keras with the following architecture:

- **Embedding Layer**: 128-dimensional word embeddings
- **Bidirectional LSTM**: 64 hidden units for processing context in both directions
- **Spatial Dropout**: 35% dropout on embeddings to prevent overfitting
- **Concatenated Pooling**: Combines Global Max and Global Average Pooling for rich feature extraction
- **Dense Layer**: 64 neurons with ReLU activation
- **Dropout**: 50% regularization
- **Output**: Single sigmoid neuron for binary classification

## Requirements

```bash
pip install tensorflow numpy matplotlib
```

## Project Structure

```
lstm-sentiment analysis/
├── sentiment_analysis.py      # Model training script
├── inference_and_evaluate.py  # Inference & evaluation script
├── sentiment_lstm_model.keras # Trained model
└── README.md
```

## Training

Run the training script to train the model:

```bash
python sentiment_analysis.py
```

This will:
- Load the IMDB dataset (25,000 training, 25,000 test reviews)
- Pad sequences to uniform length (250 words)
- Train the Bidirectional LSTM model for 5 epochs
- Plot training/validation accuracy and loss
- Save the trained model as `sentiment_lstm_model.keras`

## Inference & Evaluation

Run the inference script to evaluate the model and test predictions:

```bash
python inference_and_evaluate.py
```

This will:
- Load the trained model
- Evaluate accuracy on the test set
- Test on tricky sentences with negation, sarcasm, etc.
- Provide an interactive demo for custom reviews

### Output Example

```
Test Loss: 0.4647
Test Accuracy: 0.8557

Review: "This movie was not good."
  Predicted: Negative (Confidence: 0.4472)

Review: "Absolutely fantastic movie! Loved every minute."
  Predicted: Positive (Confidence: 0.9815)
```

## Key Features

- ✅ **Bidirectional LSTM** captures context from both past and future words
- ✅ **Concatenated Pooling** combines max and average pooling for robust features
- ✅ **Spatial Dropout** prevents embedding layer overfitting
- ✅ **Confidence Scores** provided for each prediction
- ✅ **Negation Handling** tested with tricky sentences

## Test Accuracy

- **Test Loss**: ~0.46
- **Test Accuracy**: ~85.5%

## Notes

- The IMDB dataset contains 50,000 movie reviews (balanced positive/negative)
- Reviews are preprocessed: lowercased, non-alphanumeric characters removed
- Word index mapping follows Keras IMDB convention (+3 offset for special tokens)
