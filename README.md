# MenuCriticAPI
## Aspect-Based Sentiment Analysis Model
This project is an implementation of a machine learning model for aspect-based sentiment analysis on restaurant reviews. The model is built using PyTorch, Transformers, and scikit-learn libraries. The original inspiration for this project comes from https://github.com/1tangerine1day/Aspect-Term-Extraction-and-Analysis.
## Model Architecture
The model architecture consists of the following layers:

1. BERT (Bidirectional Encoder Representations from Transformers) as the base pre-trained model.

2. Two Bidirectional LSTM layers for sequence modeling.

3. A linear layer for classification.

The input to the model is a review text and its aspect term, and the output is the sentiment of the aspect term in the review (positive, negative, or neutral).

## Dataset
The model is trained on a dataset of restaurant reviews, which is annotated with aspect terms and their polarities.

## Evaluation
The model is evaluated on precision, recall, and F1 score metrics using the test set. The evaluation results are displayed at the end of the training process.


