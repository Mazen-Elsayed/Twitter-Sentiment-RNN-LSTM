# Twitter-Sentiment-RNN-LSTM

## Project Description
This repository contains a Python-based Jupyter notebook implementing sentiment analysis on tweets using Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Feedforward Neural Networks. The project processes a Twitter dataset with preprocessing steps including text cleaning, tokenization, lemmatization, and stopword removal. It employs data augmentation via synonym replacement to address class imbalance and uses GloVe and Word2Vec embeddings for word representations. Multiple models, including RNN, LSTM, Enhanced LSTM, Bidirectional LSTM, and Feedforward variants, are trained and evaluated using metrics like accuracy, precision, recall, and F1-score. Exploratory Data Analysis (EDA) visualizes class distribution, tweet lengths, and frequent words. The code supports PyTorch and is compatible with Google Colab or local environments, with experiments comparing model performance on sentiment classification tasks.

## Key Features
- **Preprocessing**: Text cleaning (removing URLs, hashtags, usernames), tokenization, lemmatization, and custom stopword removal.
- **Data Augmentation**: Synonym replacement to balance class distribution.
- **Models**: RNN, LSTM, Enhanced LSTM, Bidirectional LSTM, Bidirectional LSTM with Transformer Encoder, Feedforward, and Tuned Feedforward.
- **Embeddings**: GloVe and Word2Vec for improved word representations.
- **EDA**: Visualizations of class distribution, tweet length, and frequent words per class.
- **Evaluation**: Metrics include accuracy, precision, recall, and F1-score, with confusion matrices.

## Requirements
- Python 3.12.6
- PyTorch, NumPy, Pandas, NLTK, Matplotlib, Seaborn, Scikit-learn
- Jupyter Notebook
