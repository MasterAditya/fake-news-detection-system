# Fake News Detection System

A lightweight machine learning application that uses Natural Language Processing (NLP) to classify news articles as either real or fake.

## Features

- Text preprocessing using NLTK (lowercasing, stopword removal, lemmatization)
- TF-IDF vectorization for text feature extraction
- Logistic Regression model for binary classification
- Interactive Streamlit web interface
- Real-time prediction with confidence scores
- Visualization using Plotly gauge charts

## Project Structure

```
Fake_News_Detector/
├── app.py              # Streamlit web application
├── predict.py          # Prediction module
├── preprocess.py       # Text preprocessing module
├── train_model.py      # Model training script
├── requirements.txt    # Project dependencies
├── models/            # Directory for saved models
└── data/             # Directory for datasets
```

## Setup and Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Run the web application:
```bash
streamlit run app.py
```

## Usage

1. Start the Streamlit app
2. Paste your news article text in the input area
3. Click "Analyze" to get the prediction
4. View the results and confidence score

## Technical Details

- **Preprocessing**: Includes lowercasing, special character removal, stopword removal, and lemmatization
- **Feature Extraction**: TF-IDF vectorization with 5000 max features
- **Model**: Logistic Regression with balanced class weights
- **Evaluation**: Includes accuracy, precision, recall, and F1-score metrics

## Note

This is a simplified version using mock data for educational purposes. For production use, you should:
- Use a real, comprehensive dataset
- Implement cross-validation
- Add more sophisticated feature engineering
- Consider using more advanced models
