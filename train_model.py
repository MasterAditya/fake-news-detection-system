import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from preprocess import TextPreprocessor

def create_mock_data(n_samples=1000):
    """Create mock dataset for testing purposes."""
    fake_news = [
        "SHOCKING: Celebrity admits to being alien in disguise!",
        "Scientists discover pizza is the key to immortality",
        "Government hiding dragons in secret underground facility",
        "Breaking: Moon officially declared to be made of cheese"
    ]
    
    real_news = [
        "New study shows benefits of regular exercise",
        "Local community opens new public library",
        "Scientists discover new species in Amazon rainforest",
        "Stock market shows signs of recovery after recent decline"
    ]
    
    # Generate synthetic data
    np.random.seed(42)
    fake_templates = np.random.choice(fake_news, n_samples // 2)
    real_templates = np.random.choice(real_news, n_samples // 2)
    
    data = {
        'text': np.concatenate([fake_templates, real_templates]),
        'label': np.concatenate([np.ones(n_samples // 2), np.zeros(n_samples // 2)])
    }
    
    return pd.DataFrame(data)

def train_model():
    """Train the fake news detection model."""
    # Load and preprocess data
    df = create_mock_data()
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess texts
    df['processed_text'] = df['text'].apply(preprocessor.preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], 
        test_size=0.2, random_state=42
    )
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_tfidf)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model and vectorizer
    joblib.dump(model, 'models/model.joblib')
    joblib.dump(vectorizer, 'models/vectorizer.joblib')
    
    return {
        'model': model,
        'vectorizer': vectorizer
    }

if __name__ == '__main__':
    train_model()
