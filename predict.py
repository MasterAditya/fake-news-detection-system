import joblib
from preprocess import TextPreprocessor

class NewsPredictor:
    def __init__(self):
        self.model = joblib.load('models/model.joblib')
        self.vectorizer = joblib.load('models/vectorizer.joblib')
        self.preprocessor = TextPreprocessor()
    
    def predict(self, text):
        """
        Predict whether a news article is fake or real.
        Returns prediction (0 for real, 1 for fake) and probability.
        """
        # Preprocess the input text
        processed_text = self.preprocessor.preprocess_text(text)
        
        # Vectorize the text
        text_vectorized = self.vectorizer.transform([processed_text])
        
        # Get prediction and probability
        prediction = self.model.predict(text_vectorized)[0]
        probability = self.model.predict_proba(text_vectorized)[0]
        
        return {
            'prediction': 'FAKE' if prediction == 1 else 'REAL',
            'probability': probability[1] if prediction == 1 else probability[0]
        }
