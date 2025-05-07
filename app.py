import streamlit as st
import plotly.graph_objects as go
from predict import NewsPredictor
import os

def create_gauge_chart(probability, prediction):
    """Create a gauge chart for probability visualization."""
    color = "red" if prediction == "FAKE" else "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        },
        title={'text': f"Confidence Score (%)"}
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Page config
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stAlert {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("🔍 Fake News Detector")
    st.markdown("""
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
        This tool uses machine learning to analyze news articles and determine if they're likely to be fake or real.
        Simply paste your article text below and click 'Analyze' to get the prediction.
        </div>
    """, unsafe_allow_html=True)
    
    # Check if model exists
    if not os.path.exists('models/model.joblib'):
        st.error("Model not found. Please run train_model.py first to train the model.")
        return
    
    # Initialize predictor
    predictor = NewsPredictor()
    
    # Text input
    text = st.text_area("📰 Paste your news article here:", height=200)
    
    # Make prediction when user clicks the button
    if st.button("🔍 Analyze"):
        if not text:
            st.warning("Please enter some text to analyze.")
            return
        
        # Get prediction
        result = predictor.predict(text)
        
        # Create two columns for results
        col1, col2 = st.columns(2)
        
        with col1:
            # Display prediction result
            if result['prediction'] == 'FAKE':
                st.error(f"### ⚠️ This article appears to be FAKE!")
            else:
                st.success(f"### ✅ This article appears to be REAL!")
        
        with col2:
            # Display gauge chart
            st.plotly_chart(create_gauge_chart(
                result['probability'],
                result['prediction']
            ))
        
        # Display confidence information
        st.info(f"""
            ### 📊 Analysis Details
            - Prediction: {result['prediction']}
            - Confidence Score: {result['probability']:.1%}
            - Confidence Level: {'High' if result['probability'] > 0.8 else 'Moderate' if result['probability'] > 0.6 else 'Low'}
        """)

if __name__ == "__main__":
    main()
