import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from app.preprocess import clean_text
import joblib
import os

# Load dataset
df = pd.read_csv('data/reviews.csv')

# Clean text
df['cleaned_review'] = df['review'].apply(clean_text)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review'])

# Labels
y = df['sentiment']

# Save vectorizer for app usage later
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print(f"Feature matrix shape: {X.shape}")
