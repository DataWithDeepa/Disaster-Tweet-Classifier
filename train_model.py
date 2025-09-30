# train_model.py

import pandas as pd
import re, string, pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

STOP = set(stopwords.words('english'))
LEM = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", str(text))
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [LEM.lemmatize(w) for w in tokens if w not in STOP and w not in string.punctuation]
    return " ".join(tokens)

# Load CSV
df = pd.read_csv("cleaned_disaster_tweets.csv")  # Ensure this CSV is full, clean

df = df.dropna(subset=['text']).reset_index(drop=True)
df['clean_text'] = df['text'].apply(clean_text)

X_text = df['clean_text']
y = df['target']

# Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))  # 5000 features fixed
X = vectorizer.fit_transform(X_text)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X, y)

# Save pickles
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))
pickle.dump(model, open("disaster_tweet_model.pkl", "wb"))

print("✅ Model & Vectorizer saved with 5000 features")
