"""
Trains the full-size STYLEPRINT fake-news style classifier and saves it
as a single joblib artifact (model.joblib) for the Flask API to load.

Usage:
    python train_model.py --fake Fake.csv --true True.csv --out model.joblib
"""
import argparse
import re
import string
import json

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATELINE_PATTERN = re.compile(r'^.{0,80}?\(Reuters\)\s*-?\s*')


def load_and_clean(fake_path, true_path):
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(true_path)
    fake['label'] = 1  # FAKE
    real['label'] = 0  # REAL

    df = pd.concat([fake, real], ignore_index=True)
    df = df.drop(columns=['date'], errors='ignore')

    df['text'] = df['text'].astype(str).apply(lambda t: DATELINE_PATTERN.sub('', t, count=1))
    df['text'] = df['text'].apply(lambda t: re.sub(r'reuters', '', t, flags=re.IGNORECASE))
    df['text'] = df['text'].apply(lambda t: re.sub(r'photo by.*?(getty images|images)\.?', '', t, flags=re.IGNORECASE))
    df['text'] = df['text'].apply(lambda t: re.sub(r'featured image (is |via ).*', '', t, flags=re.IGNORECASE))
    df['text'] = df['text'].apply(lambda t: re.sub(r'pic\.twitter\.com/\S+', '', t))
    df['text'] = df['text'].apply(lambda t: re.sub(r'https?://\S+|www\.\S+', '', t))

    df['text'] = df['title'].astype(str) + ' ' + df['text'].astype(str)
    df = df.drop(columns=['title'])

    df = df[df['text'].str.strip().str.len() > 20]
    df = df.drop_duplicates(subset='text')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake', default='Fake.csv')
    parser.add_argument('--true', default='True.csv')
    parser.add_argument('--out', default='model.joblib')
    parser.add_argument('--max-features', type=int, default=20000)
    args = parser.parse_args()

    print("Loading and cleaning data...")
    df = load_and_clean(args.fake, args.true)
    df['clean_text'] = df['text'].apply(clean_text)
    print(f"Total articles after cleaning: {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['label'], test_size=0.15, random_state=42, stratify=df['label']
    )

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(
        stop_words='english', max_df=0.9, min_df=5,
        max_features=args.max_features, ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training model...")
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_tfidf, y_train)

    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"Held-out accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=['REAL', 'FAKE'], zero_division=0))

    artifact = {
        'vectorizer': vectorizer,
        'model': model,
        'accuracy': acc,
        'n_features': len(vectorizer.get_feature_names_out()),
        'trained_on': 'Kaggle Fake and Real News Dataset',
    }
    joblib.dump(artifact, args.out)
    print(f"Saved model artifact to {args.out}")


if __name__ == '__main__':
    main()
