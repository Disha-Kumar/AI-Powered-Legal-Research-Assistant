import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def process_data(df):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['content'])
    return X, df['type'], vectorizer

# Example usage
df = collect_data()
X, y, vectorizer = process_data(df)
print(X.shape, y.shape)
