import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from ml.model import create_model

def train_model(X, y):
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, random_state=42)
    model = create_model(X.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
    return model, lb

# Example usage
if __name__ == "__main__":
    # Mock data for example usage
    from sklearn.feature_extraction.text import TfidfVectorizer
    data = {
        'content': [
            'Case law content 1',
            'Statute content 2',
            'Legal precedent content 3',
            'Case law content 4',
            'Statute content 5'
        ],
        'type': ['case law', 'statute', 'precedent', 'case law', 'statute']
    }
    df = pd.DataFrame(data)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['content'])
    y = df['type']

    # Train model with mock data
    model, lb = train_model(X, y)
