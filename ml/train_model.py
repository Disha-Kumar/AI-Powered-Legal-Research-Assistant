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
model, lb = train_model(X, y)
