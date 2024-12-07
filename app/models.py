import numpy as np

def recommend_document(query, model, vectorizer, lb):
    X_query = vectorizer.transform([query])
    prediction = model.predict(X_query.toarray())
    predicted_label = lb.inverse_transform(prediction)[0]
    return predicted_label
