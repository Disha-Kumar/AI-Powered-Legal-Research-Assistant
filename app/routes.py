from flask import Flask, jsonify, request, render_template
from data.collect_data import collect_data
from data.process_data import process_data
from ml.train_model import train_model
from app.models import recommend_document
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation():
    data = request.json
    query = data['query']
    
    df = collect_data()
    X, y = process_data(df)
    model, lb = train_model(X, y)
    vectorizer = TfidfVectorizer().fit(df['content'])
    recommended_document = recommend_document(query, model, vectorizer, lb)
    
    return jsonify({'recommended_document': recommended_document})

if __name__ == '__main__':
    app.run(debug=True)
