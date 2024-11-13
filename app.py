from flask import Flask, request, jsonify, render_template
import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from PIL import Image
import pytesseract
import os

app = Flask(__name__)

# Load and prepare the corpus from JSON
def load_corpus(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for chapter in data['chapters']:
        for section in chapter['sections']:
            for article in section['articles']:
                content = article.get('content', '')
                records.append({
                    'content': content
                })
    return pd.DataFrame(records)

# Train TF-IDF vectorizer and save models
def train_vectorizer(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['content'])
    
    # Save vectorizer and TF-IDF matrix
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    np.save('tfidf_matrix.npy', tfidf_matrix.toarray())

# Load vectorizer and TF-IDF matrix
def load_model():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    tfidf_matrix = np.load('tfidf_matrix.npy')
    return vectorizer, tfidf_matrix

# Extract text from an image file
def extract_text_from_image(image_path):
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text.strip() or "File does not contain text."
    except Exception as e:
        return "Error processing image."

# Check if text is related to commercial court
def is_relevant_text(text, vectorizer, tfidf_matrix):
    query_vec = vectorizer.transform([text])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    return similarity.max() > 0.5  # Threshold for relevance

# Get a relevant response based on a text query
def get_response(query, corpus_df, vectorizer, tfidf_matrix):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    best_match_index = similarity.argmax()
    return corpus_df.iloc[best_match_index]['content']

# Main route for rendering the chatbot interface
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint for handling text and image queries
@app.route('/api/query', methods=['POST'])
def query():
    user_message = request.form.get('message')
    file = request.files.get('file')
    
    # Load corpus and model
    corpus_df = load_corpus('data/Enhanced_CommercialCourtCorpus.json')
    vectorizer, tfidf_matrix = load_model()
    
    # Handle text query
    if user_message:
        response = get_response(user_message, corpus_df, vectorizer, tfidf_matrix)
    
    # Handle image upload
    elif file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)
        
        # Extract text from image
        text = extract_text_from_image(file_path)
        
        if text == "File does not contain text.":
            response = text
        elif is_relevant_text(text, vectorizer, tfidf_matrix):
            response = get_response(text, corpus_df, vectorizer, tfidf_matrix)
        else:
            response = "Not related to commercial court."
    else:
        response = "No valid input provided."

    return jsonify({"response": response})

# Train the vectorizer if not already trained
def initialize():
    corpus_df = load_corpus('data/Enhanced_CommercialCourtCorpus.json')
    train_vectorizer(corpus_df)

if __name__ == '__main__':
    # Uncomment below line to train vectorizer on first run
    # initialize()
    app.run(debug=True)
