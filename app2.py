import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from serpapi import GoogleSearch

# Load dataset
dataset_path = "cleaned_output_questions (1).csv"
df = pd.read_csv(dataset_path)

# Preprocess and vectorize question
def preprocess(text):
    return text.lower()

df['question'] = df['question'].apply(preprocess)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['question'])

# SerpAPI settings
SERP_API_KEY = "3f2546369d64be3f711f4d6f1b52afbf002d4b0b42095838402413f0b135fadb"

# Function to get the best matching answer
def get_answer(user_question):
    user_question = preprocess(user_question)
    user_tfidf = vectorizer.transform([user_question])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    best_match_index = np.argmax(cosine_similarities)
    
    # Set a threshold for similarity
    similarity_threshold = 0.8
    if cosine_similarities[best_match_index] < similarity_threshold:
        # If below threshold, search on Google
        return search_google(user_question)
    else:
        # Return the best-matching answer from the dataset
        return df['answer'].iloc[best_match_index]

# Function to search Google and get Wikipedia result
def search_google(query):
    search_params = {
        "q": query + " site:wikipedia.org",  # Search specifically in Wikipedia
        "api_key": SERP_API_KEY
    }
    
    search = GoogleSearch(search_params)
    results = search.get_dict()
    
    # Extract the first Wikipedia result
    for result in results.get('organic_results', []):
        if 'wikipedia.org' in result.get('link', ''):
            return result.get('snippet', 'المعذرة، لا توجد إجابة متاحة في الوقت الحالي.')
    
    # Return fallback message if no Wikipedia result is found
    return "المعذرة، لا توجد إجابة متاحة في الوقت الحالي."

# Streamlit app
st.title("Question Answering App")

# Input box for user question
user_question = st.text_input("Ask a question:")

if st.button("Submit"):
    if user_question:
        # Get the answer for the user question
        answer = get_answer(user_question)
        
        # Display the question and answer
        st.write(f"**You asked:** {user_question}")
        st.write(f"**Answer:** {answer}")
