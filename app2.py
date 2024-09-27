import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from serpapi import GoogleSearch
import requests
from bs4 import BeautifulSoup

# Load dataset
dataset_path = "C:\\Users\\Administrator\\Downloads\\cleaned_output_questions (1).csv"
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

# Function to scrape Wikipedia and get the first statement
def scrape_wikipedia(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the first paragraph in the Wikipedia article
        paragraphs = soup.find_all('p')
        for paragraph in paragraphs:
            # Clean the paragraph text and find the first full statement ending with '.'
            paragraph_text = paragraph.get_text().strip()
            sentences = paragraph_text.split('.')
            if sentences:
                first_sentence = sentences[0].strip() + '.'
                return first_sentence
        return "المعذرة، لا توجد إجابة متاحة في الوقت الحالي."
    
    except Exception as e:
        return "حدث خطأ أثناء الوصول إلى صفحة ويكيبيديا."

# Function to search Google and get Wikipedia result
def search_google(query):
    search_params = {
        "q": query + " site:wikipedia.org",  # Search specifically in Wikipedia
        "api_key": SERP_API_KEY
    }
    
    search = GoogleSearch(search_params)
    results = search.get_dict()
    
    # Extract the first Wikipedia result URL
    for result in results.get('organic_results', []):
        if 'wikipedia.org' in result.get('link', ''):
            wikipedia_url = result.get('link')
            return scrape_wikipedia(wikipedia_url)
    
    # Return fallback message if no Wikipedia result is found
    return "المعذرة، لا توجد إجابة متاحة في الوقت الحالي."

# Streamlit app layout
st.title("Arabic Question Answering System")

# Input from the user
user_question = st.text_input("اكتب سؤالك هنا:")

# Button to submit the question
if st.button("إرسال"):
    if user_question:
        answer = get_answer(user_question)
        st.write(f"السؤال: {user_question}")
        st.write(f"الإجابة: {answer}")
    else:
        st.write("يرجى إدخال سؤال")
