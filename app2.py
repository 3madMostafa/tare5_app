import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
dataset_path = "cleaned_output_questions (1).csv"
df = pd.read_csv(dataset_path)

# Preprocess and vectorize question
def preprocess(text):
    return text.lower()

df['question'] = df['question'].apply(preprocess)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['question'])

# Function to get the best matching answer
def get_answer(user_question):
    user_question = preprocess(user_question)
    user_tfidf = vectorizer.transform([user_question])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    best_match_index = np.argmax(cosine_similarities)
    
    # Set a threshold for similarity
    similarity_threshold = 0.8
    if cosine_similarities[best_match_index] < similarity_threshold:
        # If below threshold, return fallback message
        return "المعذرة، لا توجد إجابة متاحة في الوقت الحالي."
    else:
        # Return the best-matching answer from the dataset
        return df['answer'].iloc[best_match_index]

# Streamlit UI setup
st.title("Question Answer App")
user_question = st.text_input("Ask a question:")

if st.button("Submit"):
    if user_question:
        answer = get_answer(user_question)
        st.write(f"**You asked:** {user_question}")
        st.write(f"**Answer:** {answer}")
