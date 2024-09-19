import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset (question and answer)
dataset_path = "C:\\Users\\Administrator\\Downloads\\cleaned_output_questions (1).csv"
df = pd.read_csv(dataset_path)

# Preprocess and vectorize the question
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
    best_match_index = cosine_similarities.argmax()
    return df['answer'].iloc[best_match_index]

# Streamlit app setup
st.title('Question Answering Model')

st.write("""
This app allows you to ask a question, and it will return the most relevant answer.
""")

# Create a text input for the user's question
user_input = st.text_input("Type your question here:")

# When the user submits the question
if user_input:
    # Get the best answer using the TF-IDF model
    answer = get_answer(user_input)
    
    # Display the user's question and the model's answer
    st.write(f"**You asked:** {user_input}")
    st.write(f"**Answer:** {answer}")
