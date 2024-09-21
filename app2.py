import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset (question and answer)
dataset_path = "cleaned_output_questions (1).csv"
df = pd.read_csv(dataset_path)

# Preprocess and vectorize the question
def preprocess(text):
    return text.lower()

df['question'] = df['question'].apply(preprocess)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['question'])

# Function to get the best matching answer
def get_answer(user_question, threshold=0.3):  # Added a threshold argument
    user_question = preprocess(user_question)
    user_tfidf = vectorizer.transform([user_question])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Get the best match and its similarity score
    best_match_index = cosine_similarities.argmax()
    best_similarity_score = cosine_similarities[best_match_index]
    
    # Check if the similarity score is above the threshold
    if best_similarity_score >= threshold:
        return df['answer'].iloc[best_match_index]
    else:
        return "خارج المنهج"  # Return this if the question is out of scope

# Streamlit app setup
st.title('Question Answering Model')

st.write("""
This app allows you to ask a question, and it will return the most relevant answer.
""")

# Create a text input for the user's question
user_input = st.text_input("Type your question here:")

# When the user submits the question
if user_input:
    # Get the best answer using the TF-IDF model with a threshold
    answer = get_answer(user_input, threshold=0.3)  # You can adjust the threshold as needed
    
    # Display the user's question and the model's answer
    st.write(f"**You asked:** {user_input}")
    st.write(f"**Answer:** {answer}")
