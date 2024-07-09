'''
#AspireNX Internship Task 2
#Task: Movie Genre Classifier
#Category: Machine Learning
#Author: Navansh Krishna Goswami
#Date: 9th July 2024
#Github: https://github.com/navansh03 
#Linkedin:https://www.linkedin.com/in/navansh-krishna-goswami-341713248/
# '''


#importing the necessary libraries
import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


#load the model and vectorizer
with open('movie_genre_classifier_NB.pkl', 'rb') as file:
    model_NB = pickle.load(file)
with open('movie_genre_classifier_LR.pkl', 'rb') as file:
    model_LR = pickle.load(file)
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)


# Streamlit Interface
st.title('Movie Genre Prediction')
plot_summary = st.text_area("Enter the movie plot summary or a gist:")


# Final the Genre by both the models
if st.button('Predict Genre'):
    plot_summary_vec = vectorizer.transform([plot_summary])
    genre_prediction1 = model_NB.predict(plot_summary_vec)
    genre_prediction2 = model_LR.predict(plot_summary_vec)
    st.markdown(f'Predicted Genre With Naive Bayes Model: **{genre_prediction1[0]}**')
    st.markdown(f'Predicted Genre With Logistic Regression: **{genre_prediction2[0]}**')
    