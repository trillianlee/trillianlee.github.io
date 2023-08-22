import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# Sample medical records data
data = {
    "text": [
        "Patient has a fever and cough",
        "Patient is experiencing chest pain",
        "Patient reports headache and fatigue",
        "Patient has a cough and body aches",
        "Patient complains of nausea and dizziness",
    ],
    "class": ["Class A", "Class B", "Class A", "Class A", "Class B"],
}

df = pd.DataFrame(data)

# Model training
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['class']
model = MultinomialNB()
model.fit(X, y)

# Streamlit app
st.title("Medical Record Classifier")

# User input
input_text = st.text_area("Enter the medical record:", "")

if st.button("Classify"):
    if input_text:
        input_vector = vectorizer.transform([input_text])
        probabilities = model.predict_proba(input_vector)[0]

        st.write("Class Probabilities:")
        for idx, class_name in enumerate(model.classes_):
            st.write(f"{class_name}: {probabilities[idx]}")

        # Plot probabilities
        plt.bar(model.classes_, probabilities)
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.title("Class Probabilities")
        st.pyplot(plt)
