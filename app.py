import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    # Load medical records
    medical_records = pd.read_csv("medical_records.csv")

    # Preprocessing the data
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(medical_records[['age', 'gender', 'symptom']].to_dict(orient='records'))
    y = medical_records['diagnosis']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Making predictions
    y_pred = clf.predict(X_test)

    # Calculating accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Display the probabilities as a bar chart
def plot_probabilities(probabilities):
    labels = list(probabilities.keys())
    probs = list(probabilities.values())

    fig, ax = plt.subplots()
    ax.barh(labels, probs)
    ax.set_xlabel('Probability')
    ax.set_title('Naive Bayes Classification Probabilities')
    st.pyplot(fig)

# Main Streamlit app
def main():
    st.title("Medical Records Classification")

    # Load generated medical records
    medical_records = pd.read_csv("medical_records.csv")

    # Display the generated medical records
    st.subheader("Generated Medical Records")
    st.dataframe(medical_records)

    # Get a sample record for classification
    sample_record = medical_records.sample(1)

    # Display the sample record
    st.subheader("Sample Record for Classification")
    st.write(sample_record)

    # Perform classification using the trained classifier
    sample_X = vectorizer.transform(sample_record[['age', 'gender', 'symptom']].to_dict(orient='records'))
    probabilities = dict(zip(clf.classes_, np.exp(clf.predict_log_proba(sample_X))[0]))

    # Display classification probabilities
    st.subheader("Classification Probabilities")
    st.write(probabilities)

    # Plot the probabilities as a bar chart
    st.subheader("Classification Probability Chart")
    plot_probabilities(probabilities)

if __name__ == '__main__':
    main()
