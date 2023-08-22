import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import streamlit as st

# Simulated data generation (replace this with real data)
np.random.seed(42)
num_records = 100
data = {
    'Age': np.random.randint(18, 80, num_records),
    'BloodPressure': np.random.randint(80, 180, num_records),
    'Cholesterol': np.random.choice(['High', 'Medium', 'Low'], num_records),
    'Diabetes': np.random.choice(['Yes', 'No'], num_records)
}
df = pd.DataFrame(data)

# Convert categorical variables to numerical
df['Cholesterol'] = df['Cholesterol'].map({'High': 0, 'Medium': 1, 'Low': 2})
df['Diabetes'] = df['Diabetes'].map({'Yes': 1, 'No': 0})

# Split the data
X = df.drop('Diabetes', axis=1)
y = df['Diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict probabilities
probs = nb_classifier.predict_proba(X_test)

# Streamlit app
st.title("Naive Bayes Medical Record Classification")
st.write("Predicted Probabilities:")

# Plot the bar chart
fig, ax = plt.subplots()
labels = ['No Diabetes', 'Diabetes']
x = np.arange(len(labels))
width = 0.35

for i, label in enumerate(labels):
    ax.bar(x + i * width, probs[:, i], width, label=label)

ax.set_ylabel('Probability')
ax.set_title('Naive Bayes Classification Probabilities')
ax.set_xticks(x + width / 2)
ax.set_xticklabels(X_test.index, rotation=45)
ax.legend()

for i, v in enumerate(probs):
    ax.text(i * width - 0.15, v[0] + 0.02, f'{v[0]:.2f}', color='black')
    ax.text(i * width + 0.15, v[1] + 0.02, f'{v[1]:.2f}', color='black')  # Display the value for the second class

st.pyplot(fig)

