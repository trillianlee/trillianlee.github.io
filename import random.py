import random
import pandas as pd

def generate_medical_records(num_records=100):
    records = []
    for _ in range(num_records):
        record = {
            'age': random.randint(18, 80),
            'gender': random.choice(['Male', 'Female']),
            'symptom': random.choice(['Fever', 'Cough', 'Headache', 'Fatigue']),
            'diagnosis': random.choice(['Cold', 'Flu', 'Headache', 'COVID-19'])
        }
        records.append(record)
    return pd.DataFrame(records)

if __name__ == "__main__":
    medical_records = generate_medical_records()
    medical_records.to_csv("medical_records.csv", index=False)
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
