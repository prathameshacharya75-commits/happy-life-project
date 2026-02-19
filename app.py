import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Happy Life", page_icon="🌿")

st.title("🌿 Happy Life – Suicide Risk Detection")

# -----------------------
# Train Model Function
# -----------------------
@st.cache_resource
def train_model():

    # Load dataset
    data = pd.read_csv("suicide_dataset.csv")

    # If dataset loaded as single column, split it
    if len(data.columns) == 1:
        data[['text','label']] = data.iloc[:,0].str.split(',', expand=True)
        data = data.drop(columns=[data.columns[0]])

    # Clean column names
    data.columns = data.columns.str.strip()

    # Convert labels to numeric
    data['label'] = data['label'].map({
        'suicide': 1,
        'non-suicide': 0
    })

    # Remove missing values
    data = data.dropna()

    X = data['text']
    y = data['label']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, vectorizer, accuracy


model, vectorizer, accuracy = train_model()

st.write(f"Model Accuracy: {accuracy:.2f}")

# -----------------------
# User Input
# -----------------------

user_input = st.text_area("Enter your thoughts:")

if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        text_vector = vectorizer.transform([user_input])
        prediction = model.predict(text_vector)[0]

        if prediction == 1:
            st.error("⚠️ High Risk Detected")
            st.write("You are not alone. Please talk to someone you trust.")
        else:
            st.success("✅ Emotionally Safe")
            st.write("Keep staying positive 🌟")
