import streamlit as st
from transformers import pipeline

# Load the text classification model
classifier = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    return_all_scores=True
)

# Streamlit application
st.title("Text Classification for you")
st.write("Enter your text below to classify its sentiment.")

text = st.text_area("Enter the text to classify", "")

if st.button("Classify"):
    if text.strip():
        results = classifier(text)[0]

        max_result = max(results, key=lambda x: x["score"])

        st.write("Text:", text)
        st.write("Label:", max_result["label"])
        st.write("Score:", max_result["score"])
    else:
        st.warning("Please enter some text first.")
