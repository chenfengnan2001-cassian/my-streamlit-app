
import streamlit as st
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

@st.cache_resource
def load_classifier():
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        low_cpu_mem_usage=False
    )

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1
    )

    return classifier


st.title("Text Classification for you")
st.write("Enter your text below to classify its sentiment.")

text = st.text_area("Enter the text to classify", "")

if st.button("Classify"):
    if text.strip():
        with st.spinner("Classifying..."):
            classifier = load_classifier()
            result = classifier(text)[0]

        st.write("Text:", text)
        st.write("Label:", result["label"])
        st.write("Score:", result["score"])
    else:
        st.warning("Please enter some text first.")
