
import streamlit as st

# Import model components directly
try:
    import transformers
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
    from transformers.pipelines import pipeline

except Exception as e:
    st.error("Failed to import Transformers components.")
    st.exception(e)
    st.stop()


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
        with st.spinner("Loading model and classifying..."):
            classifier = load_classifier()
            result = classifier(text)[0]

        st.write("Text:", text)
        st.write("Label:", result["label"])
        st.write("Score:", result["score"])
    else:
        st.warning("Please enter some text first.")
