import streamlit as st
from airTravelSentimentAnalysis.pipeline.prediction import PredictionPipeline


# Streamlit UI
st.title("✈️ Air Travel Intent Prediction")
text_input = st.text_area("Enter a flight-related review:")


if st.button("Get Intent"):
    if text_input.strip():
        result = PredictionPipeline(text_input).predict()
        print("My result \n", result)  # Debugging line to check the result structure
        st.markdown(f"**Predicted Intent:** {result['predicted_label']}")
        st.markdown(f"**Confidence:** {result['confidence_score']:.2%}")
    else:
        st.warning("Please enter some text.")
