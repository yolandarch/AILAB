import streamlit as st
import requests

def predict_kata(text) :
    respon = requests.post("http://127.0.0.1:8000/predict_sentiment", json={"text" : text})

    hasil_NB = respon.json()["predicted_sentiment_nb"]
    hasil_SVM = respon.json()["predicted_sentiment_svm"]
    akurasi_NB = respon.json()["accuracy_nb"]
    akurasi_SVM = respon.json()["accuracy_svm"]

    return hasil_NB, hasil_SVM, akurasi_NB, akurasi_SVM


st.title("test API NLP")
texttopredict = st.text_area("input text","Isi dengan kalimat bahasa inggris")

# Button to predict sentiment
if st.button("Predict Sentiment"):
    result_nb, result_svm, acuracy_nb, acuracy_svm = predict_kata(texttopredict)
        # Display the predictions
    st.write(f"Predicted Sentiment (Naive Bayes) - {acuracy_nb} : {result_nb}")
    st.write(f"Predicted Sentiment (Linear SVM) - {acuracy_svm} : {result_svm}")
