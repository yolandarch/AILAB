from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
app = FastAPI()

# Load data
column_names = ['id', 'Konteks', 'Response', 'Komentar']
df = pd.read_csv("twitter_training.csv", header=None, names=column_names)
df = df.dropna(axis=0)
df_validation = pd.read_csv("twitter_validation.csv", header=None, names=column_names)
df_validation = df_validation.dropna(axis=0)

# Preprocessing Teks
import nltk
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text)
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(filtered_words)

df['clean_Komentar'] = df['Komentar'].apply(preprocess_text)
df_validation['clean_Komentar'] = df_validation['Komentar'].apply(preprocess_text)

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
train_df = df
val_df = df_validation

# TF-IDF Vectorization for Naive Bayes
tfidf_vectorizer_nb = TfidfVectorizer(max_features=5000)
X_train_tfidf_nb = tfidf_vectorizer_nb.fit_transform(df['clean_Komentar'])
y_train_nb = df['Response']

# TF-IDF Vectorization for Naive Bayes
tfidf_vectorizer_nb = TfidfVectorizer(max_features=5000)
X_train_tfidf_nb = tfidf_vectorizer_nb.fit_transform(train_df['clean_Komentar'])
y_train_nb = train_df['Response']
X_val_tfidf_nb = tfidf_vectorizer_nb.transform(val_df['clean_Komentar'])
y_val_nb = val_df['Response']

# Model Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf_nb, y_train_nb)

# TF-IDF Vectorization for Linear SVM
tfidf_vectorizer_svm = TfidfVectorizer(max_features=5000)
X_train_tfidf_svm = tfidf_vectorizer_svm.fit_transform(train_df['clean_Komentar'])
y_train_svm = train_df['Response']
X_val_tfidf_svm = tfidf_vectorizer_svm.transform(val_df['clean_Komentar'])
y_val_svm = val_df['Response']

# Model Linear SVM
svm_model = LinearSVC()
svm_model.fit(X_train_tfidf_svm, y_train_svm)

class InputText(BaseModel):
    text: str

@app.post("/predict_sentiment")
def predict_sentiment(input_text: InputText):
    cleaned_text = preprocess_text(input_text.text)
    vectorized_text_nb = tfidf_vectorizer_nb.transform([cleaned_text])
    vectorized_text_svm = tfidf_vectorizer_svm.transform([cleaned_text])

    prediction_nb = nb_model.predict(vectorized_text_nb)[0]
    prediction_svm = svm_model.predict(vectorized_text_svm)[0]

    # Evaluate accuracy on validation set
    accuracy_nb = accuracy_score(y_val_nb, nb_model.predict(X_val_tfidf_nb))
    accuracy_svm = accuracy_score(y_val_svm, svm_model.predict(X_val_tfidf_svm))

    return {
        "predicted_sentiment_nb": prediction_nb,
        "predicted_sentiment_svm": prediction_svm,
        "accuracy_nb": accuracy_nb,
        "accuracy_svm": accuracy_svm
    }