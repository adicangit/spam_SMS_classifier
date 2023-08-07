import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

ps = PorterStemmer()
stopwords.words('english')
import string


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS spam classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    #1. preprocess
    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)

        y = []
        for i in text:
            if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
                y.append(ps.stem(i))

        return " ".join(y)

    transformed_sms = transform_text(input_sms)
    #2. vectorise
    vector_input = tfidf.transform([transformed_sms])
    #3. predict
    result = model.predict(vector_input)[0]
    #4. display
    if result == 1 :
        st.header("Spam")
    else :
        st.header("Not Spam")




