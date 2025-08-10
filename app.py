from flask import Flask, request, render_template
import pickle
import gensim
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
from nltk.tokenize import word_tokenize
from gensim.utils import simple_preprocess
import pandas as pd
import nltk
nltk.download('wordnet')

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('classifier.pkl', 'rb'))
wv = gensim.models.KeyedVectors.load('word2vec.model')


def preprocess(text):
    
    corpus=[]
    message=re.sub('[^a-zA-Z]',' ',text)
    message=message.lower()
    message=message.split()
    review=[lemmatizer.lemmatize(word,pos='v') for word in message]
    review=' '.join(review)
    corpus.append(review)
    
    print(corpus)
    words = []
    for sent in corpus:  # each text can have multiple sentences
        words.append(simple_preprocess(sent))
        
        
    
    print(words)
    
    X = []
    for i in range(len(words)):
        # Get the list of word vectors for words in words[i]
        vecs = [wv[word] for word in words[i] if word in wv.index_to_key]

        if vecs:  # Only compute mean if list is not empty
            X.append(np.mean(vecs, axis=0))
        else:
            X.append(np.zeros(wv.vector_size))
        X_df=pd.DataFrame(X)
        X_df.dropna(inplace=True)
        return np.array(X_df)


@app.route('/',methods=['GET','POST'])
def predict_spam():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        text=request.form['message']
        input_test=preprocess(text)
        prediction = model.predict(input_test)[0]
        result="Spam" if prediction == 1 else "Ham"
        
        return render_template('home.html',result=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8000,debug=True)

