# import Flask class from the flask module
from flask import Flask, request, jsonify, render_template
import pandas as pd

import numpy as np
import pickle

user_final_rating = pd.read_pickle("models/user_final_rating.pkl")
df = pd.read_csv("models/df.csv")
word_vectorizer = pickle.load(
    open('models/word_vectorizer.pkl', 'rb'))
logit = pickle.load(
    open('models/logit_model.pkl', 'rb'))

def recommend(user_input):
    d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]

    # Based on positive sentiment percentage.
    i= 0
    a = {}
    for prod_name in d.index.tolist():
      product_name = prod_name
      product_name_review_list =df[df['prod_name']== product_name]['Review'].tolist()
      features= word_vectorizer.transform(product_name_review_list)
      logit.predict(features)
      a[product_name] = logit.predict(features).mean()*100
      
    b= pd.Series(a).sort_values(ascending = False).head(5).index.tolist()
    return b

# Create Flask object to run
app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    username = str(request.form.get('reviews_username'))
    print(username)
    prediction = recommend(username)
    print("Output :", prediction)
    return render_template('index.html', prediction_text='Your Top 5 Recommendations are:\n {}'.format(prediction))
    #return prediction[0]


if __name__ == "__main__":
    print("**Starting Server...")
    # Call function that loads Model
    print("**Model loaded...")
    # Run Server
    app.run(host="127.0.0.1", port=5000)
    #app.run(debug = True)

