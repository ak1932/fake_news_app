from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle 
import pandas as pd
import re
import string
from newsapi import NewsApiClient

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb')) #read mode

newsapi = NewsApiClient(api_key='7fc63da2eec0495ba504458b4f5d3200')

@app.route("/")
def home():
    return render_template('index.html')



def wordopt (text):
    text = text.lower()
    text = re.sub("\[.*?\]", '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', "", text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub("[%s]" % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub("\w*\d\w*", '', text)
    return text

def output_lable(n):
    if n == 0:
        return "fake"
    elif n == 1:
        return "real"

def manual_testing(news, model) -> bool:

    testing_news = {"text":[news]}

    new_def_test = pd.DataFrame(testing_news)
    print(new_def_test.head())

    new_def_test["text"] = new_def_test["text"].apply(wordopt)

    vectorization = pickle.load(open('tokenizer.pkl','rb')) #read mode
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    pred_DT = model.predict(new_xv_test)[0]

    return True if pred_DT==1 else False

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        topic = request.form["topic"]
        news = request.form["news"]


        # /v2/top-headlines
        top_headlines = newsapi.get_everything(q=f'${topic}',
                                                  language='en')
        prediction_text = top_headlines


        # if(news==""):
        #     prediction_text=""
        # else:
        #     prediction = manual_testing(news, model)
        #     if(prediction):
        #         prediction_text = 'Fake News'
        #     else:
        #         prediction_text = 'Fake News'

        return render_template("index.html", prediction_text=prediction_text)



if __name__ == "__main__":
    app.run(debug=True)
