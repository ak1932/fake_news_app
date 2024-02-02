import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
import re

import string
import pickle

data_fake = pd.read_csv('Fake.csv')

data_true = pd.read_csv('True.csv')

print(data_fake.head())
print(data_true.head())

data_fake["class"] = 0
data_true['class'] = 1

print(f"{data_fake.shape=} {data_fake.shape=}")

# Manual testing
data_fake_manual_testing = data_fake.tail(10)
for i in range(len(data_fake) - 1,len(data_fake) - 10):
    data_fake.drop([i], axis = 0, inplace = True)

data_true_manual_testing = data_true.tail(10)
for i in range(len(data_true) - 1,len(data_true) - 10):
    data_true.drop([i], axis = 0, inplace = True)

data_fake_manual_testing.loc[:, ("class")] = 0
data_true_manual_testing.loc[:, ("class")] = 1

# Merging both fake and true
data_merge = pd.concat([data_fake, data_true], axis = 0)
data_merge.head(10)

data = data_merge.drop(['title', 'subject', 'date'], axis = 1)


# fn to clean text
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


data[ 'text'] = data['text'].apply(wordopt)

x = data['text']
y = data['class']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)


pred_lr = LR.predict(xv_test)
LR.score(xv_test, y_test)
print(classification_report(y_test, pred_lr))


from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


pred_dt = DT.predict(xv_test)
DT.score(xv_test, y_test)
print(classification_report(y_test, pred_dt))

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Real News"

def manual_testing(news):

    testing_news = {"text":[news]}

    new_def_test = pd.DataFrame(testing_news)

    new_def_test["test"] = new_def_test["test"].apply(wordopt)
    new_x_test = new_def_test["test"]
    new_xv_test = vectorization.transform(new_x_test)

    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)

    return print(f"\n\nLR Prediction: {output_lable(pred_LR[0])}\nDT Prediction: {output_lable(pred_DT[0])}")

manual_testing("Ayodhya Ram mandir is going good")


##save the model
#file = open("expense_model.pkl", 'wb')
#pickle.dump(model, file)



#import pandas as pd
#import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
#from sklearn.linear_model import LinearRegression
#import pickle
#data = pd.read_csv("./insurance.csv")
#le = LabelEncoder()
#le.fit(data['sex'])
#data['Sex'] = le.transform(data['sex'])
#le.fit(data['smoker'])
#data['Smoker'] = le.transform(data['smoker'])
#le.fit(data['region'])
#data['Region'] = le.transform(data['region'])
##independent and dependent columns
#x = data[["age", "bmi", "children", "Sex", "Smoker", "Region"]]
#y = data['charges']
##split in train and test
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
##model training
#linreg = LinearRegression()
#linreg.fit(x_train, y_train)
##model testing
#predictions = linreg.predict(x_test)
#linreg.score(x_test,y_test)
