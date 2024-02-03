# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Pb8x0km_r5ELyDGdZtCuB3UM_g0oIXig
"""

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
print(f"{data_fake.shape=} {data_fake.shape=}")

data_fake.loc[:, "class"]=0
data_true.loc[:, "class"]=1

# Merging both fake and true
data_merge = pd.concat([data_fake, data_true], axis = 0)
data_merge.head(10)

data = data_merge.drop(['title', 'subject', 'date'], axis = 1)

print(data.head())
data.tail()

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

data.head()

x = data['text']
y = data['class']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Logistic Regression

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)
LR.score(xv_test, y_test)
print(classification_report(y_test, pred_lr))

# DecisionTreeClassifier

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
    print(new_def_test.head())

    new_def_test["text"] = new_def_test["text"].apply(wordopt)

    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)

    return print(f"\n\nLR Prediction: {output_lable(pred_LR[0])}\nDT Prediction: {output_lable(pred_DT[0])}")


manual_testing("WASHINGTON (Reuters) - A federal appeals court in Washington on Friday accepted a bid by President Donald Trump’s administration to prevent the U.S. military from accepting transgender recruits starting Jan. 1, the second court to issue such a ruling this week. Four federal judges around the country have issued injunctions blocking Trump’s ban on transgender people from the military, including one that was also handed down on Friday. The administration has appealed the previous three rulings. In a six-page order, the three-judge-panel of the U.S. Court of Appeals for the District of Columbia Circuit said the administration had “not shown a strong likelihood that they will succeed on the merits of their challenge” to a district court’s order blocking the ban. On Thursday the Richmond, Virginia-based 4th U.S. Circuit Court of Appeals said it was denying the administration’s request while the appeal proceeds. The two courts’ actions could prompt the administration to ask the conservative-majority U.S. Supreme Court to intervene. Also on Friday, a federal trial court in Riverside, California, blocked the ban while the case proceeds, making it the fourth to do so, after similar rulings in Baltimore, Seattle and Washington, D.C. U.S. District Judge Jesus Bernal said without the injunction the plaintiffs, including current and aspiring service members, would suffer irreparable harm. “There is nothing any court can do to remedy a government-sent message that some citizens are not worthy of the military uniform simply because of their gender,” he added. The administration had argued that the Jan. 1 deadline for accepting transgender recruits was problematic because tens of thousands of personnel would have to be trained on the medical standards needed to process transgender applicants, and the military was not ready for that. The Obama administration had set a deadline of July 1, 2017, to begin accepting transgender recruits, but Trump’s defense secretary, James Mattis, postponed that date to Jan. 1. In an August memorandum, Trump gave the military until March 2018 to revert to a policy prohibiting openly transgender individuals from joining the military and authorizing their discharge. The memo also halted the use of government funds for sex-reassignment surgery for active-duty personnel.")

#save the model
file = open("../model.pkl", 'wb')
pickle.dump(DT, file)