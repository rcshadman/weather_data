
# coding: utf-8

import numpy as np
import pandas as pd
from pprint import pprint as pp
import json, os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import random
random.seed(0)

df = pd.read_csv('weather_data/input_data_nov.csv')
df = df.iloc[:,1:-2]
df = df.dropna(axis=0,how='any')
df[['Day','Hour','Humidity']] = df[['Day','Hour','Humidity']].astype(int)
df[['Temp','DewPt','Pressure']] = df[['Temp','DewPt','Pressure']].astype(float)

df.insert(1, 'Position', '')
df.rename(columns={'City': 'Location','Day':'Date'}, inplace=True)

geo = { 
'Sydney': [-33.86,151.20,19],
'Melbourne': [-37.66,144.84,124],
'Brisbane': [-27.47,153.02,28],
'Gold_Coast': [-28.01,153.42,3.9],
'Adelaide': [-34.92,138.59,44.7],
'Darwin': [-12.46,130.84,37],
'Wollongong': [-34.42,150.89,19],
'Canberra': [-35.28,149.12,576.7],
'Newcastle': [-32.92,151.77,12.81],
'Hobart': [-42.83,147.50,6]
}


df['Position'] = [geo[city]  for city in df['Location']]
weather_conditions= { 
    'Rain' : ['Light Rain',
              'Light Rain Showers',
              'Light Drizzle',
              'Light Thunderstorms and Rain',
              'Heavy Rain Showers',
              'Unknown Precipitation',
              'Thunderstorms and Rain',
              'Rain',
              'Rain Showers',
              'Thunderstorm',
              'Heavy Thunderstorms and Rain',
              'Heavy Rain',
              'Drizzle',
              'Heavy Drizzle'],
    'Cloudy':['Mostly Cloudy',
              'Partly Cloudy',
              'Overcast',
              'Scattered Clouds'
             ],
    'Clear':['Clear'],
    'Snow':['Snow']
}

def simplify(cond):
    if cond in weather_conditions['Rain']:
        cond = 'Rain'
    elif cond in weather_conditions['Clear']:
        cond = 'Clear'
    elif cond in weather_conditions['Snow']:
        cond = 'Snow'
    elif cond in weather_conditions['Cloudy']:
        cond = 'Cloudy'
    else:
        cond = 'Unknown'
    return cond

df['Condition'] = df['Condition'].apply(simplify)
df = df[df['Condition'] != 'Unknown']



df = df[['Date','Hour','Location','Position','Condition','Pressure','Temp','Humidity','DewPt']]
df = df.sort_values(['Location','Date','Hour'])

tmp = pd.DataFrame({'Time':['{:02}:{:02}:{:02}'.format(hr,random.randint(0,60),random.randint(0,60)) for hr in df['Hour'] ],
                        'Date':['2017-11-{:02}'.format(day) for day in df['Date']]
                        })


df.insert(4,'LocalTime','')
df['LocalTime'] = tmp['Date'] + ' ' + tmp['Time']

data = { }
for city in df['Location'].unique():
    data[city] = None
    data[city] = df[ df['Location'] == city ]


trainingData = df.copy()

# To feed sklearn I have to convert the 'conditions' text data to a numeric value
labels = {}
for label, condition in enumerate(trainingData['Condition'].unique()):
    labels[condition] = label
trainingData['label'] = trainingData['Condition'].apply(lambda c: labels.get(c))

# trainingY = pd.get_dummies(trainingData['Condition'], 'Condition')
trainingY = trainingData['label']

trainingData.drop(['Location','Position','LocalTime','Condition','DewPt','label'], axis=1, inplace=True)


trainingData.head()


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


# Split my data into train and test to avoid overfiting
X_train, X_test, Y_train, Y_test = train_test_split(trainingData, trainingY)


#  I will train a Support Vector Machine classifier
#     note: I tried with a Logistic Regression but I only got 68% accuracy

# classifier = SVC()
# classifier = SVC(kernel='rbf', verbose=True)
classifier = SVC(kernel='poly',degree=2,verbose=True)
# classifier = LogisticRegression(C=1e5)
# classifier = KNeighborsClassifier()

 
classifier.fit(X=X_train, y=Y_train)

# Now I'll check the accuracy of my model
train_ac = classifier.score(X=X_train, y=Y_train)
test_ac = classifier.score(X=X_test, y=Y_test)

print('Training accuracy: {}'.format(train_ac))
print('Testing accuracy: {}'.format(test_ac))

