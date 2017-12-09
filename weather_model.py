# #!/usr/bin/env python3
# coding: utf-8


import numpy as np
import pandas as pd
from pprint import pprint as pp
import json, os, sys, random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import datetime
# import time

random.seed(0)

GEOGRAPHICAL_DATA = { 
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

##cleanup and format data
def prepare_data(df):
  
  # remove last two features with so many nulls
  df = df.iloc[:,1:-2]

  # drop rows with nulls
  df = df.dropna(axis=0,how='any')

  # change the datatype
  df[['Day','Hour','Humidity']] = df[['Day','Hour','Humidity']].astype(int)
  df[['Temp','DewPt','Pressure']] = df[['Temp','DewPt','Pressure']].astype(float)

  # rename column
  df.rename(columns={'City': 'Location','Day':'Date'}, inplace=True)

  # insert position columns for cities
  df.insert(1, 'Position', '')


  # populate data in the position coloumn 
  df['Position'] = [ GEOGRAPHICAL_DATA[city]  for city in df['Location']]

  # simplify the weathers to rain, cloudy, and clear

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

  # rearrange the columns
  df = df[['Date','Hour','Location','Position','Condition','Temp','Pressure','Humidity','DewPt']]

  # sort by location,data and hour
  df = df.sort_values(['Location','Date','Hour'])

  # generate LocalTime Column from day and hour
  tmp = pd.DataFrame(
      {
          'Time':['{:02}:{:02}:{:02}'.format(hr,random.randint(0,60),random.randint(0,60)) for hr in df['Hour'] ],
          'Date':['2017-11-{:02}'.format(day) for day in df['Date']]
      })

  df.insert(4,'LocalTime','')
  df['LocalTime'] = tmp['Date'] + ' ' + tmp['Time']


  # To feed sklearn I have to convert the 'conditions' text data to a numeric value
  labels = {}
  for label, condition in enumerate( df['Condition'].unique() ):
      labels[condition] = label

  df['label'] = df['Condition'].apply(lambda c: labels.get(c))

  # handy map to refer back to the condition
  label_map = { v:k for k,v in labels.items() }
  return  df,label_map


# Learning algorithm 
def learn(trainingData, features_to_drop):


    # trainingY = pd.get_dummies(trainingData['Condition'], 'Condition')
    trainingY = trainingData['label']

    trainingData.drop(features_to_drop, axis=1, inplace=True)

    # Split my data into train and test to avoid overfiting
    X_train, X_test, Y_train, Y_test = train_test_split(trainingData, trainingY)

    #  I will train a Support Vector Machine classifier
    #  note: I tried with a Logistic Regression but I only got 68% accuracy

    # classifier = SVC()
    # classifier = SVC(kernel='rbf', verbose=True)
    classifier = SVC(kernel='poly',degree=2)
    # classifier = LogisticRegression(C=1e5)
    # classifier = KNeighborsClassifier()


    classifier.fit(X=X_train, y=Y_train)

    # Now I'll check the accuracy of my model
    train_ac = classifier.score(X=X_train, y=Y_train)
    test_ac = classifier.score(X=X_test, y=Y_test)
    # print('Training accuracy: {} '.format(train_ac))
    # print('Testing accuracy: {}'.format(test_ac))
    # print('-----------------            *****            --------------------')

    return classifier

def create_predictors(dff):
  # Drop unnecessary features
  features_to_drop = ['Location','Position','LocalTime','Condition','DewPt','label']

  # create a predictor for each city and maintain in a dict to use it later for generating data
  city_weather_predictor = { }
  # store the covariance of relevent features to generate closely related random points
  city_stats_meta = {}

  for city in dff['Location'].unique():
      trainingData = dff[ dff['Location'] == city ].copy()
      
      # print('City: {} '.format(city))
      classifier = learn(trainingData,features_to_drop)
      city_weather_predictor.update({city:classifier})
      
      # generate covariance n mean for each city n store it city_stats_meta
      statistics = trainingData.describe()
      mean = [
        statistics['Temp']['mean'],
        statistics['Pressure']['mean'],
        statistics['Humidity']['mean']
        ]
      cov = statistics[['Temp','Pressure','Humidity']].cov()

      city_stats_meta.update({city:[mean,cov]})
  return city_weather_predictor ,city_stats_meta

def generate_atmostphere(**kwarg):
  predictor = kwarg['predictor']
  city_stats_meta = kwarg['city_stats']
  weather_lookup = kwarg['weather_lookup']

  weather_generator = np.random.multivariate_normal
  city_list = [each_city for each_city in GEOGRAPHICAL_DATA.keys()]
  city = np.random.choice(city_list)
  generated_values = weather_generator(*city_stats_meta[city]).tolist()
  # print(generated_values)
  # adding day and time to generated weather data, to predict the condition
  day,hour = '{:%d %H}'.format(datetime.datetime.now()).split(' ')
  day,hour = int(day),int(hour)
  generated_values = [day,hour]+generated_values

  predicted_value = predictor[city].predict([generated_values])[0]
  
  Condition = weather_lookup[predicted_value]
  Temp = generated_values[2]
  Pressure = generated_values[3]
  Humidity = generated_values[4]
  # LocalTime = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
  Localtime= '{:%Y-%m-%dT%H:%M:%SZ}'.format(datetime.datetime.now())
  Position = GEOGRAPHICAL_DATA[city]

  output_dict = {
    'city':city,
    'lat':Position[0],
    'long':Position[1],
    'elevation':Position[2],
    'localtime':Localtime,
    'condition':Condition,
    'temp':Temp,
    'pressure':Pressure,
    'humidity':Humidity
  }
  output_string = '{city}|{lat},{long},{elevation}|{localtime}|{condition}|+{temp:.01f}|{pressure:.01f}|{humidity:.0f}'.format(**output_dict)
  print(output_string)
  # import ipdb; ipdb.set_trace()
def main(argv):
  
  if 'test' in argv:
    import ipdb; ipdb.set_trace()
  else:
      historical_data = pd.read_csv('training_data/input_data_nov.csv')
      clean_data, label_map = prepare_data(historical_data)
      predictor, city_stats_meta = create_predictors(clean_data)
      
      while(1):
        weather = generate_atmostphere( predictor=predictor,
                                   city_stats=city_stats_meta,
                                   weather_lookup=label_map)
        

      import ipdb; ipdb.set_trace()
      # predicted_value = city_predictor['Adelaide'].predict([[0,0,0,0,0]])[0]
      # label_map[predicted_value]


if __name__ == "__main__":
  main(sys.argv)