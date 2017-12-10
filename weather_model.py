# #!/usr/bin/env python3
# coding: utf-8

# weather_mode.py

"""
 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; 

"""
__author__ = "Md Shadman Alam"
__email__ = "rcshadman@gmail.com"


import __future__
import numpy as np
import pandas as pd
from pprint import pprint as pp
import json
import os
import sys
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from time import sleep
import datetime
import warnings
warnings.filterwarnings("ignore")

# Set random seed to 0
random.seed(0)

# Look for Geographical data in order [ latitude , longitude and elevation ]
GEOGRAPHICAL_DATA = {
    'Sydney': [-33.86, 151.20, 19],
    'Melbourne': [-37.66, 144.84, 124],
    'Brisbane': [-27.47, 153.02, 28],
    'Gold_Coast': [-28.01, 153.42, 3.9],
    'Adelaide': [-34.92, 138.59, 44.7],
    'Darwin': [-12.46, 130.84, 37],
    'Wollongong': [-34.42, 150.89, 19],
    'Canberra': [-35.28, 149.12, 576.7],
    'Newcastle': [-32.92, 151.77, 12.81],
    'Hobart': [-42.83, 147.50, 6]
}


def prepare_data(df):
    """
    This function is used to clean and format the data in to pandas readable format
    """

    try:
        # remove last two features with so many nulls
        df = df.iloc[:, 1:-2]

        # drop rows with nulls
        df = df.dropna(axis=0, how='any')

        # change the datatype
        df[['Day', 'Hour', 'Humidity']] = df[
            ['Day', 'Hour', 'Humidity']].astype(int)
        df[['Temp', 'DewPt', 'Pressure']] = df[
            ['Temp', 'DewPt', 'Pressure']].astype(float)

        # rename column
        df.rename(columns={'City': 'Location', 'Day': 'Date'}, inplace=True)
        # insert position columns for cities
        df.insert(1, 'Position', '')
        # populate data in the position coloumn
        df['Position'] = [GEOGRAPHICAL_DATA[city] for city in df['Location']]

        # simplify the weathers to rain, cloudy, and clear
        weather_conditions = {
            'Rain': ['Light Rain',
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
            'Cloudy': ['Mostly Cloudy',
                       'Partly Cloudy',
                       'Overcast',
                       'Scattered Clouds'
                       ],
            'Clear': ['Clear'],
            'Snow': ['Snow']
        }

        def simplify(cond):
            """
            helper function
            """
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
        df = df[['Date', 'Hour', 'Location', 'Position',
                 'Condition', 'Temp', 'Pressure', 'Humidity', 'DewPt']]

        # sort by location,data and hour
        df = df.sort_values(['Location', 'Date', 'Hour'])

        # generate LocalTime Column from day and hour
        tmp = pd.DataFrame(
            {
                'Time': ['{:02}:{:02}:{:02}'.format(hr, random.randint(0, 60), random.randint(0, 60)) for hr in df['Hour']],
                'Date': ['2017-11-{:02}'.format(day) for day in df['Date']]
            })

        df.insert(4, 'LocalTime', '')
        df['LocalTime'] = tmp['Date'] + ' ' + tmp['Time']

        # To feed sklearn I have to convert the 'conditions' text data to a
        # numeric value
        labels = {}
        for label, condition in enumerate(df['Condition'].unique()):
            labels[condition] = label
        
        df['label'] = df['Condition'].apply(lambda c: labels.get(c))

        # handy map to refer back to the condition
        label_map = {v: k for k, v in labels.items()}

    except Exception as e:
        print(e)

    return df, label_map


def learn(trainingData, features_to_drop):
    """
    Learning algorithm 
    I will train a Support Vector Machine classifier
    note:  with  Logistic Regression , only got 68% accuracy
    """

    trainingY = trainingData['label']
    trainingData.drop(features_to_drop, axis=1, inplace=True)

    # Split my data into train and test to avoid overfiting
    X_train, X_test, Y_train, Y_test = train_test_split(trainingData, trainingY)

    # classifier = SVC()
    classifier = SVC(kernel='poly', degree=2)
    # classifier = LogisticRegression(C=1e5)
    # classifier = KNeighborsClassifier()

    classifier.fit(X=X_train, y=Y_train)

    # check the accuracy of my model
    # train_ac = classifier.score(X=X_train, y=Y_train)
    # test_ac = classifier.score(X=X_test, y=Y_test)

    return classifier


def create_predictors(dff):
    """
    Creates classifiers and generator meta-data for each city 
    """

    try:

        # Drop unnecessary features
        features_to_drop = ['Location', 'Position',
                            'LocalTime', 'Condition', 'DewPt', 'label']

        # create a predictor for each city and maintain in a dict to use it
        # later for generating data
        city_weather_predictor = {}
        # store the covariance of relevent features to generate closely related
        # random points
        city_stats_meta = {}

        for city in dff['Location'].unique():
            trainingData = dff[dff['Location'] == city].copy()

            # print('City: {} '.format(city))
            classifier = learn(trainingData, features_to_drop)
            city_weather_predictor.update({city: classifier})
            
            # generate covariance n mean for each city n store it
            # city_stats_meta
            statistics = trainingData.describe()
            mean = [
                statistics['Temp']['mean'],
                statistics['Pressure']['mean'],
                statistics['Humidity']['mean']
            ]
            cov = statistics[['Temp', 'Pressure', 'Humidity']].cov()

            city_stats_meta.update({city: [mean, cov]})
        
    except Exception as e:
        print(e)

    return city_weather_predictor, city_stats_meta




def write_to_file(output):
  """
  Create an output  directory if doesnt exist and writes the output string to file.
  """
  current_directory = os.getcwd()
  output_directory = os.path.join(current_directory, 'Output')
  if not os.path.exists(output_directory):
    os.makedirs(output_directory)

  output_filepath = os.path.join(output_directory, 'Weather_Data.txt')
  
  with open(output_filepath,'a') as output_file:
    output_file.write( output+'\n')
    output_file.flush()

def generate_weather(**kwarg):
  """
   Generates weather data for each city ,predicts the condition and generate the outpur string
  """

  # Transform the weather data for generating the output string
  weather_df = kwarg['weather_data']
  weather_df = weather_df[['City','Day','Hour','Temp','Pressure','Humidity']]
  weather_df = weather_df.iloc[:,0:8].dropna(axis=0,how='any')
  weather_df = weather_df.sample(frac=1).reset_index(drop=True)
  weather_df = weather_df.sort_values(['Day','Hour'])

  # Generate samples
  for i in range(0,kwarg['no_of_samples']):
     
      row = weather_df.iloc[i]
      city,day,hour,temp, pressure, humidity = row     
      predicted_value = kwarg['predictor'].get(city).predict([ row[1:] ])[0]
      condition = kwarg['weather_lookup'][ predicted_value ]
      time = '{:02}:{:02}:{:02}'.format(hour,random.randint(0,60),random.randint(0,60))
      date = '2017-11-{:02}'.format(day)
      localtime = time + 'T' + date + 'Z'
      position = GEOGRAPHICAL_DATA[city]

      output_dict = {
      'city':city,
      'lat':position[0],
      'long':position[1],
      'elevation':position[2],
      'localtime':localtime,
      'condition':condition,
      'temp':temp,
      'pressure':pressure,
      'humidity':humidity
      }

      output_string = '''{city}|{lat},{long},{elevation}|{localtime}|'''.format(**output_dict)
      output_string += '''{condition}|+{temp:.01f}|{pressure:.01f}|{humidity:.0f}'''.format(**output_dict)
      sleep(0.1)

      print(output_string)
      write_to_file(output_string)


def main(argv):
    """
    Main function:

    """

    try:
    
      if len(argv) < 2:
        print("Enter Sample like - python3 weather_model.py 10 ")
        sys.exit()

      if 'test' in argv:
          import ipdb
          ipdb.set_trace()

      NUMBER_OF_SAMPLES =  int(argv[1])
      
      if NUMBER_OF_SAMPLES > 0 and NUMBER_OF_SAMPLES <= 4000:
        # Load the data
        historical_data = pd.read_csv('training_data/input_data_201710.csv')
        weather_data = pd.read_csv('training_data/input_data_201711.csv')
        # Clean and format the data
        clean_data, label_map = prepare_data(historical_data)
        # created predictors and generators
        predictor, city_stats_meta = create_predictors(clean_data)
        # generate weather data  
        param = {
                'predictor':predictor,
                'city_stats':city_stats_meta,
                'weather_lookup':label_map,
                'weather_data': weather_data,
                'no_of_samples': NUMBER_OF_SAMPLES
                }

        # weather = generate_weather_data(**param)
        generate_weather(**param)
      
      else:
        raise Exception('Please enter a valid number of samples, beteween 1 and 4000')
         
    except Exception as e:
      print(e)
      


if __name__ == "__main__":
    main(sys.argv)











