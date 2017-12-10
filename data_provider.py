# #!/usr/bin/env python3
# coding: utf-8

# data_provider.py


"""
 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation; 

 Run this generatate new training data. 
 Note: The api is limited to 10 requests per minute, therefore we have a delay of 6.5 seconds between each call
 """

__author__ = "Md Shadman Alam"
__email__ = "rcshadman@gmail.com"


import requests as web
import json
import sys
import os
import shutil
import glob
from time import sleep
from pprint import pprint as pp
import pandas as pd

START_DATA = 1
END_DATE = 32
YEARMONTH = ''
DELAY = 6.5
COUNTRY = 'Australia'
KEY = '15f7f977b8b6a26a'

city_list = [
    'Sydney',
    'Melbourne',
    'Brisbane',
    'Gold_Coast',
    'Adelaide',
    'Darwin',
    'Wollongong',
    'Canberra',
    'Newcastle',
    'Hobart',
]

features_list = [
    'date',
    'conds',
    'hum',
    'tempm',
    'pressurem',
    'dewptm',
    'wsdpm',
    'vism'
]


def request(url):
    response = web.get(url)
    parsed_json = response.json()
    return parsed_json


def clean(raw_data):
    clean_data = []
    for obj in raw_data['history']['observations']:
        tmp = {key: val for key, val in obj.items() if key in features_list and obj[key] not in ('', '-999', '-9999', '-9999.0', '-9999.00')}
        clean_data.append(tmp)
    hourly_features = {each['date']['hour']: each for each in clean_data}
    return hourly_features


def write_to_file(data, date, city):
    current_directory = os.getcwd()
    resource_directory = os.path.join(current_directory, 'resource')
    date_directory = os.path.join(resource_directory, YEARMONTH)
    final_directory = os.path.join(date_directory, city)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    file_name = date+'.txt'
    file_path = os.path.join(final_directory, file_name)
    print(file_path)

    with open(file_path, 'w') as to_file:
        json.dump(data, to_file)


def extract():
    for city in city_list:
        for day in range(START_DATA, END_DATE):
            param = {
                'key': KEY,
                'date': YEARMONTH+'{:02}'.format(day),
                'country': COUNTRY,
                'city': city
            }
            url = "http://api.wunderground.com/api/{key}/history_{date}/q/{country}/{city}.json".format(
                **param)
            sleep(DELAY)
            print('@URL: - ' + url)
            response = request(url)
            json_data = clean(response)
            write_to_file(json_data, param['date'], city)
            print("DONE : -> " + " - city - " +
                  city + " - date - " + param['date'])


def consolidate_to_csv():
    csv_rows = []
    for city in city_list:
        current_directory = os.getcwd()
        resource_directory = os.path.join(current_directory, 'resource')
        date_directory = os.path.join(resource_directory, YEARMONTH)
        if  os.path.isdir(date_directory+'/'+city):
            city_directory = os.path.join(date_directory, city)
            day_files = [f for f in os.listdir(city_directory) if f[-4:] == '.txt']

            # Loop through every day
            for day_file in day_files:
                day_file_path = os.path.join(city_directory, day_file)
                day_data = json.load(open(day_file_path))

                # Get the day number from the file name
                day = day_file.split('.txt')[0][6:]

                # Loop through every hour in each file:
                for hour in [key for key in day_data.keys()]:
                    csv_row = [city, day, hour]
                    for feature in features_list[1:]:
                        feature_data = day_data[hour].get(feature)
                        csv_row.append(feature_data)
                    csv_rows.append(csv_row)

    columns = ['City', 'Day', 'Hour', 'Condition', 'Humidity',
               'Temp', 'Pressure', 'DewPt', 'WindSpeed', 'Visibility']
    all_data = pd.DataFrame(csv_rows, columns=columns)

    training_directory = os.path.join(current_directory, 'training_data')
    if not os.path.exists(training_directory):
        os.makedirs(training_directory)
    
    output_file_path = os.path.join(training_directory, 'input_data_{}.csv'.format(YEARMONTH))
    all_data.to_csv(output_file_path)


def main(argv):
    try:
        if 'test' in argv:
            import ipdb
            ipdb.set_trace()

        if '--month' in argv and len(argv) > 2:
            
            args_map = { args:index for index,args in enumerate(argv) }
            month = argv[ args_map['--month'] + 1]
            
            if int( month[4:]) in range(1,13):
                global YEARMONTH
                YEARMONTH = argv[ args_map['--month'] + 1]

            else:
                raise Exception("Enter Sample like - python3 data_provider.py [ fetch | compile | test ] --month 201702 or check the month!")
                sys.exit()
        

            if 'fetch'in argv:
                print('Extracting for year - ' + YEARMONTH)
                extract()

            if 'compile' in argv:
                consolidate_to_csv()
                print('Consolidated')
        else:
            raise Exception("Enter Sample like - python3 data_provider.py [ fetch | compile | test ] --month 201702 or check the month!")
            sys.exit()

            
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main(sys.argv)
