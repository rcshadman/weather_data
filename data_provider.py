#!/usr/bin/env python3

import requests as web
import json,sys, os, shutil, glob
from time import sleep
from pprint import pprint as pp
import pandas as pd

START_DATA=1
END_DATE=32
DELAY=6.5
YEARMONTH='201710'
COUNTRY='Australia'
KEY='15f7f977b8b6a26a'

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
		tmp = {key: val for key, val in obj.items() if key in features_list and obj[key] not in ('','-999','-9999','-9999.0','-9999.00')}
		clean_data.append(tmp)
	hourly_features = { each['date']['hour']: each for each in clean_data }
	return hourly_features

def write_to_file(data,date,city):
	current_directory = os.getcwd()
	resource_directory = os.path.join(current_directory,'resource')
	final_directory = os.path.join(resource_directory, city)
	if not os.path.exists(final_directory):
		os.makedirs(final_directory)
	file_name = date+'.txt'
	file_path = os.path.join(final_directory,file_name)
	print(file_path)

	with open(file_path,'w') as to_file:
		json.dump(data,to_file)

def extract():
	for city in city_list:
		for day in range(START_DATA,END_DATE):
			param={
			'key':KEY,
			'date':YEARMONTH+'{:02}'.format(day),
			'country':COUNTRY,
			'city':city
			}
			url = "http://api.wunderground.com/api/{key}/history_{date}/q/{country}/{city}.json".format(**param)
			sleep(DELAY)
			print(url)
			response = request('@URL: - ' + url)
			json_data = clean(response)
			write_to_file(json_data,param['date'],city)
			print("DONE : -> " + " - city - "+ city + " - date - " +param['date'])



def consolidate_to_csv():
	csv_rows = []
	for city in city_list:
		city_directory = os.path.join('.', city)
		day_files = [f for f in os.listdir(city_directory) if f[-4:]=='.txt']
		
		# Loop through every day
		for day_file in day_files:
			day_file_path = os.path.join(city_directory, day_file)
			day_data = json.load(open(day_file_path))

			# Get the day number from the file name
			day = day_file.split('.txt')[0][6:]
			
			# Loop through every hour:
			for hour in [ key for key in day_data.keys() ]:
				csv_row = [city, day, hour]
				for feature in features_list[1:]:
					feature_data = day_data[hour].get(feature)
					csv_row.append(feature_data)
				csv_rows.append(csv_row)


	columns = ['City', 'Day', 'Hour', 'Condition', 'Humidity', 'Temp', 'Pressure', 'DewPt', 'WindSpeed', 'Visibility']
	all_data = pd.DataFrame(csv_rows, columns=columns)
	all_data.to_csv('../training_data/input_data_oct2.csv')	

def main(argv):
	if 'fetch' in argv:
		extract()
	if 'compile' in argv:
		consolidate_to_csv()
	if 'test' in argv:
		import ipdb; ipdb.set_trace()


if __name__ == "__main__":
	main(sys.argv)
