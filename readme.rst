

# Weather Generator

#### Aim : To generate weather data for cities in given format

#### Plan : 

- Fetch the consolidated data from weather website for past few months (done using data_provider.py script)
- Transform the data and selected relevant features
- Since each city has different weather conditions, make a classifier for each city and
  use to predict the condition of the weather based on Temperature, Pressure and Humidity
- To generate the weather data:
    - Plan A:
        Either Use covariance amongst all the features and generate random data closely related to each other
    - Plan B:
        Use some other months data to provide features and use our classifier to predict the weather it.
        
        
#### Assumptions:

- There are only three weather types - Rain, Snow and Sunny 

- 

#### Download the dependencies:


```
sudo pip3 install -r requirement.txt 
```

#### Run :

- Run the program from the project directory
- Must provide number of samples as argumemts ranging between 0-4000
```
python3 weather_model.py 200
```


#### Notebook:

- The jupyter book can be used the to the data analysis and visualizations included in it.

##### To View the notebook

```
jupyter notebook
```


#### Resources :

 * [Wunder Ground](https://serbian.wunderground.com/weather/api/d/docs?d=index) for Weather data.
 * [Sci Kit Learn](http://scikit-learn.org/) for analytics
 
 ###### Cheers :+1:
