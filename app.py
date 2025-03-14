#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
from datetime import datetime
import geoip2.database
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the saved model
model = joblib.load('random_forest_model.pkl')

# Load the saved transformers
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('country_freq_encoding.pkl', 'rb') as f:
    country_freq_encoding = pickle.load(f)

with open('sex_encoding.pkl', 'rb') as f:
    sex_encoding = pickle.load(f)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
   # For rendering results on HTML GUI

    signup_datetime = datetime.strptime(request.form['signup_datetime'], '%Y-%m-%dT%H:%M')
    signup_datetime = signup_datetime.strftime('%m/%d/%Y %H:%M')

    tr_datetime = datetime.strptime(request.form['tr_datetime'], '%Y-%m-%dT%H:%M')
    tr_datetime = tr_datetime.strftime('%m/%d/%Y %H:%M')

    amount = float(request.form['amount'])

    device_id = int(request.form['device_id'])

    store = request.form['store']

    browser = request.form['browser']

    sex = request.form['sex']

    age = int(request.form['age'])

    ip_address = request.form['ip_address']

    # Create a dictionary of the data
    input_data = {
        'signup_datetime': [signup_datetime],
        'tr_datetime': [tr_datetime],
        'amount': [amount],
        'device_id': [device_id],
	'store': [store],
	'browser': [browser],
	'sex': [sex],
	'age': [age],
	'ip_address': [ip_address]
    }

    # Create DataFrame
    df = pd.DataFrame(input_data)

    # Initialize the reader
    reader = geoip2.database.Reader('GeoLite2-Country.mmdb')

    # Function to fetch country from IP address
    def get_country_from_ip(ip):
    	try:
        	response = reader.country(ip)
        	return response.country.name
    	except:
        	return "Unknown"

    # Apply the function to the dataset
    df['Country'] = df['ip_address'].apply(get_country_from_ip)

    # Close the reader
    reader.close()

    # Extracting transaction hour, day of week, and time difference
    df['signup_hour'] = pd.to_datetime(df['signup_datetime']).dt.hour
    df['transaction_hour'] = pd.to_datetime(df['tr_datetime']).dt.hour
    df['transaction_day'] = pd.to_datetime(df['tr_datetime']).dt.dayofweek
    df['time_since_signup'] = (pd.to_datetime(df['tr_datetime']) - pd.to_datetime(df['signup_datetime'])).dt.total_seconds() / (3600 * 24)  # Days

    print(df)
    print(df.dtypes)

    # Apply preprocessing using the loaded transformers
    #  Add the country frequency
    country = df.loc[0, 'Country']
    country_freqency = country_freq_encoding.get(country, 0)
    df['country_freqency'] = country_freqency

    #  Add the sex encoding
    sex_encoded = sex_encoding.get(sex, -1)
    df['sex'] = sex_encoded

    #  Add the scaled numerical features
    time_since_signup = df.loc[0, 'time_since_signup']
    numerical_features = np.array([[amount, age, time_since_signup]])
    scaled_numerical = scaler.transform(numerical_features)

    # Assign scaled values to the DataFrame
    df['amount'] = scaled_numerical[0][0]
    df['age'] = scaled_numerical[0][1]
    df['time_since_signup'] = scaled_numerical[0][2]

    #  Add the store and browser encoding
    categorical_features = pd.DataFrame([{
        'store': store,
        'browser': browser
    }])
    encoded_categorical = encoder.transform(categorical_features)

    # Get the feature names from the encoder if available (optional)
    try:
        encoded_feature_names = encoder.get_feature_names_out(['store', 'browser'])
    except:
        encoded_feature_names = [f'encoded_{i}' for i in range(encoded_categorical.shape[1])]

    # Add each encoded column to df
    for idx, col_name in enumerate(encoded_feature_names):
        df[col_name] = encoded_categorical[0][idx]

    

    # Dropping unnecessary columns for modeling
    columns_to_drop = [
        'store', 'browser', 'Country', 'signup_datetime', 'tr_datetime', 'ip_address'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    model_features_order = [
    'amount',
    'device_id',
    'sex',
    'age',
    'signup_hour',
    'transaction_hour',
    'transaction_day',
    'time_since_signup',
    'store_pets',
    'store_toys',
    'browser_FireFox',
    'browser_IE',
    'browser_Opera',
    'browser_Safari',
    'country_freqency'
    ]

    df = df[model_features_order]
    
    print(df)
    print(df.dtypes)
        
    prediction = model.predict(df)

    return render_template('index.html', prediction_text='Transaction should be  {}'.format(prediction))
   

if __name__ == "__main__":
    app.run(debug=True)

