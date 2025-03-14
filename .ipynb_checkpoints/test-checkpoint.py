#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib

# Load the saved model
model = joblib.load('random_forest_model.pkl')

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
   # For rendering results on HTML GUI
    
    features = [x for x in request.form.values()]
    final_features = pd.DataFrame(features)
    
    prediction = model.predict(final_features)

    return render_template('index.html', prediction_text='Transaction should be  {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)

