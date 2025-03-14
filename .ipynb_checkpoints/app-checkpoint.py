#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import threading
''' 
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def run_flask():
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=True)

# Run Flask in a background thread
thread = threading.Thread(target=run_flask)
thread.start()


# Load the saved model
model = joblib.load('random_forest_model.pkl')
'''
# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
'''
@app.route('/predict',methods=['POST'])
def predict():
    
   # For rendering results on HTML GUI
    
    int_features = [int(x) for x in request.form.values()]
    final_features = pd.DataFrame(int_features)
    
    prediction = model.predict(final_features).tolist()

    return render_template('index.html', prediction_text='Transaction should be  {}'.format(prediction))

'''
if __name__ == "__main__":
    app.run(debug=True)

   


# In[ ]:




