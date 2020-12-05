# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 17:34:56 2020

@author: dell
"""


import numpy as np
import os
import cv2
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model


app = Flask(__name__)

MODEL_PATH = 'OCRmodel.h5'
model = load_model(MODEL_PATH)

def model_predict(image_path):
  x = cv2.imread(image_path)
  gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
  p = cv2.resize(gray, (128,128))
  y = np.expand_dims(p, axis=2)
  y = np.expand_dims(y, axis=0)
  probab = model.predict(y)
  result = np.argmax(probab)
  return result


@app.route('/', methods=['GET'])
def home():
    return render_template('Sample.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads')
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        return preds
    return None

if __name__ == "__main__":
    app.run()