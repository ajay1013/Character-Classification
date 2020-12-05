# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 16:06:59 2020

@author: dell
"""


import numpy as np
import os
import cv2
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model


app = Flask(__name__)

upload_folder = r'C:\Users\dell'

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

@app.route('/',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            file_path = os.path.join(upload_folder, image_file.filename)
            image_file.save(file_path)
            preds = model_predict(file_path)
            return render_template("Sample.html", prediction=preds)
    return render_template("Sample.html", prediction='NAN')

if __name__ == "__main__":
    app.run(port=12000)