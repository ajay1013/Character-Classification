# -*- coding: utf-8 -*-
"""Optical character recognition.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16xvGn62tw-Yu1D5yp4sWn73q1CLgRnLI
"""

#import packages
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import math
import os
from shutil import copyfile

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

!wget http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishFnt.tgz

!tar -xvzf EnglishFnt.tgz

"""#first 0 to 9
#then a to z
#then A to Z
"""

if not os.path.isdir('/content/dataset'):
  os.mkdir('/content/dataset')
if not os.path.isdir('/content/dataset/train'):
  os.mkdir('/content/dataset/train')
if not os.path.isdir('/content/dataset/valid'):
  os.mkdir('/content/dataset/valid')
if not os.path.isdir('/content/dataset/test'):
  os.mkdir('/content/dataset/test')

for i in sorted(os.listdir('/content/English/Fnt')):
  if not os.path.isdir('/content/dataset/train/'+i):
    os.mkdir('/content/dataset/train/'+i)
  if not os.path.isdir('/content/dataset/valid/'+i):
    os.mkdir('/content/dataset/valid/'+i)
  if not os.path.isdir('dataset/test/'+i):
    os.mkdir('/content/dataset/test/'+i)

base = '/content/English/Fnt/Sample'

for char in range(1, 63):
  classLen = len(os.listdir(base + str(char).zfill(3)))

  trainLen = math.floor(classLen*0.80)
  validLen = math.ceil(classLen*0.15)

  randFnt = np.random.randint(low = 1, high = classLen, size = classLen)
  randTrain = randFnt[:trainLen]
  randValid = randFnt[trainLen : trainLen+validLen]
  randTest = randFnt[trainLen+validLen :]

  for imgNo in randTrain:
    src = base+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
    des = 'dataset/train/Sample'+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
    copyfile(src, des)

  for imgNo in randValid:
    src = base+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
    des = 'dataset/valid/Sample'+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
    copyfile(src, des)

  for imgNo in randTest:
    src = base+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
    des = 'dataset/test/Sample'+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
    copyfile(src, des)

data_generator = ImageDataGenerator(rescale=1.0/255.0)

train_generator = data_generator.flow_from_directory('/content/dataset/train', target_size = (128,128), 
    batch_size = 128,
    color_mode = 'grayscale',
    class_mode = 'categorical')

validation_generator = data_generator.flow_from_directory('/content/dataset/valid',target_size = (128,128),
    batch_size = 128,
    color_mode = 'grayscale',
    class_mode = 'categorical')

path = '/content/dataset/train'

for i in os.listdir(path):
  each_folder = os.path.join(path,i)
  print(len(os.listdir(each_folder)))
  print(i)

def ocrModel():
  model = Sequential()
  model.add(Conv2D(32, (4,4), strides = (1,1), activation = 'relu', input_shape = (128, 128, 1)))
  model.add(MaxPooling2D(pool_size = (4,4), strides = (2,2)))
  model.add(Conv2D(64, (4,4), strides = (1,1), activation = 'relu', input_shape = (128, 128, 1)))
  model.add(MaxPooling2D(pool_size = (4,4),strides = (2,2)))
  model.add(Flatten())
  model.add(Dense(310, activation='relu'))
  model.add(Dense(62, activation = 'softmax'))
  model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
  return model

model = ocrModel()
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

fit_history = model.fit(train_generator, epochs = 20, validation_data = validation_generator)

plt.plot(fit_history.history["loss"], label="Loss")
plt.plot(fit_history.history["val_loss"], label="Val Loss")
plt.legend()

plt.plot(fit_history.history["acc"], label="acc")
plt.plot(fit_history.history["val_acc"], label="Val acc")
plt.legend()

test_generator = data_generator.flow_from_directory(
    '/content/dataset/test',
    target_size = (128,128),
    shuffle = False,
    color_mode='grayscale')

eval = model.evaluate_generator(test_generator, verbose=1)
print('Model performance:')
print('loss for test dataset is : {}'.format(eval[0]))
print('accuracy for test dataset is : {}'.format(eval[1]))

model.save('/content/OCRmodel.h5')

def image(Image_Path):
  x = cv2.imread(Image_Path)
  gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
  p = cv2.resize(gray, (128,128))
  y = np.expand_dims(p, axis=2)
  y = np.expand_dims(y, axis=0)
  probab = model.predict(y)
  result = np.argmax(probab)
  return result

import cv2

! wget https://i.pinimg.com/originals/0f/62/65/0f6265fe8dc448b8f032f161db77c033.png

image('/content/0f6265fe8dc448b8f032f161db77c033.png')

!wget https://printables.space/files/uploads/download-and-print/large-printable-numbers/2-a4-1200x1697.jpg

image('/content/2-a4-1200x1697.jpg')

!wget https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/LetterA.svg/1200px-LetterA.svg.png

image('/content/1200px-LetterA.svg.png')

