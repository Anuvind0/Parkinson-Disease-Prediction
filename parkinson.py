#Importing necessary modules
import tensorflow as tf
import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
#Importing Dataset
park_file='/parkinsons.csv'
park=pd.read_csv(park_file)
label=np.array(park['status'])
column_names=[]
for i in park.columns:
  if i ==('name'):
    pass
  elif i=='status':
    pass
  else:
    column_names.append(i)
print(len(label))
y=[]
with open('/parkinsons.csv','r') as csvfile:
  csvreader=csv.reader(csvfile)
  rows=list(csvreader)
for i in range(1,196):
  x=list(rows[i])
  del x[0]
  del x[16]
  for j in range(len(x)):
    c=float(x[j])
    x[j]=c
  y.append(x)
print(y)
training_data=np.array(y[0:150])
test_data=np.array(y[150:])
label_data=np.array(label[0:150])
test_label_data=np.array(label[150:])
model=tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1,input_shape=(150,22))])
model.compile(optimizer='adam',loss='mean_absolute_error',metrics=['accuracy'])
history=model.fit(training_data,label_data,epochs=500)
model.evaluate(test_data,test_label_data)
k=training_data[0:]
o=label_data[0:]
print(k)
f=model.predict(k)
for i in range(len(f)):
  if f[i]>0.5:
    f[i]=1
  else:
    f[i]=0
count=0
for i in range(len(f)):
  if f[i]==o[i]:
    count+=1
print(count)
plt.plot(history.history['accuracy'],'b')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.show()
plt.plot(history.history['loss'],'r')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
