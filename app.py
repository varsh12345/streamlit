import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.models import load_model
import streamlit as st
import pandas as pd
import pandas_datareader as data
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from keras.models import load_model
import streamlit as st


start = '2017-04-15'
end = st.text_input('Enter End Date','2022-04-15')
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker ','AAPL')
df=data.DataReader(user_input,'yahoo',start,end)

st.subheader('Date from '+ start + ' to '+ end)
st.write(df.tail())
st.write(df.describe())
df1=df.reset_index()['Close']
df1.head()
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

time_step=100;
dataX=[]
dataY = []
for i in range(len(train_data)-time_step-1):
    a = train_data[i:(i+time_step), 0] 
    dataX.append(a)
    dataY.append(train_data[i + time_step, 0])
X_train, y_train =np.array(dataX), np.array(dataY)

datax=[]
datay=[]
for i in range(len(test_data)-time_step-1):
    a = test_data[i:(i+time_step), 0] 
    datax.append(a)
    datay.append(test_data[i + time_step, 0])
X_test, ytest =np.array(datax), np.array(datay)
time_step = 100
# X_train, y_train = create_dataset(train_data, time_step)
# X_test, ytest = create_dataset(test_data, time_step)
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

# model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=10,batch_size=64,verbose=1)
model = load_model('keras_model.h5')
import tensorflow as tf

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))

math.sqrt(mean_squared_error(ytest,test_predict))

look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
fig = plt.figure(figsize=(12,6))
st.subheader('Training data and Testing data')
plt.plot(scaler.inverse_transform(df1),'b',label='original')
plt.plot(trainPredictPlot,'r',label='training data')
plt.plot(testPredictPlot,'g',label='testing data')
plt.show()
st.pyplot(fig)


x_input=test_data[len(test_data)-100:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

from numpy import array
lst_output=[]
n_steps=100
i=0
while(i<20):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        # print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1,100, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        # print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, 100,1))
        yhat = model.predict(x_input, verbose=0)
        # print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        # print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1

day_new=np.arange(1,101)
day_pred=np.arange(101,121)

import matplotlib.pyplot as plt
st.subheader('Prediction of future 20 DAYS')
fig = plt.figure(figsize=(12,6))
plt.plot(day_new,scaler.inverse_transform(df1[len(df1)-100:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
st.pyplot(fig)

df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[len(df1)-100:])
fig = plt.figure(figsize=(12,6))
df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)
st.pyplot(fig)
