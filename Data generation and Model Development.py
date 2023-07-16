from project_ctrl import *
# run this file to get the results.
# To get the high accuracy required in the paper you must do the following
## set: n_observations = 40,000
## run the whole file from begging to end
# the more n_observations created in the data, the more accurate the model is
# the accuracy of the model is directly proportional to the size of the data used in training
n_observations = 50000
# do not change the number of steps specified below
n_steps = 100
# the line below will generate new observations (i.e. new simulation samples)
data_prepared = create_observations(n_observations)
# the line below just checks the shape
data_prepared.shape
# save dataset in 3D format
# this will create the 10 GB of data that i told you about, however if you increase the number of samples, the data size will also increase
# for example 40,000 samples appx equals 10GB, and so on
np.save('data/data_prepared_3D.npy', data_prepared)
# the line below will simply load the data again, just to make sure that it was saved successfully
data_prepared = np.load('data/data_prepared_3D.npy')
# Modelling
# these are the imports
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Bidirectional, GRU, LSTM, Dense
from tensorflow.keras.models import Sequential
# below we set the X and y of the data
X_data = data_prepared[:, :,0:-3]
y_data = data_prepared[:, :,-3:]
# then we initiate the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, 9)))
model.add(Dense(units=3, activation='selu'))
# here we compile the model and set the loss function and the training metrics
model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])
# the line below will apply the training
history = model.fit(X_data, y_data, epochs=5)
sample = data_prepared[1500]
# below we generate a test sample
x_sample = sample[:,0:-3].reshape(1, sample.shape[0], 9)
y_sample = sample[:,-3:].reshape(1, sample.shape[0], 3)
# we make a prediction
predictions = model.predict(x_sample).reshape(1, sample.shape[0], 3)
s_true = y_sample[:,:,0].flatten()
v_true = y_sample[:,:,1].flatten()
i_true = y_sample[:,:,2].flatten()

s_pred = predictions[:,:,0].flatten()
v_pred = predictions[:,:,1].flatten()
i_pred = predictions[:,:,2].flatten()
# then we import the matplot lib for plotting
import matplotlib.pyplot as plt
t = np.arange(1, s_true.shape[0]+1)
# plot lines
plt.plot(t, s_true, label = "S_true")
plt.plot(t, v_true, label = "v_true")
plt.plot(t, i_true, label = "i_true")
plt.plot(t, s_pred, label = "LSTM_S_pred")
plt.plot(t, v_pred, label = "LSTM_V_pred")
plt.plot(t, i_pred, label = "LSTM_I_pred")
plt.legend()
plt.show()
# finally the model is saved and this will automatically replace the old model
model.save('lstm_model')