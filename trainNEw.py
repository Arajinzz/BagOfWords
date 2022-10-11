# A SIMPLE MODEL TO USE LATER

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy as np


# HERE GOES DATASET
dataset = np.load('bow.npy', allow_pickle=True)
# SHUFFLE DATA
np.random.shuffle(dataset)


# HERE WE SEPERATE FEATURES FROM LABELS
X = dataset[:, :-9]
y = dataset[:, -9:]

#print()
#exit(1)

model = Sequential()

# INPUT AND FIRST LAYER
model.add(Dense(128, input_shape=(X.shape[1], ), activation='relu'))

# OUTPUT LAYER
# CAN USE SIGMOID
model.add(Dense(y.shape[1], activation='softmax'))

# CAN USE https://keras.io/losses/
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, validation_split=0.2, epochs=100, batch_size=32)

_, accuracy = model.evaluate(X, y)

print('Accuracy : ', accuracy * 100)

model.save('test_model.h5')
'''
# PREDICTIONS
# FEATURES TO PREDICT
toPredict = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(8, 1)
predictions = model.predict(toPredict)

print(predictions)'''
