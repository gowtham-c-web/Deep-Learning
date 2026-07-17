# XOR Gate using Deep Neural Network
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
print("Gowtham C")
print("814724243047")
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
Y = np.array([
    [0],
    [1],
    [1],
    [0]
])
model = Sequential()
model.add(Dense(
    units=4,
    input_dim=2,
    activation='relu'
))
model.add(Dense(
    units=1,
    activation='sigmoid'
))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(
    X,
    Y,
    epochs=500,
    verbose=0
)
prediction = model.predict(X)
print("Input:")
print(X)
print("\nExpected Output:")
print(Y)
print("\nPredicted Output:")
print(prediction)
print("\nRounded Prediction:")
print(np.round(prediction))
