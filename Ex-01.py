# XOR Gate using Deep Neural Network

import numpy as np

# Import Deep Learning libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Step 1: Create XOR input data

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


# Step 2: Create XOR output labels

Y = np.array([
    [0],
    [1],
    [1],
    [0]
])


# Step 3: Create Neural Network model

model = Sequential()


# Step 4: Add hidden layer

model.add(Dense(
    units=4,
    input_dim=2,
    activation='relu'
))


# Step 5: Add output layer

model.add(Dense(
    units=1,
    activation='sigmoid'
))


# Step 6: Compile the model

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# Step 7: Train the model

model.fit(
    X,
    Y,
    epochs=500,
    verbose=0
)


# Step 8: Test the model

prediction = model.predict(X)


print("Input:")
print(X)

print("\nExpected Output:")
print(Y)

print("\nPredicted Output:")
print(prediction)


# Convert probabilities to 0 or 1

print("\nRounded Prediction:")
print(np.round(prediction))
