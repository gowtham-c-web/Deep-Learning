import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation="softmax")
])
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, verbose=1)
print("Name: Herman Joel Raj J")
print("Reg.no: 814724243059")
print("Accuracy:", model.evaluate(x_test, y_test, verbose=0)[1])
pred = np.argmax(model.predict(x_test[0:1]), axis=1)
plt.imshow(x_test[0].reshape(28, 28), cmap="gray")
plt.title(f"Predicted: {pred[0]} Actual: {y_test[0]}")
plt.axis("off")
plt.show()