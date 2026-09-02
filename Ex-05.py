import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
print("Name: Gowtham C")
print("Reg.no: 814724243047")
text_data = [
    "I love this product!",
    "This is great. Highly recommended.",
    "Terrible experience. Never buying again.",
    "The service was average.",
]
labels = np.array([1, 1, 0, 0])
max_words = 1000
vectorize_layer = layers.TextVectorization(
    max_tokens=max_words, output_mode="int", standardize="lower_and_strip_punctuation"
)
vectorize_layer.adapt(text_data)
sequences = vectorize_layer(text_data)
max_sequence_length = sequences.shape[1]
model = Sequential(
    [
        tf.keras.Input(shape=(max_sequence_length,)),
        layers.Embedding(input_dim=max_words, output_dim=128),
        layers.LSTM(128),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)
model.fit(sequences, labels, epochs=10, verbose=1)
test_data = [
    "This is a good movie.",
    "I can't believe how bad this service is.",
]
test_sequences = vectorize_layer(test_data)
predictions = model.predict(test_sequences)
for text, pred in zip(test_data, predictions):
    sentiment = "positive" if pred[0] > 0.5 else "negative"
    print(f"'{text}' -> {sentiment} ({pred[0]:.4f})")
