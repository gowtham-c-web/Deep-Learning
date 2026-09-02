import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
print("Name: Gowtham")
print("Reg.no: 814724243047")
sentences = [
    "I love natural language processing",
    "Parts of speech tagging is important",
]
tags = ["PRON VERB ADJ NOUN NOUN", "NOUN ADP NOUN NOUN ADJ"]
word_vocab = sorted(list(set(" ".join(sentences).split())))
tag_vocab = sorted(list(set(" ".join(tags).split())))
word2idx = {word: idx + 2 for idx, word in enumerate(word_vocab)}
word2idx["<PAD>"] = 0
word2idx["<OOV>"] = 1
idx2word = {idx: word for word, idx in word2idx.items()}
tag2idx = {tag: idx + 1 for idx, tag in enumerate(tag_vocab)}
tag2idx["<PAD>"] = 0
idx2tag = {idx: tag for tag, idx in tag2idx.items()}
word_vocab_size = len(word2idx)
tag_vocab_size = len(tag2idx)
input_sequences = [
    [word2idx[word] for word in sentence.split()] for sentence in sentences
]
target_sequences = [
    [tag2idx[tag] for tag in tag_sequence.split()] for tag_sequence in tags
]
max_sequence_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(
    input_sequences, padding="post", maxlen=max_sequence_length
)
target_sequences = pad_sequences(
    target_sequences, padding="post", maxlen=max_sequence_length
)
target_sequences = np.expand_dims(target_sequences, -1)
input_layer = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(
    input_dim=word_vocab_size,
    output_dim=128,
    mask_zero=True,
)(input_layer)
encoder = LSTM(128, return_sequences=True)(embedding_layer)
decoder = Dense(tag_vocab_size, activation="softmax")(encoder)
model = Model(inputs=input_layer, outputs=decoder)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(input_sequences, target_sequences, epochs=10, verbose=1)
new_text = "Seq2Seq models are versatile tools for NLP tasks"
new_text_sequence = [word2idx.get(word, 1) for word in new_text.split()]
new_text_sequence = pad_sequences(
    [new_text_sequence], padding="post", maxlen=max_sequence_length
)
predictions = model.predict(new_text_sequence)
predicted_indices = np.argmax(predictions, axis=-1)[0]
words = new_text.split()[:max_sequence_length]
predicted_tags = [
    idx2tag.get(idx, "<PAD>") for idx in predicted_indices[: len(words)]
]
print(f"Text: {' '.join(words)}")
print(f"Predicted Tags: {' '.join(predicted_tags)}")
