from collections import Counter
import matplotlib.pyplot as plt
import keras.utils as image
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.datasets import fetch_lfw_people
import pandas as pd
import numpy as np
print("Name: Gowtham C")
print("Reg.no: 814724243047")
# Load the LFW dataset
faces = fetch_lfw_people(
    min_faces_per_person=100,
    resize=1.0,
    slice_=(slice(60, 188), slice(60, 188)),
    color=True
)

class_count = len(faces.target_names)
print(faces.target_names)
print(faces.images.shape)

sns.set()

# Plot first 18 sample images
fig, ax = plt.subplots(3, 6, figsize=(18, 10))
for i, axi in enumerate(ax.flat):
    # Scale pixel values so Matplotlib doesn't clip everything above 1.0
    axi.imshow(faces.images[i] / 255)
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])

# Count occurrences of each target
counts = Counter(faces.target)
names = {}
for key in counts.keys():
    names[faces.target_names[key]] = counts[key]

# Plot class distribution bar chart
df = pd.DataFrame.from_dict(names, orient='index')
df.plot(kind='bar')

# Create a balanced mask to pick 100 images per person
mask = np.zeros(faces.target.shape, dtype=bool)
for target in np.unique(faces.target):
    mask[np.where(faces.target == target)[0][:100]] = True

x_faces = faces.data[mask]
y_faces = faces.target[mask]

# Reshape flattened data back to image format
x_faces = np.reshape(
    x_faces,
    (
        x_faces.shape[0],
        faces.images.shape[1],
        faces.images.shape[2],
        faces.images.shape[3]
    )
)
print(x_faces.shape)

# Normalize pixel values and encode labels
face_images = x_faces / 255
face_labels = to_categorical(y_faces)

# Split data into training and validation sets
x_train, x_test, y_train, y_test = train_test_split(
    face_images,
    face_labels,
    train_size=0.8,
    stratify=face_labels,
    random_state=0
)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',
          input_shape=(face_images.shape[1:])))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(class_count, activation='softmax'))

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Train model
hist = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=20,
    batch_size=25
)

# Plot training and validation accuracy
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.figure()
plt.plot(epochs, acc, '-', label='Training Accuracy')
plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Generate and plot confusion matrix
y_predicted = model.predict(x_test)
mat = confusion_matrix(y_test.argmax(axis=1), y_predicted.argmax(axis=1))

plt.figure(figsize=(10, 8))
sns.heatmap(
    mat.T,
    square=True,
    annot=True,
    fmt='d',
    cbar=False,
    cmap='Blues',
    xticklabels=faces.target_names,
    yticklabels=faces.target_names
)
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()

# Test the model on a local custom image
x = image.load_img(r'C:\Users\Administrator\Desktop\Newfolder\george.jpg', target_size=(face_images.shape[1:]))
plt.figure()
plt.xticks([])
plt.yticks([])
plt.imshow(x)
plt.show()

x = image.img_to_array(x) / 255
x = np.expand_dims(x, axis=0)
y = model.predict(x)[0]

# Print prediction probabilities
for i in range(len(y)):
    print(f"{faces.target_names[i]}: {y[i]:.4f}")
