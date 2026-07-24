import numpy as np 
import pandas as pd 
from sklearn.datasets import fetch_lfw_people 
faces = fetch_lfw_people(min_faces_per_person=100, resize=1.0, slice_=(slice(60,188), slice(60, 188)), color=True) 
class_count = len(faces.target_names) 
print("Gowtham C")
print("814724243047")
print(faces.target_names) 
print(faces.images.shape) 
%matplotlib inline 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set() 
fig, ax = plt.subplots(3, 6, figsize=(18, 10)) 
for i, axi in enumerate(ax.flat): 
    axi.imshow(faces.images[i] / 255)
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
from collections import Counter 
counts = Counter(faces.target)
names = {} 
for key in counts.keys(): 
    names[faces.target_names[key]] = counts[key] 
df = pd.DataFrame.from_dict(names, orient='index')
df.plot(kind='bar') 
mask = np.zeros(faces.target.shape, dtype=bool) 
for target in np.unique(faces.target): 
    mask[np.where(faces.target == target)[0][:100]] = 1 
x_faces = faces.data[mask] 
y_faces = faces.target[mask] 
x_faces = np.reshape(x_faces, (x_faces.shape[0], faces.images.shape[1], faces.images.shape[2], faces.images.shape[3])) 
x_faces.shape 
from tensorflow.keras.utils import to_categorical 
from sklearn.model_selection import train_test_split 
face_images = x_faces / 255 # Normalize pixel values
face_labels = to_categorical(y_faces) 
x_train, x_test, y_train, y_test = train_test_split(face_images, face_labels, train_size=0.8, stratify=face_labels, random_state=0) 
from keras.layers import Dense 
from keras.models import Sequential
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D
from keras.layers import Flatten 
model = Sequential() 
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(face_images.shape[1:]))) 
model.add(MaxPooling2D(2, 2)) 
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D(2, 2)) 
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D(2, 2)) 
model.add(Flatten()) 
model.add(Dense(128, activation='relu')) 
model.add(Dense(class_count, activation='softmax')) 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
model.summary() 
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=25) 
acc = hist.history['accuracy'] 
val_acc = hist.history['val_accuracy'] 
epochs = range(1, len(acc) + 1) 
plt.plot(epochs, acc, '-', label='Training Accuracy') 
plt.plot(epochs, val_acc, ':', label='Validation Accuracy') 
# FIX 2: Fixed syntax error caused by the closing smart quote (’)
plt.title('Training and Validation') 
plt.xlabel('Epoch') 
plt.ylabel('Accuracy') 
plt.legend(loc='lower right')
plt.plot() 
from sklearn.metrics import confusion_matrix 
y_predicted = model.predict(x_test) 
mat = confusion_matrix(y_test.argmax(axis=1), y_predicted.argmax(axis=1)) 
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='Blues', xticklabels=faces.target_names, yticklabels=faces.target_names) 
plt.xlabel('Predicted label')
plt.ylabel('Actual label') 
from keras.preprocessing import image 
x = image.load_img('george.jpg', target_size=(face_images.shape[1:]))
plt.xticks([]) 
plt.yticks([]) 
plt.imshow(x) 
x = image.img_to_array(x) / 255
x = np.expand_dims(x, axis=0) 
y = model.predict(x)[0] 
for i in range(len(y)): 
    print(faces.target_names[i] + ': ' + str(y[i]))
