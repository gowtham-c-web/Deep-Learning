import os
import numpy as np
import keras.utils as image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50V2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, RandomFlip, RandomRotation, RandomTranslation, RandomZoom
BASE_PATH = "./arctic-wildlife"
print("Name: Gowtham C")
print("Reg.no: 814724243047")
def load_images_from_path(path, label):
    images = []
    labels = []
    if not os.path.exists(path):
        print(f"Warning: Path {path} does not exist.")
        return images, labels
    for file in os.listdir(path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img = image.load_img(os.path.join(path, file), target_size=(224, 224))
            images.append(image.img_to_array(img))
            labels.append(label)
    return images, labels
def show_images(images):
    if not images:
        return
    num_imgs = min(len(images), 8)
    fig, axes = plt.subplots(1, num_imgs, figsize=(20, 5), subplot_kw={'xticks': [], 'yticks':[]})
    if num_imgs == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(images[i] / 255.0)
    plt.show()
x_train, y_train = [], []
x_test, y_test = [], []
class_labels = ['arctic fox', 'polar bear', 'walrus']
dir_mapping = {
    'arctic_fox': 0,
    'polar_bear': 1,
    'walrus': 2
}
print("Loading training images...")
for folder_name, label_idx in dir_mapping.items():
    path = os.path.join(BASE_PATH, 'train', folder_name)
    imgs, lbls = load_images_from_path(path, label_idx)
    x_train.extend(imgs)
    y_train.extend(lbls)
    print(f" Loaded {len(imgs)} training images for {folder_name}")
print("\nLoading test images...")
for folder_name, label_idx in dir_mapping.items():
    path = os.path.join(BASE_PATH, 'test', folder_name)
    imgs, lbls = load_images_from_path(path, label_idx)
    x_test.extend(imgs)
    y_test.extend(lbls)
    print(f" Loaded {len(imgs)} test images for {folder_name}")
show_images(x_train[-8:])
x_train = preprocess_input(np.array(x_train))
x_test = preprocess_input(np.array(x_test))
y_train_encoded = to_categorical(y_train, num_classes=3)
y_test_encoded = to_categorical(y_test, num_classes=3)
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
model = Sequential([
    RandomFlip(mode='horizontal'),
    RandomTranslation(0.2, 0.2),
    RandomRotation(0.2),
    RandomZoom(0.2),
    base_model,
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("\nStarting model training...")
hist = model.fit(x_train, y_train_encoded, validation_data=(x_test, y_test_encoded), batch_size=10, epochs=25)
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
sns.set_theme()
y_predicted = model.predict(x_test)
mat = confusion_matrix(y_test_encoded.argmax(axis=1), y_predicted.argmax(axis=1))
plt.figure()
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.title('Confusion Matrix')
plt.show()
def predict_single_image(img_path):
    if not os.path.exists(img_path):
        print(f"\nWarning: Sample image file not found at: {img_path}")
        return
    img = image.load_img(img_path, target_size=(224, 224))
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.title(f"Predicting: {os.path.basename(img_path)}")
    plt.show()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    print(f"\nPredictions for {os.path.basename(img_path)}:")
    for i, label in enumerate(class_labels):
        print(f' {label}: {predictions[0][i]:.4f}')
sample_fox_path = os.path.join(BASE_PATH, 'samples', 'arctic_fox', 'arctic_fox_140.jpeg')
sample_walrus_path = os.path.join(BASE_PATH, 'samples', 'walrus', 'walrus_143.png')
predict_single_image(sample_fox_path)
predict_single_image(sample_walrus_path)
