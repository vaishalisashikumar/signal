import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
image_dir = 'C:\\Users\\vaishali\\Downloads\\archive\\images'
annotation_dir = 'C:\\Users\\vaishali\\Downloads\\archive\\annotations'
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Image directory '{image_dir}' not found.")
if not os.path.exists(annotation_dir):
    raise FileNotFoundError(f"Annotation directory '{annotation_dir}' not found.")
images = []
labels = []
for filename in os.listdir(annotation_dir):
    if filename.endswith('.xml'):
        tree = ET.parse(os.path.join(annotation_dir, filename))
        root = tree.getroot()
        image_path = os.path.join(image_dir, root.find('filename').text)
        print(f"Reading image file: {image_path}")
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (224, 224))
            image = image / 255.0  # Normalize pixel values
            images.append(image)
            label = root.find('object').find('name').text
            labels.append(label)
        else:
            print(f"Failed to read image: {image_path}")
images = np.array(images)
labels = np.array(labels)
print("Number of images:", len(images))
print("Number of labels:", len(labels))
if len(images) == 0:
    raise ValueError("No images found in the dataset directory.")
images, labels = shuffle(images, labels, random_state=42)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(len(label_encoder.classes_), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Decode predicted labels
y_pred = model.predict(X_test)
predicted_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))
print("Predicted labels:", predicted_labels)


