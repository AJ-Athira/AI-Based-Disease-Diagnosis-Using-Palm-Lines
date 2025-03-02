import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define paths
dataset_path = "C:/all pros/ML pro/dataset"
train_csv = os.path.join(dataset_path, "updated_train_annotations.csv")
train_img_folder = os.path.join(dataset_path, "train")

# Load dataset
df = pd.read_csv(train_csv)

# Image parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_and_preprocess_image(filename):
    img_path = os.path.join(train_img_folder, filename)
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize
    return img_array

# Load images and labels
X = np.array([load_and_preprocess_image(fname) for fname in df['filename']])
y = pd.get_dummies(df['disease_diagnosis']).values  # One-hot encode labels

# Load EfficientNetB3 as feature extractor
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model

# Add custom classification layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(y.shape[1], activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=out)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=10, batch_size=BATCH_SIZE, validation_split=0.2)

# Save model
model.save("disease_model.h5")
print("Model training complete. Saved as disease_model.h5")
