import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
DATASET_PATH = "dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 6  # Updated to 6
CATEGORIES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

# Data Preprocessing
logging.info("Initializing data generators...")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset="training")

val_data = datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', subset="validation")

# Load Pretrained Model (VGG19)
logging.info("Loading VGG19 model...")
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False  # Freeze pretrained layers

# Build Model
logging.info("Building custom model on top of VGG19...")
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(CATEGORIES), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
logging.info(f"Starting model training for {EPOCHS} epochs...")
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Evaluate Model
logging.info("Evaluating model...")
val_preds = model.predict(val_data)
y_true = val_data.classes
y_pred = np.argmax(val_preds, axis=1)

print(classification_report(y_true, y_pred, target_names=CATEGORIES))
logging.info("Model evaluation complete.")

# Save the Model
os.makedirs("models", exist_ok=True)
model.save("models/eye_disease_model.h5")
logging.info("Model saved successfully!")

# Plot Training History
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()
