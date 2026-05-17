import os
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
	Conv2D,
	MaxPooling2D,
	Flatten,
	Dense,
	Dropout,
	BatchNormalization
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_PATH = "dataset"
MODEL_SAVE_PATH = "tomato_model.h5"

USE_GRAYSCALE = False
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001

CLASS_NAMES = ["best", "good", "rotten"]
NUM_CLASSES = len(CLASS_NAMES)

TRAIN_MODEL = True
TEST_SINGLE_IMAGE = True
SINGLE_IMAGE_PATH = "test.jpg"


# =========================================================
# IMAGE LOADING
# =========================================================


def load_images():
	images = []
	labels = []

	for class_index, class_name in enumerate(CLASS_NAMES):
		folder_path = os.path.join(DATASET_PATH, class_name)

		if not os.path.exists(folder_path):
			print(f"Folder not found: {folder_path}")
			continue

		for file_name in os.listdir(folder_path):
			file_path = os.path.join(folder_path, file_name)

			try:
				image = Image.open(file_path)

				if USE_GRAYSCALE:
					image = image.convert("L")
				else:
					image = image.convert("RGB")

				image = image.resize((IMG_SIZE, IMG_SIZE))

				image_array = np.array(image, dtype=np.float32)

				# Normalize
				image_array /= 255.0

				# Add grayscale channel
				if USE_GRAYSCALE:
					image_array = np.expand_dims(image_array, axis=-1)

				images.append(image_array)
				labels.append(class_index)

			except Exception as e:
				print(f"Error loading {file_path}: {e}")

	images = np.array(images)
	labels = np.array(labels)

	return images, labels


# =========================================================
# CNN MODEL
# =========================================================


def build_model(input_shape):
	model = Sequential()

	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))

	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))

	model.add(Flatten())

	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(NUM_CLASSES, activation='softmax'))

	optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

	model.compile(
		optimizer=optimizer,
		loss='categorical_crossentropy',
		metrics=['accuracy']
	)

	return model


# =========================================================
# PLOTS
# =========================================================


def plot_training(history):
	# Accuracy Plot
	plt.figure(figsize=(8, 5))
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model Accuracy')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend(['Train', 'Validation'])
	plt.grid(True)
	plt.show()

	# Loss Plot
	plt.figure(figsize=(8, 5))
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend(['Train', 'Validation'])
	plt.grid(True)
	plt.show()


# =========================================================
# CONFUSION MATRIX
# =========================================================


def plot_confusion_matrix(cm):
	plt.figure(figsize=(6, 6))
	plt.imshow(cm)
	plt.title("Confusion Matrix")
	plt.colorbar()

	plt.xticks(range(NUM_CLASSES), CLASS_NAMES)
	plt.yticks(range(NUM_CLASSES), CLASS_NAMES)

	for i in range(NUM_CLASSES):
		for j in range(NUM_CLASSES):
			plt.text(j, i, str(cm[i][j]), ha='center', va='center')

	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	plt.show()


# =========================================================
# SINGLE IMAGE PREDICTION
# =========================================================


def predict_single_image(model, image_path):
	try:
		image = Image.open(image_path)

		if USE_GRAYSCALE:
			image = image.convert("L")
		else:
			image = image.convert("RGB")

		image = image.resize((IMG_SIZE, IMG_SIZE))

		image_array = np.array(image, dtype=np.float32)
		image_array /= 255.0

		if USE_GRAYSCALE:
			image_array = np.expand_dims(image_array, axis=-1)

		image_array = np.expand_dims(image_array, axis=0)

		prediction = model.predict(image_array, verbose=0)

		predicted_class = np.argmax(prediction)
		confidence = np.max(prediction) * 100

		print("\nPrediction Result")
		print("-----------------")
		print(f"Class: {CLASS_NAMES[predicted_class]}")
		print(f"Confidence: {confidence:.2f}%")

	except Exception as e:
		print(f"Prediction Error: {e}")


# =========================================================
# MAIN
# =========================================================


def main():
	print("Loading Dataset...")

	X, y = load_images()

	print(f"Total Images: {len(X)}")

	y = to_categorical(y, NUM_CLASSES)

	X_train, X_temp, y_train, y_temp = train_test_split(
		X,
		y,
		test_size=0.30,
		random_state=42,
		shuffle=True
	)

	X_val, X_test, y_val, y_test = train_test_split(
		X_temp,
		y_temp,
		test_size=0.50,
		random_state=42,
		shuffle=True
	)

	print(f"Training Images: {len(X_train)}")
	print(f"Validation Images: {len(X_val)}")
	print(f"Testing Images: {len(X_test)}")

	if USE_GRAYSCALE:
		input_shape = (IMG_SIZE, IMG_SIZE, 1)
	else:
		input_shape = (IMG_SIZE, IMG_SIZE, 3)

	model = build_model(input_shape)

	model.summary()

	# Data Augmentation
	data_generator = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.2,
		horizontal_flip=True,
		brightness_range=[0.8, 1.2]
	)

	if TRAIN_MODEL:
		print("\nTraining Model...")

		history = model.fit(
			data_generator.flow(X_train, y_train, batch_size=BATCH_SIZE),
			epochs=EPOCHS,
			validation_data=(X_val, y_val)
		)

		print("\nSaving Model...")
		model.save(MODEL_SAVE_PATH)

		plot_training(history)

	else:
		print("\nLoading Saved Model...")
		model = load_model(MODEL_SAVE_PATH)

	# =====================================================
	# TESTING
	# =====================================================

	print("\nEvaluating Model...")

	test_loss, test_accuracy = model.evaluate(X_test, y_test)

	print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
	print(f"Test Loss: {test_loss:.4f}")

	# Predictions
	predictions = model.predict(X_test)

	y_pred = np.argmax(predictions, axis=1)
	y_true = np.argmax(y_test, axis=1)

	# Confusion Matrix
	cm = confusion_matrix(y_true, y_pred)

	plot_confusion_matrix(cm)

	# Classification Report
	print("\nClassification Report")
	print("---------------------")
	print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

	# =====================================================
	# SINGLE IMAGE TEST
	# =====================================================

	if TEST_SINGLE_IMAGE:
		predict_single_image(model, SINGLE_IMAGE_PATH)


if __name__ == "__main__":
	main()