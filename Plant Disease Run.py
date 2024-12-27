import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model('plant_disease_model1.h5')

# Define constants
IMAGE_SIZE = (416, 416)
data_dir = 'C:/Users/user/Downloads/Plant Disease/Test/Test'
categories = ['healthy', 'powdery', 'rust']

# Prepare the test data
test_images = []
actual_labels = []
image_paths = []

for category in categories:
    path = os.path.join(data_dir, category)
    label = categories.index(category)  # Encode labels as 0, 1, 2
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        test_images.append(img_array)
        actual_labels.append(label)
        image_paths.append(img_path)

# Convert to numpy array and normalize
test_images = np.array(test_images) / 255.0
actual_labels = np.array(actual_labels)

# Make predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Print actual and predicted labels along with images
plt.figure(figsize=(15, 15))
num_images = len(test_images)
cols = 5  # Number of columns in the grid
rows = (num_images + cols - 1) // cols  # Calculate number of rows needed

for i in range(num_images):
    plt.subplot(rows, cols, i + 1)  # Dynamic subplot
    plt.imshow(test_images[i])
    plt.title(f'Actual: {categories[actual_labels[i]]}\nPredicted: {categories[predicted_labels[i]]}')
    plt.axis('off')

plt.tight_layout()
plt.show()
