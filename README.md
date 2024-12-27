# Plant-Disease-Detection-using-CNN
This project trains a convolutional neural network (CNN) using TensorFlow to classify plant disease images into three categories: *healthy*, *powdery*, and *rust*. The model is trained on preprocessed images, then used to predict plant disease on new test images. Results are visualized by comparing predicted and actual labels.
#Features:
Image Preprocessing: Loads and normalizes plant disease images.
Model Architecture: A CNN with three convolutional layers, max-pooling, and fully connected layers.
Training: The model is trained on labeled data for 10 epochs with a batch size of 8.
Prediction: Classifies new plant images and compares predicted vs. actual labels.
#Requirements:
TensorFlow
NumPy
Matplotlib
scikit-learn
# Setup:
Clone the repository.
Install dependencies: pip install -r requirements.txt.
Load your dataset into the specified directories.
Run the training script to train the model.
Use the prediction script to classify new images.
# Usage:
The trained model is saved as plant_disease_model1.h5.
Visualize training results and make predictions on new plant disease images.
# License:
None
