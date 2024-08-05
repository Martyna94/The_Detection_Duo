# /content/untitled.py 
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
# Add this import statement
import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt


def get_dataset(n_train, n_test):
    """
    Prepares the training and test dataset

    Parameters:
        n_train (int): An int value for the number of samples in the training set.
        n_test (int): An int value for the number of samples in the testing set.

    Returns:
        Array: 4 numpy arrays X_train, Y_train, X_test, Y_test
    """
    mnist_dataset = keras.datasets.mnist
    (X_train_full, Y_train_full), (X_test_full, Y_test_full) = mnist_dataset.load_data()

    X_train = X_train_full[:n_train]
    Y_train = Y_train_full[:n_train]
    X_test = X_test_full[:n_test]
    Y_test = Y_test_full[:n_test]
    
    # Convert to float32 before division to allow for floating-point results
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)

    #normalize the values of the pixels in the image to be in the range 0<= value <=1
    X_train /= 255
    X_test /= 255

    #expand the dimensions of the images, add new axis
    X_train = np.array(X_train[..., tf.newaxis], requires_grad=False)
    X_test = np.array(X_test[..., tf.newaxis], requires_grad=False)

    return X_train, Y_train, X_test, Y_test


def quanv(image, circuit, stride = 2):
    """
    Extracts features from the image using quantum convolution

    Parameters:
        image (array): A numpy array of an image.
        stride (int): An int used to choose a stridexstride square kernel.
        circuit (qnode): pennylane circuit to perform convolution.

    Returns:
        Array: A numpy array with decreased dimensions of the image.
    """
    dimensions = image.shape
    out = np.zeros((int(dimensions[0]/stride), int(dimensions[1]/stride), stride*stride))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, dimensions[0], stride):
        for k in range(0, dimensions[1], stride):
            # Process a squared 2x2 region of the image with a quantum circuit
            res = []
            for s1 in range(stride):
              for s2 in range(stride):
                res.append(image[j+s1, k+s2, 0])
            q_results = circuit(
                res
            )
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            for c in range(stride*stride):
                out[j // stride, k // stride, c] = q_results[c]
    return out


def MyModel(classes = 10):
    """
    Build a keras model for image classification

    Parameters:
        classes (int): number of layers in the end for softmax classification.

    Returns (keras): A compiled keras model for classification
    """
        
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(classes, activation="softmax")
    ])

    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def prepare_images(X_train, X_test, circuit, stride = 2):
  """
    performs convolution on the train and test datasets

    Parameters:
        X_train (Array): numpy array of the train images.
        X_test (Array): numpy array of the test images.

    Returns (Array): 2 numpy arrays of the train and test datasets after convolution is applied. 
  """
  q_train_images = []
  n_train = len(X_train)
  print("Quantum pre-processing of train images:")

  for idx, img in enumerate(X_train):
      print("{}/{}".format(idx + 1, n_train), end="\n")
      q_train_images.append(quanv(img, circuit, stride))
  
  q_test_images = []
  n_test = len(X_test)
  print("Quantum pre-processing of test images:")

  for idx, img in enumerate(X_test):
      print("{}/{}".format(idx + 1, n_test), end="\n")
      q_test_images.append(quanv(img, circuit, stride))
  
  return np.array(q_train_images), np.array(q_test_images)