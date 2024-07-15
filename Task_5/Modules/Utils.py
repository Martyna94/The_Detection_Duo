import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
import pennylane.numpy as np
from sklearn.model_selection import train_test_split


def check_image_shapes(data_dir, json_path):
    """
    Checks if all images specified in a JSON file have the same shape and logs differing shapes.

    Parameters:
        json_path (str): Path to the JSON file containing image paths and labels.
        data_dir (str): Base directory containing the images listed in the JSON file.

    Returns:
        list: A list of tuples with image paths and their unique shapes that differ from the first image's shape.
    """
    with open(json_path, 'r') as file:
        image_data = json.load(file)

    reference_shape = None
    different_shapes = []

    for img_path, label in image_data.items():
        full_path = os.path.join(data_dir, img_path)
        try:
            with Image.open(full_path) as img:
                img_shape = img.size

                if reference_shape is None:
                    reference_shape = img_shape
                    different_shapes.append((img_path, img_shape))
                elif img_shape != reference_shape:
                    different_shapes.append((img_path, img_shape))
        except IOError:
            print(f"Error opening image: {full_path}")

    return different_shapes


def count_images_per_label(dataset):
    """
    Counts the number of images for each label in the dataset.

    Parameters:
        dataset (list of tuples): A list where each tuple contains an image array and a label.

    Returns:
        dict: A dictionary with labels as keys and the count of images for each label as values.
    """
    label_counts = {}
    for _, label in dataset:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    return label_counts


def display_images_by_class(data, images_per_class):
    """
    Display a specified number of images for each unique class in the dataset.

    Parameters:
        data (list of tuples): A list where each tuple contains an image array and its corresponding class label.
        images_per_class (int): The number of images to display for each class.

    This function sorts through the provided data, organizes images by their class labels,
    and displays the specified number of images for each class in separate subplot rows.
    """
    unique_labels = sorted(set(label for _, label in data))

    fig, axes = plt.subplots(nrows=len(unique_labels), ncols=images_per_class, figsize=(15, 5 * len(unique_labels)))

    for row_idx, label in enumerate(unique_labels):
        class_images = [img for img, lbl in data if lbl == label][:images_per_class]

        for col_idx, image in enumerate(class_images):
            ax = axes[row_idx, col_idx] if len(unique_labels) > 1 else axes[col_idx]

            ax.imshow(image, cmap='gray')
            ax.set_title(f'Label {label}')
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def prepare_data(dataset):
    """
    Shuffles the dataset and separates it into flattened features and labels.

    Parameters:
        dataset (list of tuples): Each tuple contains an image data array and a corresponding label.

    Returns:
        tuple: Two lists, where the first list contains the flattened features (image data arrays)
               and the second list contains the labels..
    """
    random.shuffle(dataset)

    X_set = [data[0].flatten() for data in dataset]
    Y_set = [data[1] for data in dataset]

    return X_set, Y_set


def apply_pca(dataset, n_components):
    """
    Applies PCA to reduce the dimensionality of the given dataset to the specified number of components.

    Parameters:
        dataset (array): The dataset to be transformed, where each row represents a sample
                            and each column represents a feature.
        n_components (int): The number of principal components to retain in the transformation.

    Returns:
        array: The dataset transformed into a reduced dimensional space.
    """
    pca = PCA(n_components=n_components)
    pca.fit(dataset)
    data_reduced = pca.transform(dataset)

    return data_reduced


def split_data_into_train_val_test(X, Y, train_ratio, val_ratio, test_ratio):
    """
    Splits the dataset into training, validation, and testing sets using scikit-learn's train_test_split function,
    ensuring the data is shuffled to avoid bias from sorted labels.

    Parameters:
        X (array-like): Feature dataset where each row represents a sample.
        Y (array-like): Label dataset corresponding to the features in X.
        train_ratio (float): Proportion of the dataset to include in the train split.
        val_ratio (float): Proportion of the dataset to include in the validation split.

    Returns:
        tuple: Contains training, validation, and testing datasets:
               (X_train, Y_train, X_val, Y_val, X_test, Y_test)
    """
    if train_ratio + val_ratio + test_ratio > 1:
        raise ValueError("The sum of train_ratio and val_ratio cannot exceed 1.")

    # First, split into training + validation and test
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=test_ratio, shuffle=True, random_state=42)

    # Split training + validation into training and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=val_ratio_adjusted, shuffle=True, random_state=42)

    Y_train = np.array(Y_train)
    Y_val = np.array(Y_val)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def evaluate_model(params, X_set, Y_set, circuit):
    """
    Evaluate the model on validation/test data and return the accuracy.

    Parameters:
        params: The parameters for the quantum circuit.
        X_set:  Set images.
        Y_set:  Set labels.
        circuit: The quantum circuit function that makes predictions.
    Returns:
        Accuracy percentage of the model on the validation/test set.
    """
    test_correct = 0
    for img, true_label in zip(X_set, Y_set):
        predictions = circuit(img, params)
        predicted_label = np.argmax(predictions)

        if predicted_label == true_label:
            test_correct += 1

    accuracy_value = 100 * test_correct / len(X_set)
    return accuracy_value


def costfunc_cross_entropy(params, X_set, Y_set, circuit):
    """
    Compute the cross-entropy loss for a set of training data.
    Parameters:
        params: The parameters for the quantum circuit.
        X_set: Set images.
        Y_set: Set labels.
        circuit: he quantum circuit function that outputs probabilities.
    Returns:
        The average cross-entropy loss over the exact set
    """
    y_true = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0]
    ])

    cost = 0
    len_X_set = len(X_set)

    for i in range(len_X_set):
        prob = circuit(X_set[i], params)

        cost -= 1 / len_X_set * np.sum(y_true[Y_set[i]] * np.log(prob + 1e-8))  # Added small constant for numerical stability

    return cost

# %%
