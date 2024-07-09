import os
import json
from PIL import Image
import matplotlib.pyplot as plt


def check_image_shapes(data_dir, json_path):
    """
    Checks if all images specified in a JSON file have the same shape and logs differing shapes.

    Parameters:
        json_path (str): Path to the JSON file containing image paths and labels.
        data_dir (str): Base directory containing the images listed in the JSON file.

    Returns:
        list: A list of tuples with image paths and their unique shapes that differ from the first image's shape.
    """
    # Load JSON data
    with open(json_path, 'r') as file:
        image_data = json.load(file)

    # Initialize variables
    reference_shape = None
    different_shapes = []

    # Process each image
    for img_path, label in image_data.items():
        full_path = os.path.join(data_dir, img_path)
        try:
            with Image.open(full_path) as img:
                # Convert image to array to check its shape
                img_shape = img.size  # This returns (width, height)

                if reference_shape is None:
                    reference_shape = img_shape  # Set the first image's shape as reference
                    different_shapes.append((img_path, img_shape))
                elif img_shape != reference_shape:
                    different_shapes.append((img_path, img_shape))  # Log differing shapes
        except IOError:
            print(f"Error opening image: {full_path}")  # Handle cases where the image can't be opened

    return different_shapes


def count_images_per_label(data_set):
    """
    Counts the number of images for each label in the dataset.

    Parameters:
        data_set (list of tuples): A list where each tuple contains an image array and a label.

    Returns:
        dict: A dictionary with labels as keys and the count of images for each label as values.
    """
    label_counts = {}
    for _, label in data_set:
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
    # Extract unique labels from the data
    unique_labels = sorted(set(label for _, label in data))

    # Setup figure and axes for the image grid
    fig, axes = plt.subplots(nrows=len(unique_labels), ncols=images_per_class, figsize=(15, 5 * len(unique_labels)))

    # Iterate over each class label
    for row_idx, label in enumerate(unique_labels):
        # Select the first few images corresponding to the current label
        class_images = [img for img, lbl in data if lbl == label][:images_per_class]

        # Display each selected image
        for col_idx, image in enumerate(class_images):
            ax = axes[row_idx, col_idx] if len(unique_labels) > 1 else axes[col_idx]

            # Display the image in grayscale
            ax.imshow(image, cmap='gray')
            ax.set_title(f'Label {label}')
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# %%
