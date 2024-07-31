import os
import json
from typing import List, Tuple, Any

from PIL import Image
import numpy as np
from numpy import ndarray


class ImageLoader:
    DEFAULT_LABELS = ["good weld", "burn through", "contamination", "lack of fusion", "misalignment",
                      "lack of penetration"]

    def __init__(self, data_dirs, json_paths, image_size, label_names=None):
        """
        A class to load and process images from a directory using specified settings.

        Attributes:
            json_paths (list): A list of paths to JSON files containing image paths and labels.
            data_dirs (list): A list of directories corresponding to the JSON paths.
            image_size (tuple): Desired size to which each image will be resized (width, height).
            label_names (list of str, optional): List of strings representing label categories

        Methods:
            load_images(image_limit=None, **crop_params): Loads images with optional limiting and custom cropping.
            load_data_with_limit(image_limit, crop_params): Helper method to load images with a specified limit.
            load_all_data(crop_params): Helper method to load all images without limit.
            crop_image_region(image, **kwargs): Crops the image based on provided parameters.

        Example:
            >>> loader = ImageLoader(['./data'], ['./labels.json'], (256, 256))
            >>> data_with_limit = loader.load_images(image_limit=200, vertical_start_split=0.3, horizontal_start_split=0.2)
            >>> data_no_limit = loader.load_images(vertical_end_split=0.8, horizontal_end_split=0.7)
            >>> cropped_image_array = ImageLoader.crop_image_region(image_array, 0.25, 0.75, 0.1, 0.9)
        """
        if len(json_paths) != len(data_dirs):
            raise ValueError("The number of JSON paths must match the number of data directories")

        for path in json_paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"The JSON path {path} does not exist")

        for dir in data_dirs:
            if not os.path.isdir(dir):
                raise FileNotFoundError(f"The directory {dir} does not exist")

        self.data_dirs = data_dirs
        self.json_paths = json_paths
        self.image_size = image_size
        self.label_names = label_names if label_names is not None else ImageLoader.DEFAULT_LABELS
        self.label_num = len(self.label_names)

    def _load_data_with_limit(self, image_limit, resize, crop, crop_params, random_seed=0):
        """
        Load images with a limit per label, applying cropping with specified parameters.

        Parameters:
           image_limit (int): The maximum number of images to load per label.
           crop_params (dict): Parameters for the cropping function.

        Returns:
           list: A list of image data and labels tuple(image,label), constrained by the image limit.
       """

        image_paths_by_label = {label: [] for label in range(self.label_num)}

        # Load and merge data from both JSONs
        for json_path, data_dir in zip(self.json_paths, self.data_dirs):
            with open(json_path, 'r') as f:
                data_json = json.load(f)
            for key, value in data_json.items():
                full_path = os.path.join(data_dir,key)
                if value >= self.label_num:
                    continue
                image_paths_by_label[value].append(full_path)

        all_data: list[tuple[ndarray, Any]] = []
        label_count = {label: 0 for label in range(self.label_num)}

        # Set the seed
        np.random.seed(random_seed)

        for label, paths in image_paths_by_label.items():
            # Random Selection Using np.random.choice: This function is used to randomly select indices from the list
            # of image paths, ensuring that the selection is uniformly random without the need to shuffle the entire
            # list.
            if len(paths) > image_limit:
                selected_indices = np.random.choice(len(paths), image_limit, replace=False)
                selected_paths = [paths[i] for i in selected_indices]
            else:
                selected_paths = paths

            for i, img_path in enumerate(selected_paths):
                if not os.path.isfile(img_path):
                    continue

                try:
                    image = Image.open(img_path).convert('L')
                    img_array = np.array(image)
                    if crop:
                        img_array = self.crop_image_region(img_array, **crop_params)
                    if resize:
                        img_array = self.resize_image_array(img_array, self.image_size)

                    all_data.append((np.array(img_array, dtype=np.int32), label))
                    label_count[label] = i+1

                    if len(all_data) % 100 == 0:
                        print(f"Images loaded: {len(all_data)}, Current label: {label}, Total per label: {label_count}", end='\r', flush=True)

                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

        print('\n')
        print(f"Total images loaded : {len(all_data)}")
        print(f"Label count : {label_count}", end='\r', flush=True)
        return all_data

    def _load_all_data(self, resize, crop, crop_params):
        """
        Helper method to load all available images without any limit on the number of images per label, applying cropping with specified parameters.

        Parameters:
            crop_params (dict): Parameters for the cropping function.

        Returns:
            list: A list of all image data and labels.
        """
        all_data = []
        label_count = {label: 0 for label in range(self.label_num)}
        with open(self.json_path, 'r') as f:
            train_data_json = json.load(f)

        for folder in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    key_name = folder + '/' + img_name
                    label = train_data_json.get(key_name, -1)

                    if label == -1:
                        continue

                    image = Image.open(img_path)
                    image = image.convert('L')
                    img_array = np.array(image)
                    if crop:
                        img_array = self.crop_image_region(img_array, **crop_params)

                    if resize:
                        img_array = self.resize_image_array(img_array, self.image_size)

                    all_data.append((np.array(img_array, dtype=np.int32), label))
                    label_count[label] += 1
                    if len(all_data) % 100 == 0:
                        print(f"Total images loaded : {len(all_data)}, Label count : {label_count}", end='\r')
        print(f"Total images loaded : {len(all_data)}, Label count : {label_count}", end='\r')
        return all_data

    @staticmethod
    def crop_image_region(image, vertical_start_split=0.5, vertical_end_split=1.0,
                          horizontal_start_split=0.25, horizontal_end_split=0.75):
        """
        Crops a specified region from the given image based on provided vertical and horizontal splits.

        Parameters:
            image (numpy.ndarray): The image array to crop.
            vertical_start_split (float): Fraction of the height from the top to start cropping.
            vertical_end_split (float): Fraction of the height from the top to end cropping.
            horizontal_start_split (float): Fraction of the width from the left to start cropping.
            horizontal_end_split (float): Fraction of the width from the left to end cropping.

        Returns:
            numpy.ndarray: The cropped portion of the image.
        """

        height, width = image.shape[:2]
        vertical_start_index = int(height * vertical_start_split)
        vertical_end_index = int(height * vertical_end_split)
        horizontal_start_index = int(width * horizontal_start_split)
        horizontal_end_index = int(width * horizontal_end_split)

        return image[vertical_start_index:vertical_end_index, horizontal_start_index:horizontal_end_index]

    @staticmethod
    def resize_image_array(img_array, size):
        image = Image.fromarray(img_array)
        return np.array(image.resize(size))

    def load_images(self, image_limit=None, resize=True, crop=True, random_seed = 0, **crop_params):
        """
        Loads images with optional image limiting and customizable cropping parameters.

        Parameters:
            image_limit (int, optional): Maximum number of images to load per label. Defaults to None.
            crop (bool, optional):
            **crop_params: Arbitrary keyword arguments for cropping dimensions.

        Returns:
            list: List of tuples, each containing an image array and its corresponding label.
        """
        if image_limit is not None:
            return self._load_data_with_limit(image_limit, resize, crop, crop_params=crop_params, random_seed = random_seed)
        else:
            return self._load_all_data(resize, crop, crop_params)

    def count_images_per_label_from_json(self):
        """
        Counts the number of images per label from a JSON file that maps image paths to labels.

        Returns:
            dict: A dictionary where keys are label identifiers (integers) and values are the counts of images associated with each label.

        Example:
            >>> image_loader = ImageLoader(['./data'], ['./train.json'], (224, 224))
            >>> label_counts = image_loader.count_images_per_label_from_json()
            >>> print(label_counts)
                {0: 150, 1: 120, 2: 130}  # Example output, actual will vary based on JSON content.

        Notes:
            The JSON file should be in the format `{"image_path.jpg": label_id}`, where `label_id` is an integer.
            This method assumes that all labels in the JSON file are within the range initialized by `self.label_num`.
            If any labels are encountered that are not in the pre-initialized `label_count` dictionary, they will be added dynamically.
        """

        label_count = {label: 0 for label in range(self.label_num)}
        for json_path in self.json_paths:
            with open(json_path, 'r') as json_file:
                data_json = json.load(json_file)

            for _, label in data_json.items():
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1

        print(f'{label_count}')
        return label_count

# %%
