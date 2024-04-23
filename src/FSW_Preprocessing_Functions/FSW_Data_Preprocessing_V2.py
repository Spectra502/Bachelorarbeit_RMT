import pandas as pd
import shutil
from pathlib import Path
import torch
from torch.utils.data import Dataset, random_split
import matplotlib.pyplot as plt
import random

def transform_csv_to_df_and_copy_images(csv_path, source_dir=None, dest_dir=None):
    """
    Transforms a CSV file into a pandas DataFrame and copies image directories.
    
    :param str csv_path: Path to the CSV file.
    :param str source_dir: Source directory from which to copy images. (optional)
    :param str dest_dir: Destination directory to copy images to. (optional)
    :return: DataFrame created from the CSV file.
    :rtype: pandas.DataFrame
    """
    # Convert paths from string to Path objects if they are not None
    csv_path = Path(csv_path)
    if source_dir is not None:
        source_dir = Path(source_dir)
    if dest_dir is not None:
        dest_dir = Path(dest_dir)
    
    # Read CSV and convert it into a dataframe
    dataframe = pd.read_csv(csv_path, sep=";")
    
    # Modify the img_path to match the working directory path
    dataframe['img_path'] = dataframe['img_path'].str.replace('..', '.', 1)
    
    # Copy the data into the working directory, if paths are provided
    if source_dir and dest_dir:
        try:
            shutil.copytree(source_dir, dest_dir)
            print(f"Images copied from {source_dir} to {dest_dir}.")
        except FileExistsError:
            print(f"Path {dest_dir} already exists.")
        except Exception as e:
            print(f"An error occurred while copying files: {e}")
    
    return dataframe

def filter_and_sort_by_group(dataframe, selected_columns=None):
    """
    Divides a DataFrame into groups as originally defined in the 'Group' column,
    and optionally filters the DataFrame to only include specified columns, while ensuring
    the 'Group' column is used for segmentation but not necessarily included in the final output.
    
    :param pd.DataFrame dataframe: The input DataFrame.
    :param list selected_columns: List of columns to keep. If None, all columns are kept.
    :return: A dictionary of DataFrames, one for each group, keyed by group number.
    :rtype: dict
    """
    # Ensure 'Group' is available for grouping, even if not in final selection
    necessary_columns = ['Group']
    if selected_columns is not None:
        all_columns = list(set(selected_columns + necessary_columns))
        filtered_df = dataframe[all_columns].copy()
    else:
        filtered_df = dataframe.copy()
    
    max_group = dataframe['Group'].max()
    groups_dict = {}
    
    for group_number in range(1, max_group + 1):
        group_df = filtered_df[filtered_df['Group'] == group_number].reset_index(drop=True)
        
        # Remove 'Group' from the output if it was not specified in selected_columns
        if selected_columns is not None and 'Group' not in selected_columns:
            group_df = group_df[selected_columns]
            
        if not group_df.empty:
            groups_dict[group_number] = group_df
    
    return groups_dict

class Group_Grat_Dataset(Dataset):
    """
    Data loading.
    """
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        """
        len(dataset)
        """
        return len(self.paths)

    def __getitem__(self, idx):
        """
        dataset [0]
        """
        path = self.paths[idx]
        label = self.labels[idx]

        return path, label

def create_group_datasets(groups_dict):
    """
    Creates a Group_Grat_Dataset for each group in the groups_dict.

    :param dict groups_dict: A dictionary of DataFrames, one for each group, keyed by group number.
    :return: A dictionary of Group_Grat_Dataset instances, one for each group, keyed by group number.
    :rtype: dict
    """
    datasets_dict = {}
    for group_number, df in groups_dict.items():
        datasets_dict[group_number] = Group_Grat_Dataset(paths=df['img_path'].tolist(), labels=df['grat_label'].tolist())
    return datasets_dict


def split_datasets(datasets_dict, training=0.7, validation=None, seed=42):
    """
    Splits each dataset in the dictionary into training, optional validation, and testing subsets.
    
    :param dict datasets_dict: A dictionary of datasets to split.
    :param float training: The fraction of the dataset to use for training.
    :param float validation: The fraction of the dataset to use for validation. If None, the
                             remaining data after the training split is used for testing.
    :param int seed: The random seed for reproducible splits.
    :return: A dictionary with keys corresponding to the original dataset keys, each containing a tuple
             of Dataset objects in the order (train, validation (optional), test).
    :rtype: dict
    """
    # Initialize the random seed generator
    generator = torch.Generator().manual_seed(seed)
    
    # Initialize a dictionary to store the split datasets
    split_datasets_dict = {}
    
    for key, dataset in datasets_dict.items():
        dataset_length = len(dataset)
        train_length = int(training * dataset_length)
        
        if validation is not None:
            val_length = int(validation * dataset_length)
            test_length = dataset_length - train_length - val_length
            splits = random_split(dataset, [train_length, val_length, test_length], generator=generator)
            split_datasets_dict[key] = (splits[0], splits[1], splits[2])
        else:
            test_length = dataset_length - train_length
            splits = random_split(dataset, [train_length, test_length], generator=generator)
            split_datasets_dict[key] = (splits[0], splits[1])
    
    return split_datasets_dict

def plot_dataset_splits_with_counts(split_datasets_dict):
    """
    Plots the structure of the dataset splits as a family tree, including the number of elements
    in each dataset.

    :param dict split_datasets_dict: The dictionary containing the split datasets.
    """
    fig, axes = plt.subplots(nrows=len(split_datasets_dict), figsize=(6, len(split_datasets_dict) * 2.5))
    
    if len(split_datasets_dict) == 1:
        axes = [axes]  # Adjust if there's only one group for consistent indexing
    
    for ax, (key, splits) in zip(axes, split_datasets_dict.items()):
        lengths = [len(split) for split in splits]
        labels = ['Training', 'Validation', 'Testing'][:len(splits)]
        # Include the absolute count in the labels
        labels_with_counts = [f'{label}\n({length})' for label, length in zip(labels, lengths)]
        
        # Title for each subplot
        ax.set_title(f'Group {key} Splits')
        ax.pie(lengths, labels=labels_with_counts, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
        ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    
    plt.tight_layout()
    plt.show()


def convert_dataset_to_list(split_datasets_dict):
    """
    Converts datasets within the split_datasets_dict into lists of tuples containing the data.

    :param dict split_datasets_dict: The dictionary containing the split datasets, as returned by split_datasets.
    :return: A dictionary with the same keys as split_datasets_dict, where each value is a list of tuples
             containing the data from the respective dataset.
    :rtype: dict
    """
    list_datasets_dict = {}

    for group, datasets in split_datasets_dict.items():
        list_datasets = []
        for dataset in datasets:
            # Convert each dataset to a list of tuples (path, label)
            dataset_list = [dataset[idx] for idx in range(len(dataset))]
            list_datasets.append(dataset_list)
        list_datasets_dict[group] = tuple(list_datasets)

    return list_datasets_dict


def create_paths_datasets_from_list_v2(list_datasets_dict, transforms_dict, root=None):
    """
    Creates PathsDataset instances from a dictionary of lists of data tuples. Dynamically handles
    different combinations of data splits (e.g., only train and test, or train, validation, and test).
    
    :param dict list_datasets_dict: Dictionary containing lists of data tuples for different splits.
    :param dict transforms_dict: Dictionary containing transformations for each split, e.g., 'train', 'validation', 'test'.
    :param str root: Root directory for the datasets. (optional)
    :return: A dictionary with the same structure as list_datasets_dict, where each list of data tuples is
             replaced with a corresponding PathsDataset instance, equipped with the appropriate transform.
    :rtype: dict
    """
    paths_datasets_dict = {}
    
    # Determine the available splits based on transforms_dict keys
    available_splits = list(transforms_dict.keys())
    
    for group, data_lists in list_datasets_dict.items():
        datasets = []
        for i, data_list in enumerate(data_lists):
            # Ensure we do not exceed the number of available transforms
            if i < len(available_splits):
                transform_key = available_splits[i]
                transform = transforms_dict.get(transform_key)
                dataset = PathsDataset(root=root, files=data_list, transform=transform)
                datasets.append(dataset)
        paths_datasets_dict[group] = tuple(datasets)
    
    return paths_datasets_dict
