import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator

import torch
import torch.nn as nn
import torch.nn as F
import torch.utils.data as data
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split

from torch import tensor
from torchmetrics.classification import BinaryCalibrationError, BinaryAccuracy, BinaryF1Score, BinaryHammingDistance, BinaryJaccardIndex




def plot_calibration_curves(results, n_bins=2):
    """
    Plots the calibration curves for each group in the results dictionary in a single window.

    Parameters:
    results (dict): The dictionary containing true labels and predictions for each group.
    n_bins (int): Number of bins to use for the calibration curve.
    """
    plt.figure(figsize=(10, 8))

    for group, data in results.items():
        true_labels = data['true_labels']
        predictions = data['probabilities']  # Assuming predictions are probabilities

        fraction_of_positives, mean_predicted_value = calibration_curve(true_labels, predictions, n_bins=n_bins)

        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"Group {group}")

    # Reference line for a perfectly calibrated model
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    plt.ylabel("Fraction of positives")
    plt.xlabel("Mean predicted value")
    plt.title('Calibration Curves')
    plt.legend(loc='best')
    plt.show()


def plot_calibration_curves_matrix(results, n_bins=2):
    """
    Plots the calibration curves for each group in the results dictionary in a matrix-like arrangement.

    Parameters:
    results (dict): The dictionary containing true labels and predictions for each group.
    n_bins (int): Number of bins to use for the calibration curve.
    """
    num_groups = len(results)
    num_cols = 3  # Number of columns in the matrix-like arrangement

    # Calculate the number of rows needed to accommodate all groups
    num_rows = (num_groups + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))

    for idx, (group, data) in enumerate(results.items()):
        row = idx // num_cols
        col = idx % num_cols

        ax = axes[row, col] if num_groups > 1 else axes  # For single group, axes is not a 2D array

        true_labels = data['true_labels']
        predictions = data['probabilities']  # Assuming predictions are probabilities

        fraction_of_positives, mean_predicted_value = calibration_curve(true_labels, predictions, n_bins=n_bins)

        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"Group {group}")
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

        ax.set_ylabel("Fraction of positives")
        ax.set_xlabel("Mean predicted value")
        ax.set_title(f'Calibration Curve - Group {group}')
        ax.legend(loc='best')

    # Adjust layout to prevent overlapping titles
    plt.tight_layout()
    plt.show()

def print_metric_results(dictionary):
    metric_dict = {}
    for group in dictionary.keys():
        metric_dict[group] = {}
        preds = torch.stack(dictionary[group]["probabilities"])
        target = torch.stack(dictionary[group]["true_labels"])
        #target = torch.stack(results_pred_tensor[group]["true_labels"])
        print(f"Group {group}")
        
        bce_l1 = BinaryCalibrationError(n_bins=2, norm='l1')
        print(f"Norm L1 Error: {bce_l1(preds, target)}")
        metric_dict[group]["ece"] = bce_l1(preds, target)
        
        bce_l2 = BinaryCalibrationError(n_bins=2, norm='l2')
        print(f"Norm L2 Error: {bce_l2(preds, target)}")
        metric_dict[group]["mce"] = bce_l2(preds, target)
        
        bce_max = BinaryCalibrationError(n_bins=2, norm='max')
        print(f"Norm Max Error: {bce_max(preds, target)}")
        metric_dict[group]["rmsce"] = bce_max(preds, target)
        
        metric_acc = BinaryAccuracy()
        print(f"Accuracy: {metric_acc(preds, target)}")
        metric_dict[group]["acc"] = metric_acc(preds, target)
        
        metric_f1 = BinaryF1Score()
        print(f"F1-Score: {metric_f1(preds, target)}")
        metric_dict[group]["f1"] = metric_f1(preds, target)
        
        metric_hamming = BinaryHammingDistance()
        print(f"Hamming: {metric_hamming(preds, target)}")
        metric_dict[group]["hamming"] = metric_hamming(preds, target)
    return metric_dict

def compare_metric_results(dictionary1, dictionary2):
    groups = []
    metrics = ["ece", "mce", "rmsce", "acc", "f1", "hamming"]
    differences = {metric: [] for metric in metrics}
    colors = {metric: [] for metric in metrics}

    for group in dictionary1.keys():
        groups.append(group)
        for metric in metrics:
            diff = dictionary2[group][metric] - dictionary1[group][metric]
            differences[metric].append(abs(diff))
            colors[metric].append('red' if diff < 0 else 'green')

    # Plotting
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.flatten()

    for i, metric in enumerate(metrics):
        axs[i].bar(groups, differences[metric], color=colors[metric])
        axs[i].set_title(f"Difference in {metric}")
        axs[i].set_xlabel("Group")
        axs[i].set_ylabel("Absolute Difference")

    plt.tight_layout()
    plt.show()
