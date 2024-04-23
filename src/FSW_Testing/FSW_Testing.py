import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.nn as F
import torch.utils.data as data
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split
from torch import rand, randint, Tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_dataloaders(classification_datasets_dict, batch_size=32, num_workers=0):
    """
    Converts validation and test datasets in classification_datasets_dict into DataLoader objects.
    
    Parameters:
    classification_datasets_dict (dict): Dictionary with group numbers as keys and tuples of datasets
                                         (train, validation (optional), test) as values.
    batch_size (int): Batch size to be used for the DataLoader.
    num_workers (int): Number of subprocesses to use for data loading.
    
    Returns:
    dict: A dictionary with keys 'validation' and 'test', each containing a list of DataLoader objects
          for the respective datasets. If no validation is present, only 'test' will be included.
    """
    dataloaders = {'test': []}
    has_validation = len(next(iter(classification_datasets_dict.values()))) > 2

    if has_validation:
        dataloaders['validation'] = []

    for group_datasets in classification_datasets_dict.values():
        if has_validation:
            # Add DataLoader for validation dataset
            val_dataset = group_datasets[1]
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
            dataloaders['validation'].append(val_dataloader)
        
        # Add DataLoader for test dataset
        test_dataset = group_datasets[-1]  # Always the last in the tuple
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
        dataloaders['test'].append(test_dataloader)
    
    return dataloaders

def evaluate_model_with_predictions_tensor(dataloaders, model, device, num_classes):
    """
    Evaluates the model on the given dataloaders, returning predictions, logits, probabilities,
    and true labels for each group. Also demonstrates how to compute metrics using torchmetrics.

    Parameters:
    dataloaders (list): A list of DataLoader objects.
    model (torch.nn.Module): The PyTorch model to evaluate.
    device (torch.device): The device to run the evaluation on (e.g., 'cpu' or 'cuda').
    num_classes (int): The number of classes in the classification task.

    Returns:
    dict: A dictionary with group numbers as keys and dictionaries containing 'logits', 'probabilities',
          'true_labels', and 'predictions' lists as values.
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for dataloader in dataloaders:
            for inputs, labels, tasks in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Assuming softmax is applied in the model if needed, else apply it here
                predictions = torch.argmax(outputs, dim=1)

                for i in range(len(tasks)):
                    task = tasks[i].item()  # Assuming task is a tensor, get its Python value
                    if task not in results:
                        results[task] = {
                            'logits': [],
                            'probabilities': [],
                            'max_probability': [],
                            'true_labels': [],
                            'predictions': []
                        }

                    logits = outputs[i]
                    probabilities = torch.softmax(outputs, dim=1)
                    #probability = torch.sigmoid(logits).squeeze(0)  # Adjust dimensions as needed
                    true_label = labels[i]

                    # Append results for the current task
                    results[task]['logits'].append(logits.cpu())
                    results[task]['probabilities'].append(probabilities[i, 1].cpu())
                    results[task]['max_probability'].append(torch.max(probabilities[i, 1].cpu()))
                    #results[task]['probabilities'].append(probability.cpu())
                    #results[task]['max_probability'].append(torch.max(probability).cpu())
                    results[task]['true_labels'].append(true_label.cpu())
                    results[task]['predictions'].append(predictions[i].cpu())

                    # Update metric for the current batch
                    #metric.update(predictions[i].unsqueeze(0), true_label.unsqueeze(0))
    return results

def evaluate_model_with_predictions(dataloaders, model, device, num_classes):
    """
    Evaluates the model on the given dataloaders, returning predictions, logits, probabilities,
    and true labels for each group. Also demonstrates how to compute metrics using torchmetrics.

    Parameters:
    dataloaders (list): A list of DataLoader objects.
    model (torch.nn.Module): The PyTorch model to evaluate.
    device (torch.device): The device to run the evaluation on (e.g., 'cpu' or 'cuda').
    num_classes (int): The number of classes in the classification task.

    Returns:
    dict: A dictionary with group numbers as keys and dictionaries containing 'logits', 'probabilities',
          'true_labels', and 'predictions' lists as values.
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for dataloader in dataloaders:
            for inputs, labels, tasks in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Assuming softmax is applied in the model if needed, else apply it here
                predictions = torch.argmax(outputs, dim=1)

                for i in range(len(tasks)):
                    task = tasks[i].item()  # Assuming task is a tensor, get its Python value
                    if task not in results:
                        results[task] = {
                            'logits': [],
                            'probabilities': [],
                            'max_probability': [],
                            'true_labels': [],
                            'predictions': []
                        }

                    logits = outputs[i]
                    probability = torch.sigmoid(logits).squeeze(0)  # Adjust dimensions as needed
                    true_label = labels[i]

                    # Append results for the current task
                    results[task]['logits'].append(logits.cpu().numpy())
                    results[task]['probabilities'].append(probability.cpu().numpy())
                    results[task]['max_probability'].append(torch.max(probability).cpu().numpy())
                    results[task]['true_labels'].append(true_label.cpu().numpy())
                    results[task]['predictions'].append(predictions[i].cpu().numpy())

                    # Update metric for the current batch
                    #metric.update(predictions[i].unsqueeze(0), true_label.unsqueeze(0))
    return results


