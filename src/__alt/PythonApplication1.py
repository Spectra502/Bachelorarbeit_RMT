import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from pathlib import Path
#from skimage import io, transform
from PIL import Image
from tqdm.auto import tqdm
from typing import Dict,List
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark, dataset_benchmark, filelist_benchmark, dataset_benchmark, tensors_benchmark, paths_benchmark
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import EWC, MAS, DER, PNNStrategy, Naive
from avalanche.training.checkpoint import maybe_load_checkpoint, save_checkpoint
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.benchmarks.utils import AvalancheDataset, make_avalanche_dataset, make_detection_dataset, make_classification_dataset, make_tensor_classification_dataset, PathsDataset

import zipfile
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg
import copy

from avalanche.benchmarks.datasets import MNIST
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available())
print(device)