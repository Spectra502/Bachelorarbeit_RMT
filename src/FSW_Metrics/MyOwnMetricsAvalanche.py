import torch
from avalanche.evaluation import Metric
from torchmetrics.classification import Accuracy as TM_Acc
from torchmetrics.classification import AUROC as TM_AUR
from torchmetrics.classification import CalibrationError as TM_ECE
from torchmetrics.classification import F1Score as TM_F1
from torchmetrics.classification import JaccardIndex as TM_Jaccard
from torchmetrics.classification import HammingDistance as TM_Hamming
from torch.nn.functional import softmax
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split
from torch import rand, randint

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from pathlib import Path
#from skimage import io, transform
from PIL import Image
from tqdm.auto import tqdm
from typing import Dict,List

from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark, dataset_benchmark, filelist_benchmark, dataset_benchmark, tensors_benchmark, paths_benchmark
from avalanche.evaluation import Metric, PluginMetric
from avalanche.evaluation import Metric
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import EWC, MAS, DER, PNNStrategy, Naive
from avalanche.training.checkpoint import maybe_load_checkpoint, save_checkpoint
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.benchmarks.utils import AvalancheDataset, make_avalanche_dataset, make_detection_dataset, make_classification_dataset, make_tensor_classification_dataset, PathsDataset
from avalanche.evaluation import PluginMetric, Metric
#from avalanche.evaluation.metrics import Accuracy
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name

from torchmetrics.classification import BinaryAccuracy, Accuracy
from typing import Any, Optional, Sequence, Type, Union
import torchmetrics
from torchmetrics.classification import BinaryAccuracy, CalibrationError
from torchmetrics.classification import BinaryCalibrationError

#from torch import Tensor
from torch import Tensor
from typing_extensions import Literal

from torchmetrics.classification.base import _ClassificationTaskWrapper
from torchmetrics.classification.stat_scores import BinaryStatScores, MulticlassStatScores, MultilabelStatScores
from torchmetrics.functional.classification.accuracy import _accuracy_reduce
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import ClassificationTask
from torchmetrics.utilities.imports import _MATPLOTLIB_AVAILABLE
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE

from avalanche.evaluation import Metric

device = "cuda" if torch.cuda.is_available() else "cpu"

class AccMetric(Metric[float]):
    """
    This metric will return a `float` value
    """
    def __init__(self):
        """
        Initialize your metric here
        """
        super().__init__()
        self._torchmetrics_acc = TM_Acc(task="binary").to(device)
        
    @torch.no_grad()
    def update(self, predicted_y:Tensor, true_y:Tensor) -> None:
        """
        Update metric value here
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        self._torchmetrics_acc.update(predicted_y, true_y)

    def result(self) -> float:
        """
        Emit the metric result here
        """
        return self._torchmetrics_acc.compute().item()

    def reset(self):
        """
        Reset your metric here
        """
        return self._torchmetrics_acc.reset()

class AUROCMetric(Metric[float]):
    """
    This metric will return a `float` value
    """
    def __init__(self):
        """
        Initialize your metric here
        """
        super().__init__()
        self._torchmetrics_AUROC = TM_AUR(task="binary").to(device)
        
    @torch.no_grad()
    def update(self, predicted_y:Tensor, true_y:Tensor) -> None:
        """
        Update metric value here
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            #predicted_y = torch.max(predicted_y, 1)[1]
            predicted_y = softmax(predicted_y, dim=1)
            predicted_y = predicted_y[:,1]
            #print(predicted_y)

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        self._torchmetrics_AUROC.update(predicted_y, true_y)

    def result(self) -> float:
        """
        Emit the metric result here
        """
        return self._torchmetrics_AUROC.compute().item()

    def reset(self):
        """
        Reset your metric here
        """
        return self._torchmetrics_AUROC.reset()

class ECEMetric(Metric[float]):
    """
    This metric will return a `float` value
    """
    def __init__(self):
        """
        Initialize your metric here
        """
        super().__init__()
        self._torchmetrics_ECE = TM_ECE(task="binary", n_bins=15, norm="l1").to(device)
        
    @torch.no_grad()
    def update(self, predicted_y:Tensor, true_y:Tensor) -> None:
        """
        Update metric value here
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            #predicted_y = torch.max(predicted_y, 1)[1]
            predicted_y = softmax(predicted_y, dim=1)
            predicted_y = predicted_y[:,1]
            #print(predicted_y)

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        self._torchmetrics_ECE.update(predicted_y, true_y)

    def result(self) -> float:
        """
        Emit the metric result here
        """
        return self._torchmetrics_ECE.compute().item()

    def reset(self):
        """
        Reset your metric here
        """
        return self._torchmetrics_ECE.reset()

class F1Metric(Metric[float]):
    """
    This metric will return a `float` value
    """
    def __init__(self):
        """
        Initialize your metric here
        """
        super().__init__()
        self._torchmetrics_F1 = TM_F1(task="binary").to(device)
        
    @torch.no_grad()
    def update(self, predicted_y:Tensor, true_y:Tensor) -> None:
        """
        Update metric value here
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        self._torchmetrics_F1.update(predicted_y, true_y)

    def result(self) -> float:
        """
        Emit the metric result here
        """
        return self._torchmetrics_F1.compute().item()

    def reset(self):
        """
        Reset your metric here
        """
        return self._torchmetrics_F1.reset()

class JaccardMetric(Metric[float]):
    """
    This metric will return a `float` value
    """
    def __init__(self):
        """
        Initialize your metric here
        """
        super().__init__()
        self._torchmetrics_Jaccard = TM_Jaccard(task="binary").to(device)
        
    @torch.no_grad()
    def update(self, predicted_y:Tensor, true_y:Tensor) -> None:
        """
        Update metric value here
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        self._torchmetrics_Jaccard.update(predicted_y, true_y)

    def result(self) -> float:
        """
        Emit the metric result here
        """
        return self._torchmetrics_Jaccard.compute().item()

    def reset(self):
        """
        Reset your metric here
        """
        return self._torchmetrics_Jaccard.reset()

class HammingMetric(Metric[float]):
    """
    This metric will return a `float` value
    """
    def __init__(self):
        """
        Initialize your metric here
        """
        super().__init__()
        self._torchmetrics_Hamming = TM_Hamming(task="binary").to(device)
        
    @torch.no_grad()
    def update(self, predicted_y:Tensor, true_y:Tensor) -> None:
        """
        Update metric value here
        """
        true_y = torch.as_tensor(true_y)
        predicted_y = torch.as_tensor(predicted_y)

        if len(true_y) != len(predicted_y):
            raise ValueError("Size mismatch for true_y and predicted_y tensors")

        # Check if logits or labels
        if len(predicted_y.shape) > 1:
            # Logits -> transform to labels
            predicted_y = torch.max(predicted_y, 1)[1]

        if len(true_y.shape) > 1:
            # Logits -> transform to labels
            true_y = torch.max(true_y, 1)[1]

        self._torchmetrics_Hamming.update(predicted_y, true_y)

    def result(self) -> float:
        """
        Emit the metric result here
        """
        return self._torchmetrics_Hamming.compute().item()

    def reset(self):
        """
        Reset your metric here
        """
        return self._torchmetrics_Hamming.reset()

from avalanche.evaluation import PluginMetric
#from avalanche.evaluation.metrics import Accuracy
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name


class MyPluginMetric(PluginMetric[float]):
    """
    This metric will return a `float` value after
    each training epoch
    """

    def __init__(self):
        """
        Initialize the metric
        """
        super().__init__()

        self._accuracy_metric = MyStandaloneMetric()

    def reset(self) -> None:
        """
        Reset the metric
        """
        self._accuracy_metric.reset()

    def result(self) -> float:
        """
        Emit the result
        """
        return self._accuracy_metric.result()

    def after_training_iteration(self, strategy: 'PluggableStrategy') -> None:
        """
        Update the accuracy metric with the current
        predictions and targets
        """
        # task labels defined for each experience
        task_labels = strategy.experience.task_labels
        if len(task_labels) > 1:
            # task labels defined for each pattern
            task_labels = strategy.mb_task_id
        else:
            task_labels = task_labels[0]
            
        self._accuracy_metric.update(strategy.mb_output, strategy.mb_y, 
                                     task_labels)

    def before_training_epoch(self, strategy: 'PluggableStrategy') -> None:
        """
        Reset the accuracy before the epoch begins
        """
        self.reset()

    def after_training_epoch(self, strategy: 'PluggableStrategy'):
        """
        Emit the result
        """
        return self._package_result(strategy)
        
        
    def _package_result(self, strategy):
        """Taken from `GenericPluginMetric`, check that class out!"""
        metric_value = self.accuracy_metric.result()
        add_exp = False
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k)
                metrics.append(MetricValue(self, metric_name, v,
                                           plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(self, strategy,
                                          add_experience=add_exp,
                                          add_task=True)
            return [MetricValue(self, metric_name, metric_value,
                                plot_x_position)]

    def __str__(self):
        """
        Here you can specify the name of your metric
        """
        return "Torchmetrics_Acc_Epoch"

class TorchmetricsAUROCPluginMetric(PluginMetric[float]):
    def __init__(self):
        super().__init__()
        self._AUROC_metric = AUROCMetric()

    def reset(self) -> None:
        self._AUROC_metric.reset()

    def result(self) -> float:
        return self._AUROC_metric.result()

    def after_training_iteration(self, strategy):
        self._AUROC_metric.update(strategy.mb_output, strategy.mb_y)

    def before_training_epoch(self, strategy):
        self.reset()

    def after_training_epoch(self, strategy):
        metric_value = self.result()
        metric_name = get_metric_name(self, strategy, add_experience=False, add_task=False)
        return [MetricValue(self, metric_name, metric_value, strategy.clock.train_iterations)]

    def __str__(self):
        return "AUROC_Epoch"

from avalanche.evaluation import PluginMetric, GenericPluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name

class TorchmetricsAccuracyPluginMetric(PluginMetric[float]):
    def __init__(self):
        super().__init__()
        self._accuracy_metric = AccMetric()

    def reset(self) -> None:
        self._accuracy_metric.reset()

    def result(self) -> float:
        return self._accuracy_metric.result()

    def after_training_iteration(self, strategy):
        self._accuracy_metric.update(strategy.mb_output, strategy.mb_y)

    def before_training_epoch(self, strategy):
        self.reset()

    def after_training_epoch(self, strategy):
        metric_value = self.result()
        metric_name = get_metric_name(self, strategy, add_experience=False, add_task=False)
        return [MetricValue(self, metric_name, metric_value, strategy.clock.train_iterations)]

    def __str__(self):
        return "Binary_Accuracy_Epoch"

class TorchmetricsAccuracyGenericPluginMetric(GenericPluginMetric[float, AccMetric]):
    """
    Base class for all accuracies plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        """Creates the Accuracy plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        super().__init__(AccMetric(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)
    


class TorchmetricsECEPluginMetric(PluginMetric[float]):
    def __init__(self):
        super().__init__()
        self._ECE_metric = ECEMetric()

    def reset(self) -> None:
        self._ECE_metric.reset()

    def result(self) -> float:
        return self._ECE_metric.result()

    def after_training_iteration(self, strategy):
        self._ECE_metric.update(strategy.mb_output, strategy.mb_y)

    def before_training_epoch(self, strategy):
        self.reset()

    def after_training_epoch(self, strategy):
        metric_value = self.result()
        metric_name = get_metric_name(self, strategy, add_experience=False, add_task=False)
        return [MetricValue(self, metric_name, metric_value, strategy.clock.train_iterations)]

    def __str__(self):
        return "ECE_Accuracy_Epoch"

class TorchmetricsECEGenericPluginMetric(GenericPluginMetric[float, ECEMetric]):
    """
    Base class for all accuracies plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        """Creates the Accuracy plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        super().__init__(ECEMetric(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)

class TorchmetricsF1PluginMetric(PluginMetric[float]):
    def __init__(self):
        super().__init__()
        self._F1_metric = F1Metric()

    def reset(self) -> None:
        self._F1_metric.reset()

    def result(self) -> float:
        return self._F1_metric.result()

    def after_training_iteration(self, strategy):
        self._F1_metric.update(strategy.mb_output, strategy.mb_y)

    def before_training_epoch(self, strategy):
        self.reset()

    def after_training_epoch(self, strategy):
        metric_value = self.result()
        metric_name = get_metric_name(self, strategy, add_experience=False, add_task=False)
        return [MetricValue(self, metric_name, metric_value, strategy.clock.train_iterations)]

    def __str__(self):
        return "F1_Accuracy_Epoch"

class TorchmetricsF1GenericPluginMetric(GenericPluginMetric[float, F1Metric]):
    """
    Base class for all accuracies plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        """Creates the Accuracy plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        super().__init__(F1Metric(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)

class TorchmetricsJaccardPluginMetric(PluginMetric[float]):
    def __init__(self):
        super().__init__()
        self._Jaccard_metric = JaccardMetric()

    def reset(self) -> None:
        self._Jaccard_metric.reset()

    def result(self) -> float:
        return self._Jaccard_metric.result()

    def after_training_iteration(self, strategy):
        self._Jaccard_metric.update(strategy.mb_output, strategy.mb_y)

    def before_training_epoch(self, strategy):
        self.reset()

    def after_training_epoch(self, strategy):
        metric_value = self.result()
        metric_name = get_metric_name(self, strategy, add_experience=False, add_task=False)
        return [MetricValue(self, metric_name, metric_value, strategy.clock.train_iterations)]

    def __str__(self):
        return "Jaccard_Accuracy_Epoch"

class TorchmetricsJaccardGenericPluginMetric(GenericPluginMetric[float, JaccardMetric]):
    """
    Base class for all accuracies plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        """Creates the Accuracy plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        super().__init__(JaccardMetric(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)

class TorchmetricsHammingPluginMetric(PluginMetric[float]):
    def __init__(self):
        super().__init__()
        self._Hamming_metric = HammingMetric()

    def reset(self) -> None:
        self._Hamming_metric.reset()

    def result(self) -> float:
        return self._Hamming_metric.result()

    def after_training_iteration(self, strategy):
        self._Hamming_metric.update(strategy.mb_output, strategy.mb_y)

    def before_training_epoch(self, strategy):
        self.reset()

    def after_training_epoch(self, strategy):
        metric_value = self.result()
        metric_name = get_metric_name(self, strategy, add_experience=False, add_task=False)
        return [MetricValue(self, metric_name, metric_value, strategy.clock.train_iterations)]

    def __str__(self):
        return "Hamming_Accuracy_Epoch"

class TorchmetricsHammingPluginMetric(GenericPluginMetric[float, HammingMetric]):
    """
    Base class for all accuracies plugin metrics
    """

    def __init__(self, reset_at, emit_at, mode, split_by_task=False):
        """Creates the Accuracy plugin

        :param reset_at:
        :param emit_at:
        :param mode:
        :param split_by_task: whether to compute task-aware accuracy or not.
        """
        super().__init__(HammingMetric(), reset_at=reset_at, emit_at=emit_at, mode=mode)

    def reset(self) -> None:
        self._metric.reset()

    def result(self) -> float:
        return self._metric.result()

    def update(self, strategy):
        self._metric.update(strategy.mb_output, strategy.mb_y)

class ExperienceAccuracyGeneral(TorchmetricsAccuracyGenericPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceAccuracyGeneral, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "General_Acc_Exp"

class ExperienceECEGeneral(TorchmetricsECEGenericPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceECEGeneral, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "General_ECE_Exp"

class ExperienceF1General(TorchmetricsF1GenericPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceF1General, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "General_F1_Exp"

class ExperienceJaccardGeneral(TorchmetricsJaccardGenericPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceJaccardGeneral, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "General_Jaccard_Exp"

class ExperienceHammingGeneral(TorchmetricsHammingPluginMetric):
    """
    At the end of each experience, this plugin metric reports
    the average accuracy over all patterns seen in that experience.
    This metric only works at eval time.
    """

    def __init__(self):
        """
        Creates an instance of ExperienceAccuracy metric
        """
        super(ExperienceHammingGeneral, self).__init__(
            reset_at="experience", emit_at="experience", mode="eval"
        )

    def __str__(self):
        return "General_Hamming_Exp"