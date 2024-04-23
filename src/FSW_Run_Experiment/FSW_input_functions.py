import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights, resnet18, ResNet18_Weights
from .ResNet18_Dropout import BasicResNet18DropOut

from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark, dataset_benchmark, filelist_benchmark, dataset_benchmark, tensors_benchmark, paths_benchmark
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import EWC, MAS, DER, PNNStrategy, Naive

device = "cuda" if torch.cuda.is_available() else "cpu"

def define_train_transforms():
    #global train_transforms
    #glogbal test_transforms
    default_transforms = int(input("Use default transformations [Yes=1] / [No=0]"))
    test_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    if default_transforms:
        train_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        horizontal_flip = float(input("Horizontal flip (float) [0 <= 1]:"))
        vertical_flip = float(input("Vertical flip (float) [0 <= 1]:"))
        kernel = int(input("Kernel size (int, it must be an odd number!!!):"))
        
        train_transforms = transforms.Compose([
        #transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=horizontal_flip) if horizontal_flip > 0 else "",
        transforms.RandomVerticalFlip(p=vertical_flip) if vertical_flip > 0 else "",
        transforms.GaussianBlur(kernel_size=kernel) if kernel > 0 and kernel % 2 != 0 else transforms.GaussianBlur(kernel_size=1),
        transforms.ToTensor()
        ])
    return train_transforms, test_transforms

def choose_NN():
    #global model, criterion, optimizer
    NN_architecture = int(input("Choose one NN architecture [ResNet18: 1], [ResNet18 with Dropout: 2], [EfficientNet: 3]"))
    if NN_architecture == 1:
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_features = model.fc.in_features                                 # extract fc layers features
        model.fc = nn.Linear(num_features, 2)     # num_of_class == 2
        model = model.to(device, dtype=torch.float32)  
    elif NN_architecture == 2:
        dropout_resnet = float(input("Dropout for Resnet [0<1]:"))
        model = BasicResNet18DropOut(num_classes=2, pretrained=True, dropout_ratio=dropout_resnet, n_stoc_forwards=2)
    else:
        model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        num_features = model.classifier[1].in_features  # Get the input features of the last layer
        model.classifier[1] = nn.Linear(num_features, 2)  # Replace it with a new layer with 2 outputs
        model = model.to(device, dtype=torch.float32)    # dtype adapted to float32    
    criterion = nn.CrossEntropyLoss()                 # set loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # set optimizer
    gradient_type = input()
    return model, criterion, optimizer

def ask_strategy_size():
    """
    Outputs an array with the following parameters
    [training_size, training_epochs, evaluation_size, disable_evaluation]
    """
    print("Choose the following parameters:")
    training_size = int(input("Training batch size (int e.g. 64):"))
    training_epochs = int(input("Number of epochs (int e.g. 10):"))
    evaluation_size = int(input("Evaluation batch size (int e.g. 32):"))
    disable_evaluation = int(input("Disable evaluation [Yes=1] / [No=0]:"))
    return [training_size, training_epochs, evaluation_size, disable_evaluation]    

def choose_Strategy(model, criterion, optimizer, eval_plugin):
    #global cl_strategy
    Naive_Pre = Naive(model,        # no deepcopy of the model -> weights should be adapted to the first task before Continual Learning starts
                    optimizer,
                    criterion,
                    train_mb_size=64,
                    train_epochs=10,                       
                    eval_mb_size=32,
                    evaluator=eval_plugin,
                    eval_every=-1,  # set to -1 to disable the evaluation
                    device=device)
    
    choice = int(input("Choose strategy: EWC[1], MAS[2], DER[3]"))
    if choice == 1:
        size_cl = ask_strategy_size()
        lambda_EWC = float(input("Lambda value e.g. 1000 (EWC):"))
        mode_EWC = input("Mode (e.g. online):")
        decay_factor_EWC = float(input("Decay Factor e.g. 0.7 (EWC):"))
        cl_strategy_EWC = EWC(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=size_cl[0],
            train_epochs=size_cl[1],                       
            eval_mb_size=size_cl[2],
            evaluator=eval_plugin,
            eval_every=-1,  # set to -1 to disable the evaluation
            device=device,
            ewc_lambda=lambda_EWC,
            mode=mode_EWC,
            decay_factor=decay_factor_EWC
        )
        return cl_strategy_EWC, Naive_Pre
    elif choice == 2:
        size_cl = ask_strategy_size()
        lambda_MAS = float(input("Lambda value (MAS Strategy):"))
        alpha_MAS = float(input("Alpha value (MAS Strategy):"))
        verbose_MAS = int(input("Verbose [1:True] / [0:False]:"))
        cl_strategy_MAS = MAS(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            lambda_reg=lambda_MAS,  # Fixing the variable name here
            alpha=alpha_MAS,
            verbose=verbose_MAS,
            train_mb_size=size_cl[0],
            train_epochs=size_cl[1],
            eval_mb_size=size_cl[2],
            device=device,
            evaluator=eval_plugin,
            eval_every=-1
        )
        return cl_strategy_MAS, Naive_Pre
    elif choice == 3:
        size_cl = ask_strategy_size()
        alpha_der = float(input("Alpha value (DER Strategy):"))
        beta_der = float(input("Beta value (DER Strategy):"))
        cl_strategy_DER = DER(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=size_cl[0],
            train_epochs=size_cl[1],                       
            eval_mb_size=size_cl[2],
            evaluator=eval_plugin,
            eval_every=-1,  # set to -1 to disable the evaluation
            device=device,
            alpha=alpha_der,
            beta=beta_der
        )
        return cl_strategy_DER, Naive_Pre
    else:
        raise ValueError("No accepted input")

    
