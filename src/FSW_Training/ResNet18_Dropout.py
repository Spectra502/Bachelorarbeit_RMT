import torch
import torch.nn as nn
import torchvision.models as models

class BasicResNet18DropOut(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool, dropout_ratio: float, n_stoc_forwards: int):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_ratio),
            nn.Linear(in_features=512, out_features=num_classes)
        )
        self.nsfwds = n_stoc_forwards

    def forward(self, x):
        if self.model.training:
            return self.model(x)
        elif not self.model.training:
            return self.stochastic_inference(x, n=self.nsfwds)

    def stochastic_inference(self, x, n):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        self.model.fc.train()
        logits = [self.model.fc(x) for _ in range(0, n)]
        logits = torch.stack(logits)
        self.model.fc.eval()
        return logits.mean(dim=0)