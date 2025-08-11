import torchvision.models as models
import torch.nn as nn

class ResNetTeacher(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetTeacher, self).__init__()
        base_model = models.resnet34(pretrained=True)
        base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
        self.model = base_model

    def forward(self, x, return_features=False):
        if return_features:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            feat = self.model.layer2(x)  # <-- you can pick which layer to distill from
            x = self.model.layer3(feat)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            x = x.view(x.size(0), -1)
            logits = self.model.fc(x)
            return logits, feat
        else:
            return self.model(x)

def get_teacher():
    return ResNetTeacher()
