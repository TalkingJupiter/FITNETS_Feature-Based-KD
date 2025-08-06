import torch.nn
import torch.nn.functional as F

class StudentCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )  
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLu(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, return_features=False):
        features = self.features(x)
        logits = self.classifier(features)
        return (logits, features) if return_features else logits

def get_student():
    return StudentCNN()