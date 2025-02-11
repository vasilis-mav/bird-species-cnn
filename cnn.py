import torch
import torch.nn as nn

class BCNN(nn.Module):
    def __init__(self, input_size=(3, 224, 224), num_of_classes=525):
        super(BCNN, self).__init__()

        # Feature extraction (Convolutional layers)
        self.features = nn.Sequential(
            nn.Conv2d(input_size[0], 16, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (16, 112, 112)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 56, 56)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 28, 28)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 14, 14)
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Output: (256, 1, 1)
        )

        # Classification head (Fully connected layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Output: (256)
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_of_classes)  # Output: (525)
        )

    def forward(self, x):
        x = self.features(x)  # Feature extraction
        x = self.classifier(x)  # Classification
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        se_weight = self.global_avg_pool(x).view(b, c)
        se_weight = self.fc(se_weight).view(b, c, 1, 1)
        return x * se_weight

class BCNN_Red(nn.Module):
    def __init__(self, input_size=(3, 224, 224), num_of_classes=525):
        super(BCNN_Red, self).__init__()

        self.seq = nn.Sequential(
            # Block 1
            nn.Conv2d(input_size[0], 32, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (32, 112, 112)
            SEBlock(32),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (64, 56, 56)
            SEBlock(64),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 28, 28)
            SEBlock(128),

            # Block 4
            *[
                nn.Sequential(
                    nn.BatchNorm2d(128),
                    nn.Conv2d(128, 128, kernel_size=[1, 5], stride=1, padding='same', groups=128),
                    nn.Conv2d(128, 128, kernel_size=[5, 1], stride=1, padding='same', groups=128),
                    nn.Conv2d(128, 128, kernel_size=1, stride=1, padding='same'),
                    nn.LeakyReLU(),
                ) for _ in range(3)
            ],
            SEBlock(128),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (128, 14, 14)

            # Block 5
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.LeakyReLU(),

            *[
                nn.Sequential(
                    nn.BatchNorm2d(256),
                    nn.Conv2d(256, 256, kernel_size=[1, 5], stride=1, padding='same', groups=256),
                    nn.Conv2d(256, 256, kernel_size=[5, 1], stride=1, padding='same', groups=256),
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding='same'),
                    nn.LeakyReLU(),
                ) for _ in range(3)
            ],
            SEBlock(256),
            nn.AdaptiveAvgPool2d(1),  # Output: (256, 1, 1)
            nn.Flatten(),
            
            # Fully Connected Layers
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, num_of_classes)
        )
    
    def forward(self, x):
        return self.seq(x)
