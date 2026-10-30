import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self,
                 input_features: int,
                 output_features: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(
                input_features,    
                30,
                (3,3)),
            nn.ReLU(),
            nn.Conv2d(
                30,
                20,
                (2,2)
            ),
            nn.ReLU(),
            nn.MaxPool2d((2,2),2),
            nn.Flatten(),
            nn.LazyLinear(300),
            nn.ReLU(),
            nn.Linear(300,output_features)
        )
    def forward(self, x):
        return self.layer_stack(x)