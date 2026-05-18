from torch import nn

class WineNet(nn.Module):
    def __init__(self, hidden_neurons):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_neurons),
            nn.Sigmoid(),
            nn.Linear(hidden_neurons, 3),
        )
    
    def forward(self, x):
        return self.net(x)
