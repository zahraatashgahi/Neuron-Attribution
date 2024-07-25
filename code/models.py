import torch
import torch.nn as nn


        
class MLP(nn.Module):
    def __init__(self, num_features, num_outputs, dropout=0, batch_norm=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_features, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out1 = self.relu(out)
        out = self.fc2(out1)
        out2 = self.relu(out)
        out = self.fc3(out2)
        out = nn.functional.softmax(out, dim=0)
        return out, out2, out1


