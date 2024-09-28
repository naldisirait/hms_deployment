import torch
import torch.nn as nn
import torch.optim as optim

class FullyConnectedModel(nn.Module):
    def __init__(self):
        super(FullyConnectedModel, self).__init__()
        self.layer1 = nn.Linear(72, 128)  # First hidden layer
        self.layer2 = nn.Linear(128, 128)  # Second hidden layer
        self.layer3 = nn.Linear(128,128)
        self.layer4 = nn.Linear(128,128)
        self.output_layer = nn.Linear(128, 145)  # Output layer
        self.silu = nn.SiLU()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        x = self.output_layer(x)
        return x

# Define a basic ResNet block
class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(BasicBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # Shortcut connection
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        
        identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)
        return out

# Define the main model
class RegressionModelWithResNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionModelWithResNet, self).__init__()
        # Initial fully connected layer
        self.fc_input = nn.Linear(input_size, 128)  # Reduce dimensionality before ResNet block
        
        # ResNet block
        self.resnet_block = BasicBlock(128, 128)
        
        # Output layer
        self.fc_output = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = self.fc_input(x)
        x = self.resnet_block(x)
        x = self.fc_output(x)
        return x

# create model regression 
def create_model_fc_resnet(input_size, output_size):
    model = RegressionModelWithResNet(input_size=input_size, output_size=output_size)
    return model