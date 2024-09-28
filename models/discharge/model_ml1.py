import torch.nn as nn
import torch.optim as optim
import torch
import os

# # Define a basic ResNet block with dropout
# class BasicBlock(nn.Module):
#     def __init__(self, in_features, out_features, dropout_prob=0.5):
#         super(BasicBlock, self).__init__()
#         self.fc1 = nn.Linear(in_features, out_features)
#         self.bn1 = nn.BatchNorm1d(out_features)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(out_features, out_features)
#         self.bn2 = nn.BatchNorm1d(out_features)
#         self.dropout = nn.Dropout(p=dropout_prob)  # Add dropout layer
        
#         # Shortcut connection
#         if in_features != out_features:
#             self.shortcut = nn.Sequential(
#                 nn.Linear(in_features, out_features),
#                 nn.BatchNorm1d(out_features)
#             )
#         else:
#             self.shortcut = nn.Identity()
    
#     def forward(self, x):
#         identity = x
#         out = self.fc1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.dropout(out)  # Apply dropout after ReLU
        
#         out = self.fc2(out)
#         out = self.bn2(out)
        
#         identity = self.shortcut(identity)
#         out += identity
#         out = self.relu(out)
#         return out

# # Define the main model with dropout
# class RegressionModelWithResNet(nn.Module):
#     def __init__(self, input_size, output_size, dropout_prob=0.2):
#         super(RegressionModelWithResNet, self).__init__()
#         # Initial fully connected layer
#         self.fc_input = nn.Linear(input_size, 64)  # Reduce dimensionality before ResNet block
        
#         # ResNet block with dropout
#         self.resnet_block = BasicBlock(64, 64, dropout_prob=dropout_prob)
        
#         # Output layer
#         self.fc_output = nn.Linear(64, output_size)
#         self.dropout = nn.Dropout(p=dropout_prob)  # Add dropout before output if desired
    
#     def forward(self, x):
#         x = self.fc_input(x)
#         x = self.resnet_block(x)
#         x = self.dropout(x)  # Apply dropout before output
#         x = self.fc_output(x)
#         return x
    
# # Create model with dropout
# def create_model_fc_resnet(input_size, output_size, dropout_prob=0.2):
#     model = RegressionModelWithResNet(input_size=input_size, output_size=output_size, dropout_prob=dropout_prob)
#     return model

# Define a basic ResNet block with dropout
class BasicBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob=0.5):
        super(BasicBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(p=dropout_prob)  # Add dropout layer
        
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
        out = self.dropout(out)  # Apply dropout after ReLU
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)
        return out

# Define the main model with dropout
class RegressionModelWithResNet(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0.2):
        super(RegressionModelWithResNet, self).__init__()
        # Initial fully connected layer
        self.fc_input = nn.Linear(input_size, 64)  # Reduce dimensionality before ResNet block
        
        # ResNet block with dropout
        self.resnet_block = BasicBlock(64, 64, dropout_prob=dropout_prob)
        
        # Output layer
        self.fc_output = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(p=dropout_prob)  # Add dropout before output if desired
    
    def forward(self, x):
        x = self.fc_input(x)
        x = self.resnet_block(x)
        x = self.dropout(x)  # Apply dropout before output
        x = self.fc_output(x)
        return x

# Create model with dropout
def create_model_fc_resnet(input_size, output_size, dropout_prob=0.5):
    model = RegressionModelWithResNet(input_size=input_size, output_size=output_size, dropout_prob=dropout_prob)
    return model

def load_model_ml1(input_size, output_size):
    model = create_model_fc_resnet(input_size, output_size)
    #trained_model_ml1 = 'models/discharge/trained model/Ml_1_lr_e4_bs_32_resnet.pth'
    
    # Dynamically construct the full path
    trained_model_ml1 = os.path.join("/home", "mhews", "hms_deployment", "models", "discharge", "trained_model", "Ml 1 without filtering prec.pth")
    #trained_model_ml1 = '/opt/ews/ews_deployment/models/discharge/trained model/Ml 1 without filtering prec.pth'
    # Load the saved weights into the model
    model.load_state_dict(torch.load(trained_model_ml1, map_location=torch.device("cpu")))
    model.eval()
    print("Successfully  loaded ml1")
    return model
