import torch.nn.functional as F

'''
Network with 784 input units, a hidden layer with 128 units and a ReLU activation, 
then a hidden layer with 64 units and a ReLU activation, and finally an output layer with a softmax activation as shown above. 
You can use a ReLU activation with the nn.ReLU module or F.relu function.
'''
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.output(x), dim=1)
        
        return x
