import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def activation(x):
    return 1/(1+torch.exp(-x))

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

n_label = 64
n_input = 784
n_hidden = 256
n_output = 10

features = images.view(n_label, 28*28)

# Random weights
w_input_hidden = torch.randn(n_input, n_hidden)
w_hidden_output = torch.randn(n_hidden, n_output)

# Random Bias 
b_input_hidden = torch.randn(1, n_hidden)
b_hidden_output = torch.randn(1, n_output)

# MM 
mm_input_hidden = activation(torch.mm(features, w_input_hidden) + b_input_hidden)
mm_hidden_output = torch.mm(mm_input_hidden, w_hidden_output) + b_hidden_output
probabilities = softmax(mm_hidden_output)

print(mm_hidden_output.shape)
print(probabilities.shape)
print(probabilities.sum(dim=1))
