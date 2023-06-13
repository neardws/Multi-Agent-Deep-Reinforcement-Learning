# -*- coding: UTF-8 -*-
"""
@Project ：Hierarchical-Reinforcement-Learning 
@File    ：test_NN.py
@Author  ：Neardws
@Date    ：8/5/21 3:39 下午 
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

# Do this to display pytorch version.
# The version used in this gist is 0.3.0.post4.
print(torch.__version__)

# There are three steps to demonstrate multi head network
# 1. build the network
# 2. forward pass
# 3. backward pass


# 1. build the network
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # This represents the shared layer(s) before the different heads
        # Here, I used a single linear layer for simplicity purposes
        # But any network configuration should work
        self.shared_layer = nn.Linear(5, 5)

        # Set up the different heads
        # Each head can take any network configuration
        self.sf = nn.Softmax(dim=1)
        self.linear_output = nn.Linear(5, 1)

    def forward(self, x):

        # Run the shared layer(s)
        x = self.shared_layer(x)

        # Run the different heads with the output of the shared layers as input
        sf_out = self.sf(x)
        linear_out = self.linear_output(x)

        return sf_out, linear_out


net = Network()

# 2. Run a forward pass
fake_data = Variable(torch.FloatTensor(1, 5))
sf_out, linear_out = net(fake_data)

# 3. Run a backward pass
# To run backward pass on the output of the different heads,
# we need to specify retain_graph=True on the backward pass
# this is because pytorch automatically frees the computational graph after the backward pass to save memory
# Without the computational graph, the chain of derivative is lost

# Run backward on the linear output and one of the softmax output
linear_out.backward(retain_graph=True)

# To get the gradient of the param w.r.t linear_out, we can do
grad_linear_out = {}

for name, param in net.named_parameters():
    grad_linear_out[name] = param.grad.data.clone()

# Then, to get the gradient of the param w.r.t softmax output, we first need to clear the existing gradient data
net.zero_grad()

sf_out[0, 0].backward()

grad_sf_out = {}

for name, param in net.named_parameters():
    grad_sf_out[name] = param.grad.data.clone()

print(grad_linear_out)
print(grad_sf_out)
