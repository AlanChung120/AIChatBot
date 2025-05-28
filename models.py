import torch.nn as nn

class NeuralNet(nn.Module):
  """
  A class representing the neural network used for classification
  """

  def __init__(self, inputSize, hiddenSize, hiddenSize2,  classes):
    super(NeuralNet, self).__init__()

    # feed forward neural network with two layers
    self.network = nn.Sequential(
      nn.Linear(inputSize, hiddenSize),
      nn.ReLU(),
      nn.Linear(hiddenSize, hiddenSize),
      nn.ReLU(),
      nn.Linear(hiddenSize, hiddenSize2),
      nn.ReLU(),
      nn.Linear(hiddenSize2, hiddenSize2),
      nn.ReLU(),
      nn.Linear(hiddenSize2, classes)
    )
  
  # forward pass of the neural network 
  def forward(self, x):
    output = self.network(x)
    # no activation and no softmax (done through cross entropy)
    return output