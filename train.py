import torch
import torch.nn as nn
from chatDataset import ChatDataset
from importTrain import importTrainData
from models import NeuralNet

if __name__ == '__main__':
  # Get Training data
  (X_train, y_train, tags, allWords) = importTrainData()

  # Hyperparameters (can change)
  batchSize = 8
  hiddenSize = 8
  learningRate = 0.001
  epochs = 1000
  inputSize = len(allWords) # size of allWords/bagOfWords vector (all patterns)
  outputSize = len(tags) # size of tags (classify tags)

  # training data
  dataset = ChatDataset(X_train, y_train)
  trainLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=2)

  # device to train on
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)

  # loss and optimizer
  lossFunction = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

  # Training
  for epoch in range(epochs):
    numCorrect = 0
    for (wordsBagVector, tags) in trainLoader:
      # push to device
      wordsBagVector = wordsBagVector.to(device) # words (X_train)
      tags = tags.to(device) # labels (y_train)
      # zero out gradients
      optimizer.zero_grad()

      # forward pass
      #WHY IS wordsBagVector NONETYPE
      outputs = model(wordsBagVector)
      loss = lossFunction(outputs, tags)

      # backward step
      loss.backward() # calculate back propagation
      optimizer.step() # single optimization step
      _, predicted = outputs.max(1) # get predicted from outputs
      numCorrect += (predicted == tags).double().sum().item() # count number of correct prediction
      total += tags.size(0) # count total number of training samples

    if (epoch + 1) % 100 == 0:
      print(f'epoch {epoch + 1}/{epochs}, loss={loss.item():.4f}, accuracy={numCorrect / total:.4f}')

  print(f'final loss and accuracy, loss={loss.item():.4f}, accuracy={numCorrect / total:.4f}')