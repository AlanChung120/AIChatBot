import torch
import torch.nn as nn
from chatDataset import ChatDataset
from importTrain import importTrainData
from models import NeuralNet

if __name__ == '__main__':
  # Get Training data
  (X_train, y_train, allTags, allWords) = importTrainData()

  # Hyperparameters (can change)
  batchSize = 8 # batch size the trainLoader loads at a time (ex. 26 = 8 + 8 + 8 + 2)
  hiddenSize = 32
  hiddenSize2 = 16
  learningRate = 0.001
  epochs = 100
  inputSize = len(allWords) # size of allWords/bagOfWords vector (all patterns)
  outputSize = len(allTags) # size of tags (classify tags)

  # training data
  dataset = ChatDataset(X_train, y_train)
  # trainLoader (training samples) has the same number of elements as number of training samples (patterns in intents.json)
  # each training sample is form of numpy array (bagOfWords vector AKA input features vector)
  trainLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=2)

  # device to train on
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = NeuralNet(inputSize, hiddenSize, hiddenSize2, outputSize).to(device)
  # set the model to train mode
  model.train()

  # loss and optimizer
  lossFunction = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

  # Training
  for epoch in range(epochs):
    numCorrect = 0
    total = 0
    # convert numpy into torch tensor data type when loading from trainLoader
    for (wordsBagVector, tags) in trainLoader:
      # push to device
      wordsBagVector = wordsBagVector.to(device) # batch of words (X_train)
      tags = tags.to(device) # batch of labels/tag index (y_train)
      # zero out gradients
      optimizer.zero_grad()

      # forward pass----------------------------------------------
      # outputs: len(allTags) (column) X batch_size (row) (higher the number the more liekly it is that tag)
      # len(allTags) = number of output features/classifications
      outputs = model(wordsBagVector)
      loss = lossFunction(outputs, tags)

      # backward step------------------------------------------------
      loss.backward() # calculate back propagation
      optimizer.step() # single optimization step
      # get predicted tag from outputs (get index of highest element in each row (for each training sample in batch))
      _, predicted = outputs.max(1)
      # count number of correct prediction (check equality, turn boolean -> double (1.0, 0.0), add the elements of the numpy array, get that sum)
      numCorrect += (predicted == tags).double().sum().item()
      # count total number of training samples (add size of the current batch)
      total += tags.size(0)

    if (epoch + 1) % 10 == 0:
      print(f'epoch {epoch + 1}/{epochs}, loss={loss.item():.4f}, accuracy={numCorrect / total:.4f}')

  print(f'final loss and accuracy, loss={loss.item():.4f}, accuracy={numCorrect / total:.4f}')

  # training data to save
  trainingData = {
    "model_state": model.state_dict(),
    "input_size": inputSize,
    "output_size": outputSize,
    "hidden_size_1": hiddenSize,
    "hidden_size_2": hiddenSize2,
    "all_words": allWords,
    "tags": allTags
  }

  # save to a py torch file
  FILE = "data.pth"
  torch.save(trainingData, FILE)

  print(f'Training complete. File saved to {FILE}')