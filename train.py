from torch.utils.data import DataLoader
from chatDataset import ChatDataset
from importTrain import importTrainData
from models import NeuralNet

# Get Training data
(X_train, y_train, tags, allWords) = importTrainData()

# Hyperparameters (can change)
batchSize = 8
hiddenSize = 8
inputSize = len(allWords) # size of allWords/bagOfWords vector (all patterns)
outputSize = len(tags) # size of tags (classify tags)

dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=2)

model = NeuralNet(inputSize, hiddenSize, outputSize)