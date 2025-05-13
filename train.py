import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from chatDataset import ChatDataset
from importTrain import importTrainData


# Get Training data
(X_train, y_train) = importTrainData()

# Hyperparameters
batch_size = 8

dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)