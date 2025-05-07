import json
import numpy as np
from preprocessor import Preprocessor 

with open('intents.json', 'r') as file:
  intents = json.load(file)

toIgnore = ['?', '!', '.', ',']
preprocessor = Preprocessor()

allWords = []
tags = []
xy = []
X_train = []
y_train = []

# preprocessing steps
for intent in intents['intents']:
  tag = intent['tag']
  tags.append(tag) # add to tags
  for pattern in intent['patterns']:
    words = preprocessor.tokenize(pattern) # tokenize patterns
    allWords.extend(words) # add to all words
    xy.append((words, tag)) # x = w (tokenized patterns) -> y = tag

allWords = [preprocessor.stem(word) for word in allWords if word not in toIgnore]
allWords = sorted(set(allWords)) # stemmed and sorted all words (removed duplicates)
tags = sorted(set(tags)) # sorted tag labels (removed duplicates)

# setup training data
for (words, tag) in xy:
  bagVector = preprocessor.bagOfWordsVector(words, allWords) # get bag of words vector
  X_train.append(bagVector)
  y_train.append(tags.index(tag)) # store the associated tag index (For CrossEntropyLoss)
X_train = np.array(X_train)
y_train = np.array(y_train)