import json
import numpy as np
from preprocessor import Preprocessor 

# import train data from json and format it appropriately
def importTrainData():
  with open('intents.json', 'r') as file:
    intents = json.load(file)


  # preprocessing training data steps (NLTK):
  # tokenize -> lowercase -> stemming -> exclude punctuation -> bag of words vector
  preprocessor = Preprocessor()
  toIgnore = ['?', '!', '.', ',']

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
    y_train.append(tags.index(tag)) # store the associated tag index for tags list (For CrossEntropyLoss)
  X_train = np.array(X_train)
  y_train = np.array(y_train)

  return X_train, y_train, tags, allWords 