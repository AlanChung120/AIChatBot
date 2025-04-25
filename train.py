import json
from preprocessor import Preprocessor 

with open('intents.json', 'r') as file:
  intents = json.load(file)

toIgnore = ['?', '!', '.', ',']
preprocessor = Preprocessor()

allWords = []
tags = []
xy = []

for intent in intents['intents']:
  tag = intent['tag']
  tags.append(tag)
  for pattern in intent['patterns']:
    w = preprocessor.tokenize(pattern)
    allWords.extend(w)
    xy.append((w, tag))

print(xy)