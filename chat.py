import random
import json
import torch
from models import NeuralNet
from preprocessor import Preprocessor


if __name__ == '__main__':
  preprocessor = Preprocessor()
  with open('intents.json', 'r') as file:
    intents = json.load(file)
  
  # load trained data
  FILE = "data.pth"
  trainingData = torch.load(FILE)

  inputSize = trainingData["input_size"]
  hiddenSize = trainingData["hidden_size_1"]
  hiddenSize2 = trainingData["hidden_size_2"]
  outputSize = trainingData["output_size"]
  allWords = trainingData["all_words"]
  tags = trainingData["tags"]
  modelState = trainingData["model_state"]

  # set device and model
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = NeuralNet(inputSize, hiddenSize, hiddenSize2, outputSize).to(device)

  # load the learned parameters from the file
  model.load_state_dict(modelState)

  # set the model to evaluation mode
  model.eval()

  # chat loop
  botName = "ChatBot"
  print("Let's chat! type 'quit' to exit")
  while True:
    inputSentence = input("You: ")
    if inputSentence == "quit":
      break
  
    tokenizedSentence = preprocessor.tokenize(inputSentence)
    # turn into bag of words vector
    X = preprocessor.bagOfWordsVector(tokenizedSentence, allWords)
    # reshape it as expected from the model (1 data sample, bagOfWords/allWords vector size)
    X = X.reshape(1, X.shape[0])

    # convert numpy X into torch tensor data type
    X = torch.from_numpy(X)

    # get the tag using the model from input X
    output = model(X) # forward function is implicitly called when instance (object) is called directly
    _, predicted = output.max(1)
    tag = tags[predicted.item()]
    
    # softmax the output to convert to probability vector
    probs = torch.softmax(output, dim=1)
    # get the probability for the predicted tag (highest one)
    prob = probs[0][predicted.item()]

    # if probability is high enough (confident of this tag)
    if prob.item() > 0.75:
      for intent in intents["intents"]:
        if tag == intent["tag"]:
          # say a random response from this tag
          print(f'{botName}: {random.choice(intent["responses"])}')
    # otherwise 
    else:
      print(f'{botName}: I do not understand...')


