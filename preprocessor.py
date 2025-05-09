import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
#nltk.download('punkt_tab')

class Preprocessor:
  """
  A class used to preprocess phrases
  
  Fields
  -----------
  stemmer : PorterStemmer
      stemmer to generate root from words
  """

  stemmer = None

  def __init__(self) -> None:
    self.stemmer = PorterStemmer()
  
  # split string into words/components (Ex. aren't -> are n't)
  def tokenize(self, phrase) -> list[str]:
    return nltk.word_tokenize(phrase)

  # generate root form of words (Ex. organize -> organ)
  def stem(self, word):
    return self.stemmer.stem(word.lower())

  # generate bag of words vector (all words vector with 1/0 to indicate the word's presence)
  """
  tokenizedPhrase =    ["hello", "how", "are", "you"]
  allWords =           ["hi", "hello", "I", "you", "bye", "thank", "cool"]
  bagOfWordsVector =   [  0 ,    1   ,  0 ,  1   ,   0  ,    0   ,   0   ]
  """
  def bagOfWordsVector(self, tokenizedPhrase, allWords):
    stemmedTokenizedPhrase = [self.stem(word) for word in tokenizedPhrase] # stem tokenizedPhrase
    bagVector = np.zeros(len(allWords), dtype = np.float32) 
    for i in range(len(allWords)): # go through all words
      if allWords[i] in stemmedTokenizedPhrase:
        bagVector[i] = 1.0 # indicate with 1.0 that the word exists in stemmedTokenizedPhrase
    return bagVector