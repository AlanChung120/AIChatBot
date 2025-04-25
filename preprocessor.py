import nltk
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

  def tokenize(self, phrase) -> list[str]:
    return nltk.word_tokenize(phrase)

  def stem(self, word):
    return self.stemmer.stem(word.lower())

  def bagOfWordsVector(self, tokenizedPhrase, allWords):
    pass