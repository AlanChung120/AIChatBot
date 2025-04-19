from preprocessor import Preprocessor 

if __name__ == "__main__":
  preprocessor = Preprocessor()
  testStr = "How long does shipping take?"
  testStr = preprocessor.tokenize(testStr)
  print(testStr)
  words = ["Organize", "organizes", "organizing"]
  stemmed_words = [preprocessor.stem(w) for w in words]
  print(stemmed_words)