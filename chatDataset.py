from torch.utils.data import Dataset

class ChatDataset(Dataset):
  """
  A class used to store chat dataset derived from Dataset
  
  Fields
  -----------
  n_samples : int
      number of data points
  x_data : ndarray
      input data (list of bag of words vectors)
  y_data : ndarray
      list of y labels (list of indices representing tags)
  """
  n_samples = 0
  x_data = None
  y_data = None

  def __init__(self, X_train, y_train):
    self.n_samples = len(X_train)
    self.x_data = X_train
    self.y_data = y_train
  
  # override: get input data and y label at the given index
  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]
  
  # override: return n_samples (number of data points)
  def __len__(self):
    return self.n_samples