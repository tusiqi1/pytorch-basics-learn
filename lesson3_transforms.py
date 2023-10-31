# Data does not always come in its final processed form that is required for training machine learning algorithms.
# We use transforms to perform some manipulation of the data and make it suitable for training.

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    # modify the PIL image format to FloatTensor, and scales the image’s pixel intensity values in the range [0., 1.]
    transform=ToTensor(),
    # modify the 0,1,2 label format to one-hot encoded tensors
    # It first creates a zero tensor of size 10 (the number of labels in our dataset) and calls scatter_ which assigns a value=1 on the index as given by the label y.
    #for i in range(10):
    #   print(torch.zeros(10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(i), value=1))

    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
