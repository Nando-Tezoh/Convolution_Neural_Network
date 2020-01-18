### Start
import numpy as np # Matrix Operations (Matlab of Python)
import pandas as pd # Work with Datasources
import matplotlib.pyplot as plt # Drawing Library
import torch.optim as optim

from PIL import Image

import torch # Like a numpy but we could work with GPU by pytorch library
import torch.nn as nn # Nural Network Implimented with pytorch
import torchvision # A library for work with pretrained model and datasets

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

import glob
import os

from torchvision import datasets, transforms, models


image_size = (100, 100)
image_row_size = image_size[0] * image_size[1]
