import torch
import torch.nn as nn
import torchvision
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')