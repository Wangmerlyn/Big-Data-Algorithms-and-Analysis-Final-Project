import os
import numpy as np
import random
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import models
import data_handlers
import utilities

# Set seed for reproducibility
RANDOM_SEED = 227

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
