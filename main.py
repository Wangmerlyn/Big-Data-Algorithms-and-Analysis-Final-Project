import os
import argparse
import numpy as np
import random
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

import models
import data_handlers
import utilities


def parse_args():
    parser = argparse.ArgumentParser(description="GFusion")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        help="Dataset to use. Possible values: Cora, CiteSeer",
        choices=["Cora", "CiteSeer"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GFusion",
        help="Model to use. Possible values: GFusion, VanillaGAT, VanillaGCN.",
        choices=["GFusion", "VanillaGAT", "VanillaGCN"],
    )
    parser.add_argument(
        "--epochs", type=int, default=1001, help="Number of epochs to train."
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Initial learning rate."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (L2 loss).",
    )


if __name__ == "__main__":
    # Set seed for reproducibility
    RANDOM_SEED = 227

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
