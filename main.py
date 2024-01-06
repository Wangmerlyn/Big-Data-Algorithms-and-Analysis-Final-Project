import argparse
import os
import numpy as np
import random
from scipy import optimize
from sklearn import get_config
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import seaborn as sns
import matplotlib.pyplot as plt

import models
import data_handlers
import configs
from utilities import tsne_visualize


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
        default="VanillaGAT",
        help="Model to use. Possible values: GFusion, VanillaGAT, VanillaGCN.",
        choices=["GFusion_1", "VanillaGAT", "VanillaGCN"],
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
    return parser.parse_args()


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def validate(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[data.val_mask] == data.y[data.val_mask]
    acc = int(correct.sum()) / int(data.val_mask.sum())
    return acc


def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    # Set seed for reproducibility
    RANDOM_SEED = 227

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load dataset
    os.makedirs("./data", exist_ok=True)
    dataset = Planetoid(
        root="./data", name=args.dataset, transform=NormalizeFeatures()
    )
    data = dataset[0].to(device)

    config = configs.get_config(args.model, args.dataset)

    # Load model
    model = getattr(models, args.model)(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        **config,
    ).to(device)

    # Initialize Optimizer
    learning_rate = 0.005
    decay = 5e-4
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=decay
    )
    # Loss Function
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.NLLLoss()

    os.makedirs("checkpoints", exist_ok=True)
    losses = []
    best_val_acc = 0
    best_epoch = 0
    num_epochs = 2000

    for epoch in range(0, num_epochs):
        loss = train(model, data, optimizer, criterion)
        losses.append(loss)
        val_acc = validate(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
        if epoch % 100 == 0:
            print(f"Epoch:{epoch:03d}, Loss:{loss:.4f}")
    print(f"Best Epoch:{best_epoch:03d}, Val Acc:{best_val_acc:.4f}")

    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    test_acc = test(model, data)
    print(f"Test Acc:{test_acc:.4f}")

    losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]
    loss_indices = [i for i, l in enumerate(losses_float)]
    plt.figure(figsize=(10, 10), dpi=400)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.plot(loss_indices, losses_float)
    plt.scatter(
        best_epoch,
        losses_float[best_epoch],
        c="r",
        label="Best Epoch",
        marker="*",
        s=200,
    )
    figs_folder = "figs"
    os.makedirs(figs_folder, exist_ok=True)
    plt.savefig(f"figs/loss_vs_epoch_{args.model}.png")

    plt.cla()

    tsne_visualize(
        model.get_embedding(data.x, data.edge_index)[data.test_mask]
        .cpu()
        .detach(),
        model(data.x, data.edge_index)[data.test_mask]
        .argmax(dim=1)
        .cpu()
        .detach(),
        data.y[data.test_mask].cpu().detach(),
    )
    plt.savefig(f"figs/tsne_{args.model}.png")
