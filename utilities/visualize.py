import matplotlib.pyplot as plt
import networkx as nx
import torch


def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(10, 10), dpi=400)
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f"Epoch: {epoch}, Loss: {loss.item():.4f}", fontsize=16)
    else:
        nx.draw_networkx(
            h,
            pos=nx.spring_layout(h, seed=42),
            with_labels=False,
            node_color=color,
            cmap="Set2",
        )
    plt.show()
