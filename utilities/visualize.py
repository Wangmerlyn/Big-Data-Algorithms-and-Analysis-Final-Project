import matplotlib.pyplot as plt
import networkx as nx
import torch


def visualize_data(h, color, epoch=None, loss=None):
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


def tsne_visualize(embeddings, y_pred, y_true):
    from sklearn.manifold import TSNE

    t_sne = TSNE(n_components=2, init="pca", random_state=42)
    embeddings_t = t_sne.fit_transform(embeddings)
    plt.figure(figsize=(10, 10), dpi=400)
    plt.scatter(
        embeddings_t[y_pred == y_true, 0],
        embeddings_t[y_pred == y_true, 1],
        c=y_pred[y_pred == y_true],
        # label=y_pred[y_pred == y_true],
    )
    plt.scatter(
        embeddings_t[y_pred != y_true, 0],
        embeddings_t[y_pred != y_true, 1],
        c=y_pred[y_pred != y_true],
        # label=y_pred[y_pred != y_true],
        marker="x",
    )
    # plt.legend()
    plt.show()
