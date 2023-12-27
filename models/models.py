import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

# The GATConv class implements the Graph Attention Network layer introduced in the GAT paper.


class GAT(torch.nn.Module):
    def __init__(
        self, num_features, hidden_channels, num_classes, attention_heads=8
    ):
        super(GAT, self).__init__()
        self.conv1 = GATConv(
            num_features, hidden_channels, heads=attention_heads, dropout=0.6
        )
        self.conv2 = GATConv(
            attention_heads * hidden_channels,
            num_classes,
            heads=1,
            dropout=0.6,
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# The GCN Model


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # First Message Passing Layer
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer
        x = F.softmax(self.out(x), dim=1)
        return x
