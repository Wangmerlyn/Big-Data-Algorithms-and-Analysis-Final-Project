import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

# The GATConv class implements the Graph Attention Network layer introduced in the GAT paper.


class GFusion(torch.nn.Module):
    def __init__(
        self,
        num_features,
        hidden_channels_1,
        hidden_channels_2,
        num_classes,
        attention_heads=8,
        dropout=0.6,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels_1)
        self.conv_mid = GCNConv(hidden_channels_1, hidden_channels_2)
        self.conv2 = GCNConv(hidden_channels_2, hidden_channels_2)
        self.gatconv1 = GATConv(
            hidden_channels_2,
            hidden_channels_2 // attention_heads,
            heads=attention_heads,
            dropout=dropout,
        )
        self.out = Linear(hidden_channels_2, num_classes)

    def forward(self, x, edge_index, dropout=0.5):
        # First Message Passing Layer
        x = F.dropout(
            self.conv1(x, edge_index).relu(),
            p=dropout,
            training=self.training,
        )
        x = F.dropout(
            self.conv_mid(x, edge_index).relu(),
            p=dropout,
            training=self.training,
        )
        # x = x.relu()
        # x = F.dropout(x, p=dropout, training=self.training)

        # Second Message Passing Layer
        # x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = F.dropout(x, p=dropout, training=self.training)

        x = F.dropout(
            self.conv2(x, edge_index).relu(), p=dropout, training=self.training
        )

        x = x + F.dropout(F.elu(self.gatconv1(x, edge_index)), p=dropout)

        return F.log_softmax(self.out(x), dim=1)


class GFusion_1(torch.nn.Module):
    def __init__(
        self,
        num_features,
        hidden_channels,
        num_classes,
        attention_heads=8,
        dropout=0.6,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.gatconv1 = GATConv(
            hidden_channels,
            hidden_channels // attention_heads,
            heads=attention_heads,
            dropout=dropout,
        )
        self.gatconv2 = GATConv(
            hidden_channels,
            hidden_channels // attention_heads,
            heads=attention_heads,
            dropout=dropout,
        )
        self.out = Linear(hidden_channels, num_classes)

    def get_embedding(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.gatconv1(x, edge_index)
        x = F.elu(x)
        x = self.gatconv2(x, edge_index)
        return x

    def forward(self, x, edge_index, dropout=0.5):
        # First Message Passing Layer
        x = F.dropout(
            self.conv1(x, edge_index).relu(),
            p=dropout,
            training=self.training,
        )

        x = F.elu(self.gatconv1(x, edge_index))
        return F.log_softmax((x), dim=1)


class VanillaGAT(torch.nn.Module):
    def __init__(
        self,
        num_features,
        hidden_channels,
        num_classes,
        attention_heads=8,
        dropout=0.6,
    ):
        super(VanillaGAT, self).__init__()
        self.conv1 = GATConv(
            num_features,
            hidden_channels,
            heads=attention_heads,
            dropout=dropout,
        )
        self.conv2 = GATConv(
            attention_heads * hidden_channels,
            num_classes,
            heads=1,
            dropout=dropout,
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embedding(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x


# The GCN Model


class VanillaGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5):
        super(VanillaGCN, self).__init__()

        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # First Message Passing Layer
        dropout = self.dropout
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=dropout, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=dropout, training=self.training)

        # Output layer
        x = F.log_softmax(self.out(x), dim=1)
        return x

    def get_embedding(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x
