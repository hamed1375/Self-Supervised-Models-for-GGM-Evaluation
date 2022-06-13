"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""

import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn import GINConv, GINEConv, global_add_pool


class MLP(nn.Module):
    """MLP with linear output"""

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, init, limit_lipchitz, L):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()

        self.num_layers = num_layers
        self.output_dim = output_dim

        self.limit_lipchitz = limit_lipchitz
        self.L = L

        if init == 'orthogonal':
            self.orthogonal_initialization = True
        elif init == 'default':
            self.orthogonal_initialization = False
        else:
            raise ValueError('initialization setup is not correct')

        self.linears = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        if num_layers == 1:
            self.linears.append(nn.Linear(input_dim, output_dim))
        else:
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

        if self.orthogonal_initialization:
            for m in self.linears:
                torch.nn.init.orthogonal_(m.weight)

    def normalize_layers(self):
        if self.limit_lipchitz:
            with torch.no_grad():
                for lin in self.linears:
                    scale = self.L / torch.linalg.norm(lin.weight, ord=2)
                    if scale < 1.0:
                        lin.weight = nn.Parameter(lin.weight * scale)

    def forward(self, x):
        if self.linears[0].training:
            self.normalize_layers()
        h = x
        for i in range(self.num_layers - 1):
            h = F.relu(self.batch_norms[i](self.linears[i](h)))
        return self.linears[-1](h)

    def __len__(self):
        return self.num_layers

    def __getitem__(self, item):
        return self.linears[item]


def make_gin_conv(input_dim, out_dim, init, edge_dim=None, limit_lipchitz=False, L=1):
    mlp = MLP(num_layers=2, input_dim=input_dim, hidden_dim=out_dim, output_dim=out_dim,
              init=init, limit_lipchitz=limit_lipchitz, L=L)
    if edge_dim is None:
        return GINConv(mlp)
    else:
        return GINEConv(mlp, edge_dim=edge_dim)


class GConv(nn.Module):
    def __init__(self, args):
        super(GConv, self).__init__()

        # self.structural_feat_extractor = FeatureAdder(args=args)
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(args.num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(args.input_dim, args.hidden_dim, args.init,
                                                 args.edge_dim, args.limit_lip, args.lip_factor))
            else:
                self.layers.append(make_gin_conv(args.hidden_dim, args.hidden_dim, args.init,
                                                 args.edge_dim, args.limit_lip, args.lip_factor))
            self.batch_norms.append(nn.BatchNorm1d(args.hidden_dim))

        project_dim = args.hidden_dim * args.num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch, edge_attr=None):
        # x = self.structural_feat_extractor.add_features_batch(x, edge_index, batch.shape[0])
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            if edge_attr is None:
                z = conv(z, edge_index)
            else:
                assert edge_index.shape[1] == edge_attr.shape[0], f'edge_attr shape: {edge_attr.shape} ' \
                                                                  f'and edge_index shape:{edge_index.shape} ' \
                                                                  f'do not match'
                z = conv(z, edge_index, edge_attr)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g

    def get_graph_embed(self, x, edge_index, batch, edge_attr=None):
        self.eval()
        with torch.no_grad():
            z, g = self.forward(x, edge_index, batch, edge_attr)
            return g


class Encoder_GraphCL(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder_GraphCL, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch, edge_attr=None):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_attr1 = aug1(x, edge_index, edge_attr)
        x2, edge_index2, edge_attr2 = aug2(x, edge_index, edge_attr)
        z, g = self.encoder(x, edge_index, batch, edge_attr)
        z1, g1 = self.encoder(x1, edge_index1, batch, edge_attr1)
        z2, g2 = self.encoder(x2, edge_index2, batch, edge_attr2)
        return z, g, z1, z2, g1, g2


class FC(nn.Module):
    def __init__(self, hidden_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)


class Encoder_InfoGraph(torch.nn.Module):
    def __init__(self, encoder, local_fc, global_fc):
        super(Encoder_InfoGraph, self).__init__()
        self.encoder = encoder
        self.local_fc = local_fc
        self.global_fc = global_fc

    def forward(self, x, edge_index, batch, edge_attr=None):
        z, g = self.encoder(x, edge_index, batch, edge_attr)
        return z, g

    def project(self, z, g):
        return self.local_fc(z), self.global_fc(g)
