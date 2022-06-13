import pickle

import torch
import copy
import numpy as np
import GCL.losses as L
import GCL.augmentors as A
from tqdm import tqdm
from torch.optim import Adam
from GCL.models import DualBranchContrast, SingleBranchContrast
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from evaluation.models.gin.gin_pyg import GConv, Encoder_GraphCL, Encoder_InfoGraph, FC
from data_utils import make_dataset_ready_to_save, make_dataset_from_saved_format


def eval_graphcl(encoder_model, contrast_model, dataloader):
    encoder_model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for data in dataloader:
            _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch, data.edge_attr)
            g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
            loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
            epoch_loss += loss.item()
    return epoch_loss


def train_graphcl(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        optimizer.zero_grad()
        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch, data.edge_attr)
        g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def get_graphcl_model(dataset, args):
    dataloader = DataLoader(dataset, batch_size=64)

    aug1 = A.Identity()
    aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                           A.NodeDropping(pn=0.1),
                           # A.FeatureMasking(pf=0.1),
                           A.EdgeRemoving(pe=0.1)], 1)

    gconv = GConv(args).to(args.device)
    encoder_model = Encoder_GraphCL(encoder=gconv, augmentor=(aug1, aug2)).to(args.device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(args.device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=args.epochs, desc='(T)') as pbar:
        for epoch in range(1, args.epochs+1):
            loss = train_graphcl(encoder_model, contrast_model, dataloader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    print('evaluating the loss: ')
    eval_losses = []
    with tqdm(total=10, desc='(T)') as pbar:
        for epoch in range(10):
            eval_losses.append(eval_graphcl(encoder_model, contrast_model, dataloader))
            pbar.set_postfix({'loss': loss})
            pbar.update()

    print('\nevaluation loss: ', np.mean(eval_losses))

    torch.save(gconv, f'saved_models/{args.model_name}')
    with open(f'saved_models/{args.model_name}_loss', 'wb+') as f:
        pickle.dump(np.mean(eval_losses), f)
    return gconv, np.mean(eval_losses)


def train_infograph(encoder_model, contrast_model, dataloader, optimizer, args):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to(args.device)
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        z, g = encoder_model(data.x, data.edge_index, data.batch, data.edge_attr)
        z, g = encoder_model.project(z, g)
        loss = contrast_model(h=z, g=g, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss


def get_infograph_model(dataset, args):
    dataloader = DataLoader(dataset, batch_size=64)

    gconv = GConv(args).to(args.device)
    fc1 = FC(hidden_dim=args.hidden_dim * args.num_layers)
    fc2 = FC(hidden_dim=args.hidden_dim * args.num_layers)
    encoder_model = Encoder_InfoGraph(encoder=gconv, local_fc=fc1, global_fc=fc2).to(args.device)
    contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(args.device)

    optimizer = Adam(encoder_model.parameters(), lr=0.01)

    with tqdm(total=args.epochs, desc='(T)') as pbar:
        for epoch in range(1, args.epochs+1):
            loss = train_infograph(encoder_model, contrast_model, dataloader, optimizer, args)
            pbar.set_postfix({'loss': loss})
            pbar.update()

    torch.save(gconv, f'saved_models/{args.model_name}')
    with open(f'saved_models/{args.model_name}_loss', 'wb+') as f:
        pickle.dump(loss, f)
    return gconv, loss


def get_model(dataset, args):
    if not isinstance(dataset[0], Data):
        pyg_dataset = make_dataset_ready_to_save(copy.deepcopy(dataset), parallel=args.is_parallel)
        make_dataset_from_saved_format(pyg_dataset, args.deg_feats, args.clus_feats, args.orbit_feats)
    else:
        pyg_dataset = dataset  # already pytorch geometric dataset

    args.input_dim = 1
    if pyg_dataset[0].x is not None:
        args.input_dim = pyg_dataset[0].x.shape[1]
    args.edge_dim = None
    if pyg_dataset[0].edge_attr is not None:
        args.edge_dim = pyg_dataset[0].edge_attr.shape[1]

    if args.feature_extractor == 'graphcl':
        return get_graphcl_model(pyg_dataset, args)
    elif args.feature_extractor == 'infograph':
        return get_infograph_model(pyg_dataset, args)
    elif args.feature_extractor == 'gin-random':
        gconv = GConv(args).to(args.device)
        torch.save(gconv, f'saved_models/{args.model_name}')
        return gconv, None
    else:
        raise ValueError('model name not found')