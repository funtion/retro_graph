from tkinter import N
import numpy as np
from torchmetrics import Accuracy
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from torch_geometric.nn import MetaLayer
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch_scatter import scatter_mean
from torch_geometric import seed_everything

import fire
import os

class MLP(nn.Module):
    def __init__(self, dims, dropout) -> None:
        super().__init__()
        self.dims = dims
        self.dropout = dropout
        self.layers = nn.ModuleList([])
        # TODO: norm?
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
    
    def forward(self, x):
        for layer in self.layers:
            res = x
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if x.shape == res.shape:
                x = res + x
        return x

class EdgeModel(torch.nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.edge_mlp = MLP([dim*4, dim, dim], dropout)

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.node_mlp_1 = MLP([dim*2, dim, dim], dropout)
        self.node_mlp_2 = MLP([dim*3, dim, dim], dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)

class GlobalModel(torch.nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.global_mlp = MLP([dim*2, dim, dim], dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)

class RBF(torch.nn.Module):
    def __init__(self, low, high, dim) -> None:
        super().__init__()
        self.low = low
        self.high = high
        self.dim = dim
        self.centers = nn.Parameter(torch.arange(low, high, (high-low)/dim), requires_grad=False)
        self.gammap = ((high - low)/2)**2
    
    def forward(self, x): 
        x = x.unsqueeze(-1) # [B] => [B, 1]
        centers = self.centers.unsqueeze(0) # [dim], [1, dim]
        return torch.exp(-(x-centers)**2/self.gammap)


class GN(torch.nn.Module):
    def __init__(self, dim, dropout, n_layers) -> None:
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            self.layers.append(MetaLayer(EdgeModel(dim, dropout), NodeModel(dim, dropout), GlobalModel(dim, dropout)))

        self.output = nn.Linear(dim, 1)

        self.edge_embed = nn.Embedding(2, dim)
        self.fp_embed = nn.Linear(2048, dim//2, bias=False)
        self.rbf = RBF(0, 10, self.dim//2)

    def forward(self, data):
        # x = self.fp_embed(data.x)
        edge_attr = self.edge_embed(data.edge_attr.long())
        x_feature = data.x_feature
        mol_nodes = x_feature[:, 0] == 0
        rec_nodes = x_feature[:, 0] == 1

        x = torch.zeros((x_feature.shape[0], self.dim)).to(x_feature)
        x[mol_nodes, :self.dim//2] = self.fp_embed(data.x[mol_nodes]) ## TODO: unpack bits
        x[mol_nodes, self.dim//2:] = self.rbf(x_feature[mol_nodes, 4])

        x[rec_nodes, :self.dim//2] = self.rbf(x_feature[rec_nodes, 3])
        x[rec_nodes, self.dim//2:] = self.rbf(x_feature[rec_nodes, 4])

        bsz = data.num_graphs
        u = torch.zeros((bsz, self.dim)).to(x)
        for layer in self.layers:
            x, edge_attr, u = layer(x, data.edge_index, edge_attr, u, data.batch)
        return self.output(x)

def compute_order_loss(graph_batch, y_hat):
    loss = 0
    acc = 0
    n_graph = graph_batch.num_graphs
    total_graph = 0

    for i in range(n_graph):
        graph_y = graph_batch.y[graph_batch.batch == i]
        graph_y_hat = y_hat[graph_batch.batch == i]

        pos_idx = graph_y == 1
        neg_idx = graph_y == 0

        pos_pred = graph_y_hat[pos_idx].unsqueeze(0)
        neg_pred = graph_y_hat[neg_idx].unsqueeze(1)

        diff = pos_pred - neg_pred
        if diff.nelement() > 0:
            acc += (diff >0).float().mean()
            tau = 4
            score = diff - tau
            score[score > 0] = 0
            loss += torch.mean(-score)
            total_graph += 1

    return loss/total_graph, acc/total_graph



class Model(pl.LightningModule):
    def __init__(self, dim:int, dropout: int, n_layers: int, lr: float):
        super().__init__()
        self.gnn = GN(dim, dropout, n_layers)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.lr = lr

    def forward(self, data):
        return self.gnn(data)

    def training_step(self, data, batch_idx):
        y_hat = self(data)

        open_mask = data.y >=0
        y_hat_open = y_hat[open_mask].squeeze(1)
        y_open = data.y[open_mask]

        ce_loss = F.binary_cross_entropy_with_logits(y_hat_open, y_open)

        self.train_acc(y_hat_open > 0, y_open.long())

        order_loss, order_acc = compute_order_loss(data, y_hat.squeeze(1))

        assert order_loss.item() >= 0
        loss = ce_loss + order_loss

        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=data.num_graphs)
        self.log('train ce loss', ce_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=data.num_graphs)
        self.log('train or loss', order_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=data.num_graphs)
        self.log('train or acc', order_acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=data.num_graphs)
        return loss 

    def validation_step(self, data, batch_idx):
        y_hat = self(data)
        open_mask = data.y >=0
        y_hat_open = y_hat[open_mask].squeeze(1)
        y_open = data.y[open_mask].long()
        self.val_acc(y_hat_open> 0, y_open)

        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=data.num_graphs)
        order_loss, order_acc = compute_order_loss(data, y_hat.squeeze(1))

        self.log('val_or_acc', order_acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=data.num_graphs)
        return order_loss

    def test_step(self, data, batch_idx):
        y_hat = self(data)
        open_mask = data.y >=0
        y_hat_open = y_hat[open_mask].squeeze(1)
        y_open = data.y[open_mask].long()
        self.test_acc(y_hat_open> 0, y_open)

        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False,
                 on_epoch=True, batch_size=data.num_graphs)
        order_loss, order_acc = compute_order_loss(data, y_hat.squeeze(1))

        self.log('test_or_acc', order_acc, prog_bar=True, on_step=True, on_epoch=True, batch_size=data.num_graphs)
        return order_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ExponentialLR(optimizer, gamma=0.95),
            }
        }


class PlanDataModule(pl.LightningDataModule):
    def __init__(self, data, bsz: int, num_workers):
        super().__init__()
        for i in range(len(data)):
            data[i].x = torch.from_numpy(np.unpackbits(data[i].x.byte(), axis=1)).float()
        self.bsz = bsz
        self.num_workers = num_workers
        self.test = data[:1000]
        self.val = data[1000:2000]
        self.train = data[2000:]

    def train_dataloader(self):
        return DataLoader(self.train, self.bsz, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val, self.bsz, num_workers=2)
    
    def test_dataloader(self):
        return DataLoader(self.test, self.bsz, num_workers=2)

def main(train_data_path:str, save_path: str, bsz: int, dim:int, dropout: float, n_layers:int, max_epoch:int, lr: float, num_workers:int=6, n_shards:int=12):
    seed_everything(46)

    dataset = []
    for i in range(n_shards):
        shard_data = torch.load(os.path.join(train_data_path, f'mkdata_{i}', 'graph.pt'))
        dataset.extend(shard_data)

    datamodule = PlanDataModule(dataset, bsz, num_workers)

    # datamodule = pl.LightningDataModule(train_dataset, val_dataset, test_dataset, batch_size=bsz, num_workers=num_workers)

    model = Model(dim, dropout, n_layers, lr)

    gpus = torch.cuda.device_count()
    # strategy = pl.plugins.DDPSpawnPlugin(find_unused_parameters=False)
    checkpoint = pl.callbacks.ModelCheckpoint(save_path, monitor='val_or_acc')
    trainer = pl.Trainer(gpus=gpus, max_epochs=max_epoch, log_every_n_steps=10, callbacks=[checkpoint])

    # # Run learning rate finder
    # lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)

    # # Results can be found in
    # # lr_finder.results

    # # Plot with
    # # fig = lr_finder.plot(suggest=True)
    # # fig.show()

    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # print('new_lr', new_lr)

    # # update hparams of the model
    # model.lr = new_lr

    # Fit model
    # trainer.fit(model)

    trainer.fit(model, datamodule) 
    trainer.test(ckpt_path='best', datamodule=datamodule)

if __name__ == '__main__':
    fire.Fire(main)
