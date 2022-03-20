import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import (ASAPooling, GraphConv, global_mean_pool,
                                JumpingKnowledge)
from pathlib import Path
from torch_geometric.data import DataLoader
import os
from src.base_model import BaseModel

class ASAP(torch.nn.Module):
    def __init__(self, num_classes, num_input_channels, num_layers=10, hidden=64, ratio=0.9, dropout=0.1):
        super(ASAP, self).__init__()
        self.conv1 = GraphConv(num_input_channels, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend([
            ASAPooling(hidden, ratio, dropout=dropout)
            for i in range((num_layers) // 2)
        ])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = None
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, edge_weight, batch, _ = pool(
                    x=x, edge_index=edge_index, edge_weight=edge_weight,
                    batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_mesh_asap_geometric(
        outf='',
        train_dataset=None,
        test_dataset=None,
        nepoch=200,
        batch_size=32,
        learning_rate=0.001,
        decay_rate=0.5,
        optimizer_type='Adam',
        vertex_normals=False):

    cls_dir = Path(outf)
    cls_dir.mkdir(exist_ok=True)
    savepath = str(cls_dir) + '/best_model.pth'

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4)

    num_input_channels = 3
    if vertex_normals:
        num_input_channels = 6

    num_class = len(test_dataset.classes)
    model = ASAP(num_classes=num_class, num_input_channels=num_input_channels).to(device)
    best_model = ASAP(num_classes=num_class, num_input_channels=num_input_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=decay_rate)

    #best_instance = None
    best_instance_acc = 0.0

    start_epoch = 1

    if os.path.exists(savepath):
        checkpoint = torch.load(savepath)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['last_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_instance_acc = checkpoint['best_instance_acc']
        best_model.load_state_dict(checkpoint['best_model_state_dict'])

    model.cuda()
    best_model.cuda()

    for epoch in range(start_epoch, nepoch):
        model.train()

        total_loss = 0
        scheduler.step()

        #for data in dataloader:
        for data in train_dataloader:
            #points, target = data
            #points = points.transpose(2, 1)
            #points, target = points.cuda(), target.cuda()
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            # bla = data.y.view(batch_size, -1)

            loss = F.nll_loss(out, data.y)  # .view(batch_size, -1))
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        loss = total_loss / len(train_dataset)

        test_acc = test(model, test_dataloader)
        print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
            epoch, loss, test_acc))

        if (test_acc >= best_instance_acc):
            best_instance_acc = test_acc
            best_model = model

        state = {
            'epoch': epoch,
            'best_instance_acc': best_instance_acc,
            'last_model_state_dict': model.state_dict(),
            'best_model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(state, savepath)

    return best_model


def test(model, loader):
    model.eval()

    with torch.no_grad():
        correct = 0
        for data in loader:
            data = data.to(device)
            with torch.no_grad():
                pred = model(data).max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        return correct / len(loader.dataset)

# class Model(BaseModel):
#     def get_model(self, num_classes=9):
#         self.model = Net(num_classes, k=20)
