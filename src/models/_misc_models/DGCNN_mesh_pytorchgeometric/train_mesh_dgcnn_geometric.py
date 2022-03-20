import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import EdgeConv, global_max_pool
import numpy as np
#from pointnet.dataset import MMMDataset, ShapeNetDataset, ModelNetDataset
from pathlib import Path
import os

from models.DGCNN_pytorchgeometric.pointnet2_classification import MLP


class Net(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr='max', num_input_channels=3):
        super().__init__()

        self.conv1 = EdgeConv(MLP([2 * num_input_channels, 64, 64, 64]), aggr)
        self.conv2 = EdgeConv(MLP([2 * 64, 128]), aggr)
        self.lin1 = MLP([128 + 64, 1024])

        self.mlp = Seq(
            MLP([1024, 512]), Dropout(0.5), MLP([512, 256]), Dropout(0.5),
            Lin(256, out_channels))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        out = self.lin1(torch.cat([x1, x2], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_mesh_dgcnn_geometric(
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
    model = Net(num_class, k=20, num_input_channels=num_input_channels).to(device)
    best_model = Net(num_class, k=20, num_input_channels=num_input_channels).to(device)
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

        # with torch.no_grad():
        #     instance_acc, class_acc = test(classifier.eval(), testDataLoader)
        #
        #     if (instance_acc >= best_instance_acc):
        #         best_instance_acc = instance_acc
        #         best_epoch = epoch + 1
        #
        #     if (class_acc >= best_class_acc):
        #         best_class_acc = class_acc
        #     log_string('Test Instance Accuracy: %f, Class Accuracy: %f'% (instance_acc, class_acc))
        #     log_string('Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))
        #
        #     if (instance_acc >= best_instance_acc):
        #         logger.info('Save model...')
        #         savepath = str(checkpoints_dir) + '/best_model.pth'
        #         log_string('Saving at %s'% savepath)
        #         state = {
        #             'epoch': best_epoch,
        #             'instance_acc': instance_acc,
        #             'class_acc': class_acc,
        #             'model_state_dict': classifier.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #         }
        #         torch.save(state, savepath)
        #     global_epoch += 1

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