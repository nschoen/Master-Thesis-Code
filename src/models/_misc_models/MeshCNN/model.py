import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from options.val_options import ValOptions
from data import DataLoader
from models import create_model
from util.writer import Writer
from test import run_test
import torch
from src.base_model import BaseModel
from types import SimpleNamespace
import os
import numpy as np

# default_train_o = {
#     'dataset_mode': 'classification',
#     'ninput_edges': '',
#     'max_dataset_size': '',
#     'batch_size': 16
#     'arch': 'mconvnet',
#     'resblocks': 0,
#     'fc_n': 100,
#     'ncf': [16, 32, 32],
#     'pool_res': [1140, 780, 580],
#     'norm': 'batch',
#     'num_groups': 16,
#     'init_type': 'normal',
#     'init_gain': 0.02,
#     'num_threads': 3,
#     'gpu_ids': '0',
#     'name': 'debug',
#     'checkpoints_dir': './checkpoints',
#     'serial_batches': None,
#     'seed': None,
#     'export_folder': '',
# }

class Model(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)

        self.train_model_options = SimpleNamespace(**{
            **vars(TrainOptions(ignored_required=True).parse()),
            **{'checkpoints_dir': self.cls_path, 'cross_validation_set': False },
            **config['model_opt'],
        })

        self.val_model_options = SimpleNamespace(**{
            **vars(ValOptions(ignored_required=True).parse()),
            **{'checkpoints_dir': self.cls_path, 'serial_batches': True, 'cross_validation_set': False},
            **config['model_opt'],
        })

        self.test_model_options = SimpleNamespace(**{
            **vars(TestOptions(ignored_required=True).parse()),
            **{'checkpoints_dir': self.cls_path, 'serial_batches': True, 'cross_validation_set': False},
            **config['model_opt'],
        })
        print("self.test_model_options", self.test_model_options)

        torch.cuda.device(config['model_opt']['gpu_ids'][0])
        torch.cuda.set_device(config['model_opt']['gpu_ids'][0])

    def set_datasets(self, cross_validation_set=False):
        self.train_model_options.checkpoints_dir = self.cls_path
        self.train_model_options.cross_validation_set = cross_validation_set
        self.val_model_options.checkpoints_dir = self.cls_path
        self.val_model_options.cross_validation_set = cross_validation_set
        self.test_model_options.checkpoints_dir = self.cls_path
        self.test_model_options.cross_validation_set = cross_validation_set
        # pass

    def set_train_data_loader(self):
        self.train_loader = DataLoader(self.train_model_options)
        self.train_dataset = self.train_loader.dataset

    def set_val_data_loader(self):
        self.val_loader = DataLoader(self.val_model_options)
        self.val_dataset = self.val_loader.dataset

    def set_test_data_loader(self):
        self.test_loader = DataLoader(self.test_model_options)
        self.test_dataset = self.test_loader.dataset
        print("self.test_dataset", self.test_dataset)

    def get_model(self, num_classes=None):
        pass

    def train(self):
        self.set_train_data_loader()
        self.set_val_data_loader()
        self.set_test_data_loader()
        dataset = self.train_loader
        dataset_size = len(dataset)

        train_loss_hist = []
        val_loss_hist = []
        train_accuracy_hist = []
        val_accuracy_hist = []

        print('#training meshes = %d' % dataset_size)

        opt = self.train_model_options

        # if cls exist, take the latest checkpoint
        epoch = 1
        continue_train = False
        for root, dirs, files in os.walk(opt.checkpoints_dir):
            # continue_train = os.path.exists(os.path.join(opt.checkpoints_dir, 'latest_net.pth')) and len(files) > 2
            for file in files:
                if '_net.pth' in file and not 'latest' in file:
                    continue_train = True
                    file_epoch = int(file.replace('_net.pth', ''))
                    #print("file_epoch", file_epoch)
                    if file_epoch > epoch:
                        epoch = file_epoch

        print("epocjh", epoch)
        if continue_train:
            epoch += 1

        opt.continue_train = continue_train
        opt.epoch_count = epoch

        model = create_model(opt)
        writer = Writer(opt)
        total_steps = 0

        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            for i, data in enumerate(dataset):
                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size

                model.set_input(data)
                model.optimize_parameters()

                if total_steps % opt.print_freq == 0:
                    loss = model.loss
                    t = (time.time() - iter_start_time) / opt.batch_size
                    writer.print_current_losses(epoch, epoch_iter, loss, t, t_data)
                    writer.plot_loss(loss, epoch, epoch_iter, dataset_size)

                if i % opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, total_steps))
                    model.save_network('latest')

                iter_data_time = time.time()
            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                model.save_network('latest')
                model.save_network(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate()
            if opt.verbose_plot:
                writer.plot_model_wts(model, epoch)

            #if epoch % opt.run_test_freq == 0:
            acc = run_test(epoch, self.val_model_options)
            writer.plot_acc(acc, epoch)

            #train_loss_hist = checkpoint['train_loss'] if 'train_loss' in checkpoint else []
            #val_loss_hist = checkpoint['test_loss'] if 'test_loss' in checkpoint else []
            val_accuracy_hist.append(acc)

        writer.close()

        #accuracy, y_true, y_pred = self.get_and_save_test_results(model, acc_test_hist=val_accuracy_hist)
        self.get_and_save_test_results(
            model,
            test=True,
            dir='test_latest_model',
        )

        return None, None, None
        #return accuracy, y_true, y_pred

    @torch.no_grad()
    def test(self, model, loader):
        model.net.eval()
        correct = 0
        y_target = []
        y_pred = []

        for data in loader:
            model.set_input(data)
            out = model.forward()
            pred = out.data.max(1)[1]
            targets = model.labels
            num_correct = pred.eq(targets).sum()
            correct += num_correct.cpu().data.numpy().astype(np.float).tolist()
            y_target = y_target + targets.cpu().data.numpy().flatten().tolist()
            y_pred = y_pred + pred.cpu().data.numpy().flatten().tolist()

        #return correct, len(loader.dataset), correct / len(loader.dataset), y_target, y_pred
        return correct, len(loader.dataset), correct / len(loader.dataset), y_target, y_pred, 0.0
