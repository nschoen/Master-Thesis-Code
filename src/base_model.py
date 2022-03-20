import os
import sys
import json
from os.path import isdir, join, exists, abspath, dirname
from src.utils.show_cls_confusion_matrix_experiment import save_confusion_matrix
import csv
from src.data.dataset_loader import get_datasets, get_dir_dataset
import torch
import torch.nn.functional as F
import copy
import sklearn
import torch_geometric
import matplotlib.pyplot as plt
import src.model_loader as model_loader
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from types import SimpleNamespace
from tqdm import tqdm
import time
from src.explanation_methods.grad_cam_pn2 import show_point_grad_cam_pn2_explanation, GradCamPointnet2
from src.explanation_methods.grad_cam_pn1 import show_point_grad_cam_pn1_explanation, GradCamPointNet1
from src.explanation_methods.point_rise import show_point_rise_explanation, PointRISE
from src.explanation_methods.mesh_rise import show_mesh_rise_explanation, MeshRISE
from src.explanation_methods.grad_cam_asap import mesh_grad_cam_asap, GradCamAsap
from src.utils.point_dropping_experiment import PointDropExperiment
import numpy
import random


DEFAULT_CONFIG = {
    'model': None,
    'result_paths': './results',
    'pointnet_feature_transform': False,
    'pn2_model_type': 'pointnet2_cls_msg',
    'pn2_first_SA_n_sample_factor': 1,
    'device': 'cuda',
    'batch_size': 32,
    'epochs': 250,
    'workers': 4,
    'optimizer_type': 'Adam',  # or SGD
    'learning_rate': 0.001,
    'weight_decay_rate': 1e-4,
    'decay_step_size': 50,
    'decay_rate': 0.5,
    'results_path': None,
    'results_cls_path': None,
    'set_results_path': True,
    'num_classes': None,
    'dataset': {
        'dataset': 'eh_mmm_custom_flanges',
        'type': 'pointcloud',
        'models_csv': 'toy9.csv',
        'filter_classes': False,
        'filename_filter': False,
        'mesh_dir': 'data',  # which mesh_dir to read samples from for train and val, can be an array
        'mesh_dir_select_random': None,
        'shuffle': True,

        # point cloud
        'point_sampling_method': 'poisson',
        'n_points': 1000,
        'n_min_surface_sampled': 0,
        'data_augmentation_jitter': False,
        'normalize': True,
        'normalize_type': 'mean',
        'feature_transform': False,  # does not belong in dataset config?!
        'rotate_axes': 'y',
        'move_along_axis': '',
        'normalize_scale_factor': 1.0,
        'point_normals': False,

        # meshes
        'dataset_mesh': False,
        'max_number_triangles': None,
        'vertex_xyz': True,
        'vertex_normals': False,
        'vertex_one_vector': False,
        'edge_relative_xyz': False,
        'edge_relative_angle_and_ratio': False,
        'edge_length': False,
        'edge_vertice_normals_cosine': False,
        'transform': None,
        'test_transform': None,

        # mesh edge features
        'num_aug': 20,
        'scale_verts': True,
        'flip_edges': 0.2,
        'slide_verts': 0.2,
        'mesh_edge_normals': False,
    },
}

TEST_DATASET_CONFIG_OVERRIDE = {
    'rotate_axes': '',
    'move_along_axis': '',
    'data_augmentation_jitter': False,
    'num_aug': 0,
    "scale_verts": False,
    "slide_verts": 0.0,
    "flip_edges": 0.0,
}

ROOT = abspath('/'.join(dirname(__file__).split('/')[0:-1]))

class BaseModel:
    DATASETS_DIR = join(ROOT, 'datasets')
    EXPLAIN_SAMPLES_ROOT_DIR = join(ROOT, 'src', 'explanation_methods', 'samples')

    def __init__(self, config, dataset_transform=None):
        self.original_config = config

        self.config = {
            **DEFAULT_CONFIG,
            **config,
        }

        if 'dataset' in config:
            self.config['dataset'] = SimpleNamespace(**{
                **DEFAULT_CONFIG['dataset'],
                **config['dataset'],
            })
        else:
            self.config['dataset'] = SimpleNamespace(**DEFAULT_CONFIG['dataset'])

        self.config = SimpleNamespace(**self.config)

        if self.config.set_results_path:
            results_path, results_cls_path = self.get_experiment_result_path()
            self.results_path = results_path
            self.cls_path = results_cls_path
        else:
            self.results_path = self.config.results_path
            self.cls_path = self.config.results_cls_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.best_model = None
        #self.dataset_transform = dataset_transform if dataset_transform else self.config.dataset.transform
        self.reset_seeds()

    def reset_seeds(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        numpy.random.seed(0)
        random.seed(0)

    def get_experiment_result_path(self, subdir=''):
        experiments_path = sys.argv[0]
        if self.config.result_paths == None:
            results_path = abspath(experiments_path.replace('.py', ''))
        else:
            if self.config.result_paths.startswith('/'):
                results_path = self.config.result_paths
            else:
                dir_path = abspath(dirname(experiments_path))
                results_path = join(dir_path, self.config.result_paths)
            basename = os.path.basename(experiments_path).replace('.py', '')
            if results_path.endswith('/'):
                results_path = join(results_path, basename)
        #experiment_result_paths = os.path \
        #    .relpath(os.path.abspath(experiments_path), BaseModel.EXPERIMENTS_DIR) \
        #    .replace('.py', '').split('/')
        #results_path = join(BaseModel.RESULTS_DIR, *experiment_result_paths)
        if subdir and len(subdir) > 0:
            results_path = join(results_path, subdir)
        print("Setting experiment results path:", results_path)
        cls_path = join(results_path, 'cls')

        if not isdir(cls_path):
            os.makedirs(cls_path, exist_ok=True)

        return results_path, cls_path

    def set_datasets(self, cross_validation_set=False):
        if hasattr(self.config.dataset, "test_mesh_dir"):
            TEST_DATASET_CONFIG_OVERRIDE['mesh_dir'] = self.config.dataset.test_mesh_dir
        if hasattr(self.config.dataset, "test_mesh_dir_select_random"):
            TEST_DATASET_CONFIG_OVERRIDE['mesh_dir_select_random'] = self.config.dataset.test_mesh_dir_select_random
        if hasattr(self.config.dataset, "test_scale_verts"):
            TEST_DATASET_CONFIG_OVERRIDE['scale_verts'] = self.config.dataset.test_scale_verts
        if hasattr(self.config.dataset, "test_slide_verts"):
            TEST_DATASET_CONFIG_OVERRIDE['slide_verts'] = self.config.dataset.test_slide_verts
        if hasattr(self.config.dataset, "test_num_aug"):
            TEST_DATASET_CONFIG_OVERRIDE['num_aug'] = self.config.dataset.test_num_aug
        train_dataset, val_dataset, test_dataset = get_datasets(BaseModel.DATASETS_DIR,
                                                                self.config.dataset,
                                                                test_dataset_override=TEST_DATASET_CONFIG_OVERRIDE,
                                                                cross_validation_set=cross_validation_set)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def is_geometric(self):
        type = self.config.dataset.type
        return type in ['mesh', 'pointcloud_geoemetric',
                        'pointcloud_edge_unit', 'mesh_edge_unit',
                        'mesh_face_unit', 'mesh_face_unit_edge_attributes']

    def set_train_data_loader(self):
        dataloader_class = torch_geometric.data.DataLoader if self.is_geometric() else torch.utils.data.DataLoader
        self.train_loader = dataloader_class(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.dataset.shuffle,
            num_workers=self.config.workers)

    def set_val_data_loader(self):
        dataloader_class = torch_geometric.data.DataLoader if self.is_geometric() else torch.utils.data.DataLoader
        self.val_loader = dataloader_class(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.dataset.shuffle,
            num_workers=self.config.workers)

    def set_test_data_loader(self, batch_size_override=None):
        dataloader_class = torch_geometric.data.DataLoader if self.is_geometric() else torch.utils.data.DataLoader
        def worker_init_fn(worker_id):
            random.seed(0)
        self.test_loader = dataloader_class(
            self.test_dataset,
            batch_size=self.config.batch_size if batch_size_override is None else batch_size_override,
            shuffle=False,  # self.config.dataset.shuffle,
            num_workers=self.config.workers,
            worker_init_fn=worker_init_fn,
        )

    def get_best_model(self):
        if self.best_model:
            return self.best_model

        best_model_path = join(self.cls_path, 'best_model.pth')
        if not exists(best_model_path):
            raise Exception('The model must be trained first')

        checkpoint = torch.load(best_model_path)
        num_classes = 4
        if 'num_classes' in checkpoint:
            num_classes = checkpoint['num_classes']
        elif self.config.num_classes:
            num_classes = self.config.num_classes

        self.get_model(num_classes=num_classes)
        best_model = self.model
        #best_model.load_state_dict(checkpoint['last_model_state_dict'])
        best_model.load_state_dict(checkpoint['best_model_state_dict'])
        best_model.to(self.config.device)
        self.best_model = best_model
        return best_model

    def run_parameter_tuning(self, tune_config={}, experiment_name=False):
        experiment_name = f"tune_{experiment_name if experiment_name else 'default'}"
        resume = False
        if os.path.isdir(join(self.results_path, experiment_name)):
            resume = "LOCAL"

        scheduler = ASHAScheduler(
            #time_attr='training_iteration',
            metric='val_accuracy',
            mode='max',
            max_t=300)
        analysis = tune.run(
            self.tune_training_function,
            scheduler=scheduler,
            #metric="test_accuracy",
            local_dir=self.results_path,
            #mode="max",
            name=experiment_name if experiment_name else 'tune_default',
            resume=resume,
            #scheduler=sched,
            # stop={
            #     #"mean_accuracy": 0.95,
            #     "training_iteration": 3, # if args.smoke_test else 20,
            # },
            resources_per_trial={
                "cpu": 0,
                "gpu": 1,
            },
            #num_samples=2,  # if args.smoke_test else 20,
            config={
                "tune_config": tune_config,
                "model_config": self.original_config,
            })
        # config={
        #     "model_config": vars(self.original_config),
        #     "lr": tune.grid_search([0.001, 0.01]),  # , 0.1]),
        #     # "num_cpus_for_driver": 2,
        #     # "num_cpus_per_worker": 2,
        #     # "num_workers": 2,
        # })

        print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="min"))

        dest_file = join(self.results_path, 'tuning_results.txt')
        if os.path.exists(dest_file):
            os.remove(dest_file)
        with open(dest_file, 'w') as text_file:
            #text_file.write(str(analysis.get_best_config(metric="test_accuracy", mode="max")))
            json.dump(analysis.get_best_config(metric="val_accuracy", mode="max"), text_file, indent=2)
            #text_file.write(str(analysis.results))

        # Get a dataframe for analyzing trial results.
        #df = analysis.results_df

    @staticmethod
    def tune_training_function(config, checkpoint_dir=None):
        # https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/mnist_pytorch_trainable.py
        #print("config['tune_config']", config['tune_config'])
        model_config = {
            **config['model_config'],
            **config['tune_config'],
            "dataset": {
                **config['model_config']['dataset'],
                **(config['tune_config']['dataset'] if 'dataset' in config['tune_config'] else {}),
            }
        }
        #use_cuda = config.get("use_gpu") and torch.cuda.is_available()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(os.environ["CUDA_VISIBLE_DEVICES"])
        #print("device", device)
        model_config['device'] = device
        #model_config['learning_rate'] = config["lr"]

        # Obtain a checkpoint directory
        #with tune.checkpoint_dir(step=0) as checkpoint_dir:
        model_config['set_results_path'] = False
        model_config['results_path'] = checkpoint_dir
        model_config['results_cls_path'] = checkpoint_dir
        model = model_loader.load_model(model_config)
        model.train(report_tune=True)

    def run_cross_validation(self):
        accuracies = []
        class_acc = {}

        for i in range(5):  # 4-fold cross-validation
            self.model = None
            self.best_model = None
            self.accuracy = None
            results_path, results_cls_path = self.get_experiment_result_path(subdir=f"run-{i}")
            self.results_path = results_path
            self.cls_path = results_cls_path
            self.set_datasets(cross_validation_set=i)
            self.set_train_data_loader()
            self.set_val_data_loader()
            self.set_test_data_loader()
            accuracy, y_true, y_pred = self.train()
            accuracies.append(accuracy)

            class_pred = {}
            for i in range(len(y_true)):
                true = y_true[i]
                pred = y_pred[i]
                if true not in class_pred:
                    class_pred[true] = [0, 0]
                class_pred[true][1] = class_pred[true][1] + 1
                class_pred[true][0] = class_pred[true][0] + (1 if true == pred else 0)

            for k, val in class_pred.items():
                print("k", k, "val", val)
                if not k in class_acc:
                    class_acc[k] = []
                class_acc[k].append(val[0] / val[1])

        results_path, results_cls_path = self.get_experiment_result_path()

        # save accuracies
        with open(join(results_path, 'accuracies.txt'), 'w') as text_file:
            mean_acc = 0
            for i, acc in enumerate(accuracies):
                mean_acc += acc
                text_file.write(f"Accuracy {i}: {acc}\n")
            mean_acc = mean_acc / 5
            text_file.write(f"Mean Accuracy: {mean_acc}\n")

        # plot class accuracies by k-fold
        plt.clf()
        colors = ['red', 'blue', 'green', 'black', 'yellow']
        for i in range(4):
            x = []
            y = []
            for k, val in class_acc.items():
                x.append(k)
                y.append(val[i])
            plt.scatter(x, y, s=100, c=colors[i], label=f"Fold {i}")
        plt.title('Class accuracies by k-fold')
        plt.legend()
        plt.ylabel('Accuracy')
        plt.xlabel('Class')
        plt.savefig(join(results_path, 'crossval-class-accuracies.png'))
        plt.clf()

        # boxplot
        plt.boxplot([accuracies], meanline=True, labels=['Model'])
        plt.title('Total accuracies by k-fold')
        plt.ylabel('Accuracy')
        plt.xlabel('Model')
        plt.savefig(join(results_path, 'crossval-accuracies.png'))
        plt.clf()

    def train(self, report_tune=False):
        if not self.test_loader:
            self.set_datasets()
            self.set_train_data_loader()
            self.set_val_data_loader()
            self.set_test_data_loader()

        self.get_model(num_classes=len(self.train_dataset.classes))
        model = self.model

        # 3. Log gradients and model parameters
        #wandb.watch(model)

        start_epoch = 1
        best_instance_acc = 0.0
        train_loss_hist = []
        test_loss_hist = []  # it's actually the validation_loss
        train_accuracy_hist = []
        test_accuracy_hist = []
        pre_training_time = 0.0

        best_model = copy.deepcopy(model)

        savepath = None
        if self.cls_path:  # may not exist when tuning
            savepath = join(self.cls_path, 'best_model.pth')

        if savepath and exists(savepath):
            print("Loading model from", savepath)
            checkpoint = torch.load(savepath, map_location=self.config.device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['last_model_state_dict'])
            best_instance_acc = checkpoint['best_instance_acc']
            best_model.load_state_dict(checkpoint['best_model_state_dict'])
            train_loss_hist = checkpoint['train_loss'] if 'train_loss' in checkpoint else []
            test_loss_hist = checkpoint['test_loss'] if 'test_loss' in checkpoint else []
            train_accuracy_hist = checkpoint['train_accuracy'] if 'train_accuracy' in checkpoint else []
            test_accuracy_hist = checkpoint['test_accuracy'] if 'test_accuracy' in checkpoint else []
            pre_training_time = checkpoint['training_time'] if 'training_time' in checkpoint else 0.0

        model = model.to(self.config.device)
        best_model = best_model.to(self.config.device)

        if start_epoch < self.config.epochs:
            # initialize optimizer after model has been moved to cuda
            if self.config.optimizer_type == 'Adam':
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=self.config.weight_decay_rate,
                )
            elif self.config.optimizer_type == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            else:
                raise Exception('No valid optimizer type')
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.decay_step_size,
                                                        gamma=self.config.decay_rate)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
            # total_loss = F.nll_loss(pred, target)

            if savepath and exists(savepath):
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            torch.autograd.set_detect_anomaly(True)
            start_time = time.time()

        for epoch in range(start_epoch, self.config.epochs):
            correct = 0.0
            total_loss = 0.0

            # for data in self.train_loader:
            for batch_id, data in tqdm(enumerate(self.train_loader, 0), total=len(self.train_loader), smoothing=0.9):
                input_data, target = self.get_data(data)

                optimizer.zero_grad()

                model = model.train()
                output = model(input_data)

                # if tuple, the first value is the predicted value
                pred = output[0] if type(output) == tuple else output
                loss = self.loss_criterion(pred, target, complete_output=output)
                loss.backward()

                total_loss += pred.size(0) * loss.item()
                optimizer.step()

                pred_ids = pred.max(dim=1)[1]
                correct += pred_ids.eq(target.view(-1)).sum().item()

            scheduler.step()
            train_accuracy_hist.append(correct / len(self.train_loader.dataset))
            #wandb.log({"acc": correct / len(self.train_loader.dataset)})

            epoch_loss = total_loss / len(self.train_loader)
            avg_loss = total_loss / len(self.train_loader.dataset)
            train_loss_hist.append(avg_loss)
            #wandb.log({"loss": avg_loss})

            val_res = self.test(model, self.val_loader) # [2, 5]
            val_acc = val_res[2]
            val_loss = val_res[5]
            #wandb.log({"val_acc": val_acc})
            #wandb.log({"val_loss": val_loss})
            test_accuracy_hist.append(val_acc)
            test_loss_hist.append(val_loss)
            training_time = pre_training_time + (time.time() - start_time)
            average_epoch_time = training_time / epoch
            print('Epoch {:03d}, Avg Loss: {:.4f}, Avg Batch Loss: {:.4f}, Test: {:.4f}, Test Avg Loss: {:.4f}, Training Time: {:.4f}, Avg Epoch Time: {:.4f}'.format(
                epoch, avg_loss, epoch_loss, val_acc, val_loss, training_time, average_epoch_time))

            if report_tune:
                tune.report(val_accuracy=val_acc, loss=val_loss)

                with tune.checkpoint_dir(step=epoch - 1) as checkpoint:
                    print("checkpoint epoch", epoch, "is", checkpoint)
                    self.cls_path = checkpoint
                    self.results_path = checkpoint
                    savepath = join(self.cls_path, 'best_model.pth')

            if (val_acc >= best_instance_acc):
                best_instance_acc = val_acc
                best_model = copy.deepcopy(model)


            state = {
                'epoch': epoch,
                'best_instance_acc': best_instance_acc,
                'last_model_state_dict': model.state_dict(),
                'best_model_state_dict': best_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'num_classes': len(self.train_dataset.classes),
                'train_loss': train_loss_hist,
                'test_loss': test_loss_hist,
                'train_accuracy': train_accuracy_hist,
                'test_accuracy': test_accuracy_hist,
                'training_time': training_time,
                'avg_epoch_time': average_epoch_time,
            }
            torch.save(state, savepath)
            torch.save(best_model.state_dict(), savepath.replace('.pth', '_only_state_dict.pth'))
            torch.save(best_model.state_dict(), savepath.replace('.pth', f"_{epoch}.pth"))

        self.plot_train_info(
            loss_train_hist=train_loss_hist,
            loss_test_hist=test_loss_hist,
            acc_train_hist=train_accuracy_hist,
            acc_test_hist=test_accuracy_hist)

        best_model.eval()
        model.eval()

        self.model = model
        self.best_model = best_model

    def evaluate(self, skip_validation=False, skip_best_model=False):
        self.best_model.eval()
        self.model.eval()

        if not skip_validation:
            print("Evaluating best model on validation set:")
            self.reset_seeds()
            accuracy, y_true, y_pred = self.get_and_save_test_results(
                self.best_model,
            )

        if not skip_best_model:
            print("Evaluating best model on test set:")
            self.reset_seeds()
            self.get_and_save_test_results(
                self.best_model,
                test=True,
                dir='test_best_model',
            )

        print("Evaluating last model on test set (thesis metric):")
        self.reset_seeds()
        self.get_and_save_test_results(
            self.model,
            test=True,
            dir='test_latest_model',
        )

        print("results saved in:", self.results_path)

        # return accuracy, y_true, y_pred

    @torch.no_grad()
    def test(self, model, loader):
        model = model.eval()

        with torch.no_grad():
            correct = 0.0
            total_loss = 0.0
            y_target = []
            y_pred = []

            for data in loader:
                input_data, target = self.get_data(data)
                output = model(input_data)
                pred = output[0] if type(output) == tuple else output  # if tuple, the first value is the predicted value
                pred_ids = pred.max(dim=1)[1]
                correct += pred_ids.eq(target.view(-1)).sum().item()
                loss = self.loss_criterion(pred, target, complete_output=output)
                total_loss += pred.size(0) * loss.cpu().item()

                y_target = y_target + target.cpu().data.numpy().flatten().tolist()
                y_pred = y_pred + pred_ids.cpu().data.numpy().flatten().tolist()

        print("accuracy", correct / len(loader.dataset))

        return correct, len(loader.dataset), correct / len(loader.dataset), y_target, y_pred, total_loss / len(loader.dataset)

    @torch.no_grad()
    def inference(self, x):
        self.get_best_model()
        model = self.best_model
        model.eval()

        x = torch.unsqueeze(x, 0)
        x = x.to(self.config.device)

        output = model(x)[0]
        pred = output[0] if type(output) == tuple else output
        pred_idx = pred.max(dim=1)[1].item()
        return pred_idx

    def explain_samples(self,
                        samples_dir,
                        xai_method='gradcam',
                        ds_config_override={},
                        visualize_mesh=False,
                        discrete_heatmap=False,
                        discrete_heatmap_threshold=0.5,
                        asap_grad_cam_intermediate_layers=False,
                        explain_class_id=None):
        self.config.dataset = SimpleNamespace(**{
            **vars(self.config.dataset),
            "rotate_axes": "",
            "move_along_axis": "",
            "data_augmentation_jitter": False,
            **ds_config_override,
        })

        samples_path = join(self.EXPLAIN_SAMPLES_ROOT_DIR, samples_dir)
        dataset = get_dir_dataset(samples_path, self.config.dataset)
        dataloader_class = torch_geometric.data.DataLoader if self.is_geometric() else torch.utils.data.DataLoader
        loader = dataloader_class(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4)

        # load best model
        model = self.get_best_model()
        model.eval()
        model = model.to(self.config.device)

        self.reset_seeds()

        if self.config.model == 'PointNet':
            if xai_method == 'gradcam':
                show_point_grad_cam_pn1_explanation(self,
                                                    model,
                                                    loader,
                                                    visualize_mesh=visualize_mesh,
                                                    discrete_heatmap=discrete_heatmap,
                                                    discrete_heatmap_threshold=discrete_heatmap_threshold)
            elif xai_method == 'rise':
                show_point_rise_explanation(self,
                                            ROOT,
                                            model,
                                            loader,
                                            discrete_heatmap=discrete_heatmap,
                                            discrete_heatmap_threshold=discrete_heatmap_threshold)
            else:
                print("Unknown XAI method")
        elif self.config.model == 'PointNet2':
            if xai_method == 'gradcam':
                show_point_grad_cam_pn2_explanation(self,
                                                    model,
                                                    loader,
                                                    visualize_mesh=visualize_mesh,
                                                    discrete_heatmap=discrete_heatmap,
                                                    discrete_heatmap_threshold=discrete_heatmap_threshold,
                                                    explain_class_id=explain_class_id)
            elif xai_method == 'rise':
                show_point_rise_explanation(self,
                                            ROOT,
                                            model,
                                            loader,
                                            explain_class_id=explain_class_id)
            else:
                print("Unknown XAI method")
        elif self.config.model in ['EdgeASAP', 'EdgeGlobal']:
            if xai_method == 'gradcam':
                mesh_grad_cam_asap(self,
                                   model,
                                   loader,
                                   discrete_heatmap=discrete_heatmap,
                                   discrete_heatmap_threshold=discrete_heatmap_threshold,
                                   intermedia_layers=asap_grad_cam_intermediate_layers)
            elif xai_method == 'rise':
                show_mesh_rise_explanation(self,
                          ROOT,
                          model,
                          loader,
                          dataset,
                          discrete_heatmap=discrete_heatmap,
                          discrete_heatmap_threshold=discrete_heatmap_threshold)
            else:
                print("Unknown XAI method")
        else:
            print("No XAI method for this model implemented yet")

    def point_dropping_experiment(self, xai_method='gradcam', name=None,
                                  update_cam=False,
                                  steps=100, num_drops=None):
        print("start point dropping experiment")
        self.set_datasets()
        self.set_test_data_loader(batch_size_override=1)
        dataset = self.test_dataset
        loader = self.test_loader

        model = self.model
        model.eval()
        model = model.to(self.config.device)

        is_mesh = False
        remove_edges = False

        if self.config.model in ['PointNet', 'PointNet2']:
            if xai_method == 'rise':
                rise = PointRISE(model, gpu_batch=self.config.batch_size)
                class RISEWrapper():
                    def __init__(self, device):
                        self.classifier_output = None
                        self.device = device

                    def __call__(self, input, target_idx):
                        rise.generate_masks(input[0],
                                            3000,
                                            s=[16, 32, 64],
                                            cache_name=dataset.models[self.model_index]['filename'].split('.')[0],
                                            root_dir=ROOT)
                        exp = rise.forward(input[0])[target_idx].cpu().numpy()
                        self.classifier_output = model(input)[0]
                        return exp, None, None

                    def set_iteration_index(self, index):
                        self.model_index = index

                grad_cam = RISEWrapper(self.config.device)
            else:
                if self.config.model == 'PointNet':
                    if xai_method == 'gradcam':
                        grad_cam = GradCamPointNet1(model)
                    else:
                        print("Unknown XAI method")
                elif self.config.model == 'PointNet2':
                    if xai_method == 'gradcam':
                        grad_cam = GradCamPointnet2(model)
                    else:
                        print("Unknown XAI method")
                else:
                    print("No XAI method for this model implemented yet")
        elif self.config.model.startswith('asap'):
            is_mesh = True
            remove_edges = True
            if xai_method == 'gradcam':
                grad_cam = GradCamAsap(model, intermedia_layers=False)
            elif xai_method == 'rise':
                rise = MeshRISE(model, self.config.dataset.n_points, # device=self.config.device,
                                 gpu_batch=self.config.batch_size)

                class RISEWrapper():
                    def __init__(self, device):
                        self.classifier_output = None
                        self.device = device

                    def __call__(self, input, target_idx):
                        s = [16, 32]  # [16, 32, 64]
                        s_id = "-".join([str(sx) for sx in s])
                        rise.generate_masks(input.pos.cpu(),
                                            2000,
                                            s=s,
                                            p1=0.1,
                                            cache_name=dataset.models[self.model_index]['filename'].split('.')[0] + s_id,
                                            root_dir=ROOT)
                        exp = rise.forward(input,
                                           remove_edges=True,
                                           device=self.device,
                                           root_dir=ROOT,
                                           cache_name=dataset.models[self.model_index]['filename'].split('.')[0] + s_id)[target_idx][0].cpu()
                        self.classifier_output = model(input)
                        return exp, None, None

                    def set_iteration_index(self, index):
                        self.model_index = index

                grad_cam = RISEWrapper(self.config.device)
            else:
                print("Unknown XAI method")
        else:
            print("No XAI method for this model implemented yet")

        pd_exp_path = join(self.results_path, 'point_dropping')
        if not os.path.exists(pd_exp_path):
            os.makedirs(pd_exp_path, exist_ok=True)

        start = time.time()
        self.reset_seeds()
        PointDropExperiment(
            module=self,
            results_dir=pd_exp_path,
            classifier=model,
            grad_cam=grad_cam,
            testdataloader=loader,
            test_dataset=dataset,
            num_drops=(num_drops or self.config.dataset.n_points),
            steps=steps,
            num_iterations=len(dataset.models),
            update_cam=update_cam,
            show_visualization=False,
            file_prefix=name,
            title_prefix=name,
            create_png=True,
            random_drop=True,
            high_drop=True,
            low_drop=True,
            print_progress=True,
            device=self.config.device,
            # use_cam_points=True,
            is_mesh=is_mesh,
            remove_edges=remove_edges,
            drop_in_batches=True,
        ).run_experiment()
        print("done in ", time.time() - start)

    def get_data(self, data):
        if type(data) == list:
            # typical for pytorchs dataloader to provide a tuple with (data, target)
            input_data, target = data[:2]  # third parameters may be the varialbe transformations, ignore it!
            return input_data.to(self.config.device), target.to(self.config.device)
        else:
            # this applied to pytorch geometric's data loader which returns a data object where y is the target
            data = data.to(self.config.device)
            return data, data.y

    def loss_criterion(self, predicted, target, complete_output=None):
        loss = F.nll_loss(predicted, target.view(-1))
        # criterion = nn.CrossEntropyLoss()
        return loss

    def plot_train_info(self, loss_train_hist=[], loss_test_hist=[], acc_train_hist=[], acc_test_hist=[]):
        plt.clf()
        plt.close()

        # plot loss history
        if len(loss_train_hist) > 0:
            plt.plot(loss_train_hist, label='Average Training Loss')
            plt.plot(loss_test_hist, label='Average Test Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Average Loss over Training Epochs')
            plt.legend()
            plt.savefig(join(self.results_path, "loss_curve.png"))
            plt.clf()
            plt.close()

        # plost accuracy history
        if len(acc_train_hist) > 0:
            plt.plot(acc_train_hist, label='Training Accuracy')
            plt.plot(acc_test_hist, label='Test Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.title('Accuracy over Training Epochs')
            plt.legend()
            plt.savefig(join(self.results_path, "accuracy_curve.png"))
            plt.clf()
            plt.close()

        print("Accuracy and Loss curves have been saved in", self.results_path)


    def get_and_save_test_results(self, model=None, test=False, dir=None):
        if not model:
            print("No model provided")
            if not self.best_model:
                self.get_best_model()
            model = self.best_model

        target_path = self.results_path if dir == None else join(self.results_path, dir)
        if not os.path.isdir(target_path):
            os.makedirs(target_path, exist_ok=True)
        loader = self.test_loader if test else self.val_loader
        dataset = self.test_dataset if test else self.val_dataset
        classes = dataset.classes

        # get and save test accuracy
        _, _, accuracy, y_true, y_pred, loss = self.test(model, loader)
        # y_true, y_pred, accuracy = get_sorted_test_predictions(model, self.test_loader, custom_get_data=self.get_data)

        f1 = sklearn.metrics.f1_score(y_true, y_pred, average='weighted')

        with open(join(target_path, 'accuracy.txt'), 'w') as text_file:
            text_file.write(f"Accuracy: {accuracy}\n")
            text_file.write(f"F1-Score: {f1}")

        # create and save confusion matrix
        save_confusion_matrix(y_true, y_pred, classes, target_path, save_conf_in_wandb=(not test))

        # save a list of all test data with predicted labels and if they are right and wrong (sort it)
        with open(join(target_path, 'predictions.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(['Filename', 'True', 'Predicted'])
            for i in range(len(y_true)):
                writer.writerow([dataset.models[i]['filename'], dataset.models[i]['datapath'], classes[y_true[i]], classes[y_pred[i]]])

        plt.clf()
        plt.close()

        return accuracy, y_true, y_pred