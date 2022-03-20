# Discovering highly contributing geometric features of CAD models for the verification of technical requirements

This repository contains the related code for my master thesis and contains all relevant experiments to reproduce the results.
Exceptions of these are all experiments that are related to intellectual property (IP)
of Endress+Hauser (i.e. EH PRO production dataset). But the trained models and accuracies
of models and evaluation are provided.

Further, folders named `_misc_experiments` contain experiments that have been conducted (260+ experiments) but are
not relevant for the final results. They have been added for completeness, but have not been tested with the final code
and do not contain all code required to run them.

In the folder `<root>/src/experiments` are all final models and relevant experiments. These are structured in the scheme
`{model_type}/{dataset_problem}/{model}`, e.g. `/src/experiments/PointNet2/toy/pointnet_toy_normals` contains the final model
of PointNet++ trained on toy dataset with point nomal vectors provided. In mentioned experiemnt folder is the
`train_test.py` script which trains and evaluates the model and
outputs the results into the `{path_to_experiment}/results` folder.
Additionally in the same folder are more experiments that work with the trained model. These are for example
`point_droppling_exp_gradcam.py` which creates the point dropping experiment results from chapter 5.
Additionally, `explain_rise.py` uses the trained model to explain some samples located in `/src/explanation_methods/samples/toy`.

Please note:
- The original project folder had a size of around 700 GB. To reduce the size, cached data (point clouds, meshes) have been deleted.
If you re-run some experiments, the results may slightly change based on the non-deterministic sampling and augmentation.
Further, intermediate model results have been removed.
- Sampled points clouds and mesh augmentations are cached and create a few 10s of GBs of storage place. See `/src/datasets/<dataset_name>/data{*}/cache`
for the cached data to delete it afterwards.
- EdgeASAP and EdgeGlobal require a lot of memory, for classification results you may be have reduce the batch size in the model config

## Setup

The following setup has been tested on our server with a newly created conda environment.

System Specs:
- Ubuntu 18.04
- Nvidia Driver Version: 440.100
- CUDA Version: 10.2
- 2 x Nvidia Quadro RTX 8000 with 48 GB

The following python and package versions have been used:
- python=3.75
- pip=20.2.2
- tqdm=4.48.2
- open3d=0.12.0
- torch=1.6.0
- torchvision=0.7.0
- torch-geometric=1.6.1
- torch-cluster=1.5.7
- torch-scatter=2.0.5
- torch-sparse=0.6.7
- torch-spline-conv=1.2.0
- numpy=1.19.1
- matplotlib=3.3.1
- opencv-python=4.4.0.46
- scikit-learn=0.23.2
- scikit-image=0.17.2

Optional (not required to run most relevant experiments)
- ray=1.0.1.post1
- jupyter

#### Setup Environment

1) setup an environemnt with stated python and pip version:<br/>
```conda create -n submission python=3.7.5 pip=20.2.4```

2) Install required package by open the folder and execute the installation script: <br/>
Either call
```pip install -r requirements.txt``` or use <br />
```pip install tqdm==4.48.2 open3d==0.12.0 numpy==1.19.1 matplotlib==3.3.1 opencv-python==4.4.0.46 scikit-learn==0.23.2 scikit-image==0.17.2 ray==1.0.1.post1```

3) Install pytorch. Follow these instructions https://pytorch.org/get-started/previous-versions/#v160 to install pytorch 1.6 
with your CUDA version. If you use CUDA 10.2 and conda, you can simple use:<br/>
```conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch```<br /><br />
We prefer conda for installation of pytorch and torchvision, because the installation with pip led to the following error:
```libcusparse.so.10: cannot open shared object file: No such file or directory```
If you encounter the same error, please use conda.

4) Install pytorch geometric https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html. In my case:<br/>
```pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html``` <br/>
```pip install torch-sparse==0.6.7 -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html``` <br/>
```pip install torch-cluster==1.5.7 -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html``` <br/>
```pip install torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html``` <br/>
```pip install torch-geometric==1.6.1```

## Relevant Experiments
In the folder ```/src/experiments``` are the most relevant trained models, their configurations and corresponding experiments.
The code for the models is placed in ```/src/models```. When an experiment is executed, it loads the model from the corresponding
```model.py``` definition in the ```/src/models``` folder. Every model inherits
the ```BaseModel``` (in ```src/base_model.py```) which in turn provides methods for loading the datasets, training, testing, evaluation, tuning, explanation and
other experiments.

Classification results are placed in `<experiment_dir>/results/test_latest_model/accuracy.txt`.

<br/>
In the following the folder structure is explained in more detail:
<br />
<br />

**/datasets** <br/>
Contains the datasets, each dataset has it's own folder, e.g. for the EH MMM dataset look into ```/datasets/eh_mmm```.
Within the folder, you will find different `.csv` files which contain the filenames, labels etc. In the `data` folder 
are the actual files which are referenced in the csv files.
The folder `data-{number}` contain reduced meshes with the corresponding number of faces (augmentations).

**/src/base_model.py** <br />
The base model is used in all experiments. It's purpose is to load the datasets, setup the experiment results folder, 
run the training, cross-validation, tuning, save and load existing models, save benchmark results, run explanation methods, etc. 

**/src/experiments** <br />
Contains all experiments. These provide a config, load a model and then train it and optionally call the XAI methods.

**/src/models** <br />
Contains the actual models. The folder name is used as the ID and is referenced in the `model` field of the config
of an experiment.
To connect a model, `model.py` has to be implemented within the folder which has to inherit the BaseModel.
Check `src/models/PointNet2/model.py` for an example. 

**/src/explanation_methods** <br />
The explanation methods are placed in here.
In the folder `./samples` are samples that are used for explanation in the experiments.

**/src/data** <br />
Contains different dataset loaders, data exploration and data conversion scripts, data labeling tool etc.