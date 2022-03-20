import os
from src.data.eh_pointcloud_dataset import EHDataset
from src.data.eh_geometric_pointcloud_dataset import EHDatasetGeometricProxy
from src.data.eh_geometric_mesh_dataset import EHMeshDatasetGeometric
from src.data.eh_geometric_mesh_edge_unit_dataset import EHMeshDatasetGeometricEdgeUnit
from src.data.eh_geometric_mesh_face_unit_dataset import EHMeshDatasetGeometricFaceUnit
from src.data.eh_geometric_mesh_face_unit_edge_attr_dataset import EHMeshDatasetGeometricFaceUnitEdgeAttr
from src.data.eh_pointcloud_face_unit_dataset import EHPointcloudFaceUnitDataset
from src.data.eh_pointcloud_edge_unit_dataset import EHPointcloudEdgeUnitDataset
from types import SimpleNamespace

def get_datasets(dataset_root,
                 ds_config,
                 test_dataset_override={},
                 cross_validation_set=False):
    assert ds_config.dataset in os.listdir(dataset_root)

    root = os.path.join(dataset_root, ds_config.dataset)

    if ds_config.type == 'mesh':
        DatasetClass = EHMeshDatasetGeometric
    elif ds_config.type == 'mesh_edge_unit':
        DatasetClass = EHMeshDatasetGeometricEdgeUnit
    elif ds_config.type == 'mesh_face_unit':
        DatasetClass = EHMeshDatasetGeometricFaceUnit
    elif ds_config.type == 'mesh_face_unit_edge_attributes':
        DatasetClass = EHMeshDatasetGeometricFaceUnitEdgeAttr
    elif ds_config.type == 'pointcloud_geoemetric':
        DatasetClass = EHDatasetGeometricProxy
    elif ds_config.type == 'pointcloud_face_unit':
        DatasetClass = EHPointcloudFaceUnitDataset
    elif ds_config.type == 'pointcloud_edge_unit':
        DatasetClass = EHPointcloudEdgeUnitDataset
    else:  # pytorch point cloud dataset
        DatasetClass = EHDataset

    dataset = DatasetClass(root,
                           ds_config,
                           split='train',
                           cross_validation_set=cross_validation_set)
    val_dataset = DatasetClass(root,
                                SimpleNamespace(**{**vars(ds_config), **test_dataset_override}),
                                split='val',
                                cross_validation_set=cross_validation_set)
    test_dataset = DatasetClass(root,
                                SimpleNamespace(**{**vars(ds_config), **test_dataset_override}),
                                split='test',
                                cross_validation_set=cross_validation_set)

    return dataset, val_dataset, test_dataset

def get_dir_dataset(samples_dir, ds_config):
    if ds_config.type == 'mesh':
        DatasetClass = EHMeshDatasetGeometric
    elif ds_config.type == 'mesh_edge_unit':
        DatasetClass = EHMeshDatasetGeometricEdgeUnit
    elif ds_config.type == 'mesh_face_unit':
        DatasetClass = EHMeshDatasetGeometricFaceUnit
    elif ds_config.type == 'mesh_face_unit_edge_attributes':
        DatasetClass = EHMeshDatasetGeometricFaceUnitEdgeAttr
    elif ds_config.type == 'pointcloud_geoemetric':
        DatasetClass = EHDatasetGeometricProxy
    elif ds_config.type == 'pointcloud_face_unit':
        DatasetClass = EHPointcloudFaceUnitDataset
    elif ds_config.type == 'pointcloud_edge_unit':
        DatasetClass = EHPointcloudEdgeUnitDataset
    else:  # pytorch point cloud dataset
        DatasetClass = EHDataset

    dataset = DatasetClass(samples_dir,
                           ds_config,
                           load_dir=True)

    return dataset


def get_all_sample_dataset(dataset_root,
                           ds_config):
    assert ds_config.dataset in os.listdir(dataset_root)

    root = os.path.join(dataset_root, ds_config.dataset)

    if ds_config.type == 'mesh':
        DatasetClass = EHMeshDatasetGeometric
    elif ds_config.type == 'mesh_edge_unit':
        DatasetClass = EHMeshDatasetGeometricEdgeUnit
    elif ds_config.type == 'mesh_face_unit':
        DatasetClass = EHMeshDatasetGeometricFaceUnit
    elif ds_config.type == 'mesh_face_unit_edge_attributes':
        DatasetClass = EHMeshDatasetGeometricFaceUnitEdgeAttr
    elif ds_config.type == 'pointcloud_geoemetric':
        DatasetClass = EHDatasetGeometricProxy
    elif ds_config.type == 'pointcloud_face_unit':
        DatasetClass = EHPointcloudFaceUnitDataset
    elif ds_config.type == 'pointcloud_edge_unit':
        DatasetClass = EHPointcloudEdgeUnitDataset
    else:  # pytorch point cloud dataset
        DatasetClass = EHDataset

    dataset = DatasetClass(root,
                           ds_config,
                           split='all')

    return dataset