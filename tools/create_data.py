import copy
from pathlib import Path
import pickle

import fire, os

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database
from det3d.datasets.waymo import waymo_common as waymo_ds
from det3d.datasets.kitti import kitti_common as kitti_ds

def nuscenes_data_prep(root_path, version, nsweeps=10, filter_zero=True, virtual=False):
    nu_ds.create_nuscenes_infos(root_path, version=version, nsweeps=nsweeps, filter_zero=filter_zero)
    if version == 'v1.0-trainval':
        create_groundtruth_database(
            "NUSC",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(nsweeps, filter_zero),
            nsweeps=nsweeps,
            virtual=virtual
        )

def waymo_data_prep(root_path, split, nsweeps=1):
    waymo_ds.create_waymo_infos(root_path, split=split, nsweeps=nsweeps)
    if split == 'train': 
        create_groundtruth_database(
            "WAYMO",
            root_path,
            Path(root_path) / "infos_train_{:02d}sweeps_filter_zero_gt.pkl".format(nsweeps),
            used_classes=['VEHICLE', 'CYCLIST', 'PEDESTRIAN'],
            nsweeps=nsweeps
        )
    
def kitti_data_prep(root_path):
    kitti_ds.create_kitti_info_file(root_path)
    kitti_ds.create_reduced_point_cloud(root_path)
    create_groundtruth_database("KITTI", root_path, Path(root_path) / "kitti_infos_train.pkl")

# def nuscenes_data_prep(root_path, version, dataset_name, max_sweeps=10):
#     nu_ds.create_nuscenes_infos(root_path, version=version, max_sweeps=max_sweeps)
#     name = "infos_train.pkl"
#     if version == "v1.0-test":
#         name = "infos_test.pkl"
#     create_groundtruth_database(dataset_name, root_path, Path(root_path) / name)

if __name__ == "__main__":
    fire.Fire()
