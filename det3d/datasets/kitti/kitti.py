import sys
import pickle
import json
import random
import operator
import numpy as np

from functools import reduce
from pathlib import Path
from copy import deepcopy

from det3d.core import box_np_ops
from det3d.datasets.kitti import kitti_common as kitti
# from det3d.datasets.utils.eval import get_coco_eval_result, get_official_eval_result

# try:
#     from nuscenes.nuscenes import NuScenes
#     from nuscenes.eval.detection.config import config_factory
# except:
#     print("nuScenes devkit not found!")

from det3d.datasets.custom import PointCloudDataset
# from det3d.datasets.nuscenes.nusc_common import (
#     general_to_detection,
#     cls_attr_dist,
#     _second_det_to_nusc_box,
#     _lidar_nusc_box_to_global,
#     eval_main
# )
from det3d.datasets.registry import DATASETS

@DATASETS.register_module
class KittiDataset(PointCloudDataset):
    NumPointFeatures = 4

    def __init__(self,
                 info_path,
                 root_path,
                 pipeline=None,
                 class_names=None,
                 test_mode=False,
                 prep_func=None,
                 num_point_features=None):
        assert info_path is not None
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        self._root_path = Path(root_path)
        self._kitti_infos = infos

        print("remain number of infos:", len(self._kitti_infos))
        self._class_names = class_names
        self._prep_func = prep_func

    def __len__(self):
        return len(self._kitti_infos)

    def convert_detection_to_kitti_annos(self, detection):
        class_names = self._class_names
        det_image_idxes = [det["metadata"]["image_idx"] for det in detection]
        gt_image_idxes = [
            info["image"]["image_idx"] for info in self._kitti_infos
        ]
        annos = []
        for i in range(len(detection)):
            det_idx = det_image_idxes[i]
            det = detection[i]
            # info = self._kitti_infos[gt_image_idxes.index(det_idx)]
            info = self._kitti_infos[i]
            calib = info["calib"]
            rect = calib["R0_rect"]
            Trv2c = calib["Tr_velo_to_cam"]
            P2 = calib["P2"]
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            if final_box_preds.shape[0] != 0:
                final_box_preds[:, 2] -= final_box_preds[:, 5] / 2
                box3d_camera = box_np_ops.box_lidar_to_camera(
                    final_box_preds, rect, Trv2c)
                locs = box3d_camera[:, :3]
                dims = box3d_camera[:, 3:6]
                angles = box3d_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = box_np_ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
                box_corners_in_image = box_np_ops.project_to_image(
                    box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = np.min(box_corners_in_image, axis=1)
                maxxy = np.max(box_corners_in_image, axis=1)
                bbox = np.concatenate([minxy, maxxy], axis=1)
            anno = kitti.get_start_result_anno()
            num_example = 0
            box3d_lidar = final_box_preds
            for j in range(box3d_lidar.shape[0]):
                image_shape = info["image"]["image_shape"]
                if bbox[j, 0] > image_shape[1] or bbox[j, 1] > image_shape[0]:
                    continue
                if bbox[j, 2] < 0 or bbox[j, 3] < 0:
                    continue
                bbox[j, 2:] = np.minimum(bbox[j, 2:], image_shape[::-1])
                bbox[j, :2] = np.maximum(bbox[j, :2], [0, 0])
                anno["bbox"].append(bbox[j])
                # convert center format to kitti format
                # box3d_lidar[j, 2] -= box3d_lidar[j, 5] / 2
                anno["alpha"].append(
                    -np.arctan2(-box3d_lidar[j, 1], box3d_lidar[j, 0]) +
                    box3d_camera[j, 6])
                anno["dimensions"].append(box3d_camera[j, 3:6])
                anno["location"].append(box3d_camera[j, :3])
                anno["rotation_y"].append(box3d_camera[j, 6])

                anno["name"].append(class_names[int(label_preds[j])])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(scores[j])

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
            num_example = annos[-1]["name"].shape[0]
            annos[-1]["metadata"] = det["metadata"]
        return annos

    def evaluation(self, detections, output_dir=None, testset=False):
        """
        detection
        When you want to eval your own dataset, you MUST set correct
        the z axis and box z center.
        If you want to eval by my KITTI eval function, you must 
        provide the correct format annotations.
        ground_truth_annotations format:
        {
            bbox: [N, 4], if you fill fake data, MUST HAVE >25 HEIGHT!!!!!!
            alpha: [N], you can use -10 to ignore it.
            occluded: [N], you can use zero.
            truncated: [N], you can use zero.
            name: [N]
            location: [N, 3] center of 3d box.
            dimensions: [N, 3] dim of 3d box.
            rotation_y: [N] angle.
        }
        all fields must be filled, but some fields can fill
        zero.
        """
        if "annos" not in self._kitti_infos[0]:
            return None
        gt_annos = [info["annos"] for info in self._kitti_infos]
        dt_annos = self.convert_detection_to_kitti_annos(detections)
        # firstly convert standard detection to kitti-format dt annos
        z_axis = 1  # KITTI camera format use y as regular "z" axis.
        z_center = 1.0  # KITTI camera box's center is [0.5, 1, 0.5]
        # for regular raw lidar data, z_axis = 2, z_center = 0.5.
        # result_official_dict = get_official_eval_result(
        #     gt_annos,
        #     dt_annos,
        #     self._class_names,
        #     z_axis=z_axis,
        #     z_center=z_center)
        # result_coco = get_coco_eval_result(
        #     gt_annos,
        #     dt_annos,
        #     self._class_names,
        #     z_axis=z_axis,
        #     z_center=z_center)
        # return {
        #     "results": {
        #         "official": result_official_dict["result"],
        #         "coco": result_coco["result"],
        #     },
        #     "detail": {
        #         "eval.kitti": {
        #             "official": result_official_dict["detail"],
        #             "coco": result_coco["detail"]
        #         }
        #     },
        # }
        return None, None

    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict=input_dict)
        example["metadata"] = {}
        if "image_idx" in input_dict["metadata"]:
            example["metadata"] = input_dict["metadata"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example

    def get_sensor_data(self, query):
        read_image = False
        idx = query
        if isinstance(query, dict):
            read_image = "cam" in query
            assert "lidar" in query
            idx = query["lidar"]["idx"]
        info = self._kitti_infos[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "metadata": {
                "image_idx": info["image"]["image_idx"],
                "image_shape": info["image"]["image_shape"],
            },
            "calib": None,
            "cam": {}
        }

        pc_info = info["point_cloud"]
        velo_path = Path(pc_info['velodyne_path'])
        if not velo_path.is_absolute():
            velo_path = Path(self._root_path) / pc_info['velodyne_path']
        velo_reduced_path = velo_path.parent.parent / (
            velo_path.parent.stem + '_reduced') / velo_path.name
        if velo_reduced_path.exists():
            velo_path = velo_reduced_path
        points = np.fromfile(
            str(velo_path), dtype=np.float32,
            count=-1).reshape([-1, self.NumPointFeatures])
        res["lidar"]["points"] = points
        image_info = info["image"]
        image_path = image_info['image_path']
        if read_image:
            image_path = self._root_path / image_path
            with open(str(image_path), 'rb') as f:
                image_str = f.read()
            res["cam"] = {
                "type": "camera",
                "data": image_str,
                "datatype": image_path.suffix[1:],
            }
        calib = info["calib"]
        calib_dict = {
            'rect': calib['R0_rect'],
            'Trv2c': calib['Tr_velo_to_cam'],
            'P2': calib['P2'],
        }
        res["calib"] = calib_dict
        if 'annos' in info:
            annos = info['annos']
            # we need other objects to avoid collision when sample
            annos = kitti.remove_dontcare(annos)
            locs = annos["location"]
            dims = annos["dimensions"]
            rots = annos["rotation_y"]
            gt_names = annos["name"]
            # rots = np.concatenate([np.zeros([locs.shape[0], 2], dtype=np.float32), rots], axis=1)
            gt_boxes = np.concatenate([locs, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)
            calib = info["calib"]
            gt_boxes = box_np_ops.box_camera_to_lidar(
                gt_boxes, calib["R0_rect"], calib["Tr_velo_to_cam"])

            # only center format is allowed. so we need to convert
            # kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]
            box_np_ops.change_box3d_center_(gt_boxes, [0.5, 0.5, 0],
                                            [0.5, 0.5, 0.5])
            res["lidar"]["annotations"] = {
                'boxes': gt_boxes,
                'names': gt_names,
            }
            res["cam"]["annotations"] = {
                'boxes': annos["bbox"],
                'names': gt_names,
            }

        return res
