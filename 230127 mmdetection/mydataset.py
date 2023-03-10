import copy
import os.path as osp

import mmcv
import cv2
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

@DATASETS.register_module()
class KittiTinyDataset(CustomDataset) :
    CLASSES = ('Car', 'Pedestrian', 'Cyclist')

    def load_annotations(self, ann_file):
        cat2label = {k : i for i, k in enumerate(self.CLASSES)}
        print(cat2label)
        image_list = mmcv.list_from_file(self.ann_file)

        data_infos = []
        for image_id in image_list :
            filename = '{0:}/{1:}.jpeg'.format(self.img_prefix, image_id)
            
            # image width height
            image = cv2.imread(filename)
            height, width = image.shape[:2]

            data_info = {'filename' : str(image_id) + '.jpeg',
                         'width': width, 'height':height}
            
            # annotation sub dir prefix
            label_prefix = self.img_prefix.replace('image_2', 'label_2')

            # annotation read file 1 line
            lines = mmcv.list_from_file(osp.join(label_prefix, str(image_id) + '.txt'))

            content = [line.strip().split('') for line in lines]
            bbox_name = [x[0] for x in content]
            # bbox save
            bboxes = [[float(info) for info in x[4:3]] for x in content]

            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []

            for bbox_name, bbox in zip(bbox_name, bboxes):
                if bbox_name in cat2label :
                    gt_labels.append(cat2label[bbox_name])
                    gt_bboxes.append(bbox)

                else:
                    gt_labels_ignore.append(-1)
                    gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes = np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels = np.array(gt_labels, dtype=np.long),
                gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
                labels_ignore = np.array(gt_labels_ignore, dtype=np.long)
            )
            data_info.update(data_anno)
            data_infos.append(data_info)

        return data_infos