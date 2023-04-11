from __future__ import print_function, absolute_import
import glob
import numpy as np
import os.path as osp

from ..utils.data import BaseImageDataset

class Combined_Dataset(BaseImageDataset):
    """
    Combine datasets

    gallery ids == query ids
    trains id는 gallery id와 꼭 같을 필요없음
    """

    def __init__(self, datasets, verbose=True, **kwargs):
        super(Combined_Dataset, self).__init__()
        self.total_train_pids = 0
        self.total_query_pids = 0
        self.total_gallery_pids = 0
        self.total_camids = 0

        self.train = []
        self.query = []
        self.gallery = []

        self.combine_datasets(datasets)

        if verbose:
            print("=> Combined dataset")
            self.print_dataset_statistics(self.train, self.query, self.gallery)
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def combine_datasets(self, datasets):
        for dataset in datasets:
            # camids are duplicated in train & gallery & query thus is needs process first
            # combine all camids and make camid dict to re-label all camids
            camid_container = []
            camid_container.extend([i[2] for i in dataset.train])
            camid_container.extend([i[2] for i in dataset.gallery])
            camid_container.extend([i[2] for i in dataset.query])
            camid_container = list(set(camid_container))
            camid2idx = {camid: label for label, camid in enumerate(camid_container)}

            # make pid dict and re-label all pids and then combine dataset
            
            # make train pid dict
            pid_container = list(set([i[1] for i in dataset.train]))
            pid2idx = {pid: label for label, pid in enumerate(pid_container)}

            # re-label train pid, camid
            for fpath, pid, camid in dataset.train:
                self.train.append((fpath, pid2idx[pid] + self.total_train_pids, camid2idx[camid] + self.total_camids))

            # make query & gallery pid dict (query and gallery must have same pid dict)
            pid_container = list(set([i[1] for i in dataset.gallery]))
            pid2idx = {pid: label for label, pid in enumerate(pid_container)}

            # re-label query & gallery pid, camid
            for fpath, pid, camid in dataset.query:
                self.query.append((
                    fpath,
                    pid2idx[pid] + max(self.total_gallery_pids, self.total_query_pids),
                    camid2idx[camid] + self.total_camids
                ))
            for fpath, pid, camid in dataset.gallery:
                self.gallery.append((
                    fpath,

                    pid2idx[pid] + max(self.total_gallery_pids, self.total_query_pids),
                    camid2idx[camid] + self.total_camids
                ))
                
            self.total_train_pids += dataset.num_train_pids
            self.total_query_pids += max(dataset.num_query_pids, dataset.num_gallery_pids)
            self.total_gallery_pids += max(dataset.num_query_pids, dataset.num_gallery_pids)
            self.total_camids += len(camid2idx)