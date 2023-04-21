from __future__ import print_function, absolute_import
import os.path as osp
import glob

from ..utils.data import BaseImageDataset

class Ellexi_CCTV(BaseImageDataset):
    dataset_dir = "ellexi_CCTV"

    def __init__(self, root, dates=["2023-03-29", "2023-03-30"], verbose=True, **kwargs):
        super(Ellexi_CCTV, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.gallery_dir = osp.join(self.dataset_dir, "gallery")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.dates = dates

        self._check_before_run()

        self.pid_map = {}

        self._build_id_map()

        train = self._process_dir(self.train_dir)
        gallery = self._process_dir(self.gallery_dir)
        query = self._process_dir(self.query_dir)

        self.train = train
        self.gallery = gallery
        self.query = query

        if verbose:
            print("=> Ellexi CCTV dataset loaded")
            self.print_dataset_statistics(train, query, gallery)
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)

    def _check_before_run(self):
        """Check if files are available"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"{self.dataset_dir} is not available")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"{self.train_dir} is not available")
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(f"{self.gallery_dir} is not available")
        if not osp.exists(self.query_dir):
            raise RuntimeError(f"{self.query_dir} is not available")

    def _build_id_map(self):
        train_imgs = []
        val_imgs = []
        for date in self.dates:
            train_imgs.extend([osp.basename(f) for f in glob.glob(osp.join(self.train_dir, date, "*.png"))])
            val_imgs.extend([osp.basename(f) for f in glob.glob(osp.join(self.gallery_dir, date, "*.png"))])
        
        self.train_pids = list(sorted(set([train_img.split("_")[-1][:-4] for train_img in train_imgs])))
        self.val_pids = list(sorted(set([val_img.split("_")[-1][:-4] for val_img in val_imgs])))

        for idx, pid in enumerate(self.train_pids):
            self.pid_map[pid] = idx
        
        for idx, pid in enumerate(self.val_pids):
            if pid not in self.pid_map.keys():
                self.pid_map[pid] = len(self.pid_map)
    
    def _process_dir(self, dir_path):
        dataset = []
        
        for date in self.dates:
            img_paths = [osp.basename(f) for f in glob.glob(osp.join(dir_path, date, "*.png"))]
            for img_path in img_paths:
                pid = img_path.split("_")[-1][:-4]
                pid = self.pid_map[pid]
                camid = 0
                dataset.append((osp.join(dir_path, date, img_path), pid, camid))

        return dataset