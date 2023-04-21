from __future__ import print_function, absolute_import
import os.path as osp
import glob

from ..utils.data import BaseImageDataset

class AI_HUB(BaseImageDataset):
    """
    AI-Hub dataset

    Dataset statistics:
    ----------------------------------------
    subset   | # ids | # images | # cameras
    ----------------------------------------
    train    |   502 |   149122 |       206
    query    |   500 |    10500 |        46
    gallery  |   500 |   108861 |       190
    ---------------------------------------

    # identities: 503 (+1 for background)
    # images: 149122 (train) + 10500 (query) + 108861 (gallery)
    """
    dataset_dir = 'aihub_reid'

    def __init__(self, root, verbose=True, **kwargs):
        super(AI_HUB, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "train", "images")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "val", "images")

        self._check_before_run()

        self.pid_map = {}
        self.camid_map = {}

        self._build_id_map()

        train = self._process_dir(self.train_dir)
        query = self._process_dir(self.query_dir)
        gallery = self._process_dir(self.gallery_dir)

        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            print("=> AIHub dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if files are available"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"{self.datset_dir} is not available")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"{self.train_dir} is not available")
        if not osp.exists(self.query_dir):
            raise RuntimeError(f"{self.query_dir} is not available")
        if not osp.exists(self.gallery_dir):
            raise RuntimeError(f"{self.gallery_dir} is not available")
    
    def _build_id_map(self):
        pid_idx = 0
        camid_idx = 0

        train_imgs = [osp.basename(f) for f in glob.glob(osp.join(self.train_dir, "*.png"))]
        val_imgs = [osp.basename(f) for f in glob.glob(osp.join(self.gallery_dir, "*.png"))]

        self.train_pids = list(set([train_img.split('_')[1] for train_img in train_imgs]))
        self.train_camids = list(set([train_img.split('_')[3] for train_img in train_imgs]))
        self.val_pids = list(set([val_img.split('_')[1] for val_img in val_imgs]))
        self.val_camids = list(set([val_img.split('_')[3] for val_img in val_imgs]))

        for idx in range(max(len(self.train_pids), len(self.train_camids))):
            id = self.train_pids[idx] if idx < len(self.train_pids) else None
            camid = self.train_camids[idx] if idx < len(self.train_camids) else None

            if id:
                assert id.startswith('H'), id
                self.pid_map[id] = pid_idx
                pid_idx += 1
            if camid:
                assert len(camid) == 6, camid
                self.camid_map[camid] = camid_idx
                camid_idx += 1

        for idx in range(max(len(self.val_pids), len(self.val_camids))):
            id = self.val_pids[idx] if idx < len(self.val_pids) else None
            camid = self.val_camids[idx] if idx < len(self.val_camids) else None
            
            if id and id not in self.pid_map.keys():
                assert id.startswith('H'), id
                self.pid_map[id] = pid_idx
                pid_idx += 1
            if camid and camid not in self.camid_map.keys():
                assert len(camid) == 6, camid
                self.camid_map[camid] = camid_idx
                camid_idx += 1

    def _process_dir(self, dir_path):
        dataset = []
        img_paths = [osp.basename(f) for f in glob.glob(osp.join(dir_path, "*.png"))]

        for img_path in img_paths:
            pid = img_path.split('_')[1]
            camid = img_path.split('_')[3]

            assert pid.startswith('H'), pid
            assert len(camid) == 6, camid

            pid = self.pid_map[pid]
            camid = self.camid_map[camid]         
            
            dataset.append((osp.join(dir_path, img_path), pid, camid))

        return dataset