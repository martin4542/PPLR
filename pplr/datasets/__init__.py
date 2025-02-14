from __future__ import absolute_import
import warnings

from .market1501 import Market1501
from .msmt17 import MSMT17
from .veri import VeRi
from .ai_hub import AI_HUB
from .ellexi_CCTV import Ellexi_CCTV


__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'veri': VeRi,
    'aihub': AI_HUB,
    'ellexi': Ellexi_CCTV
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. 
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


def get_dataset(name, root, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, *args, **kwargs)
