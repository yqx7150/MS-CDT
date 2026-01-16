# flake8: noqa
import os.path as osp

import sys

sys.path.insert(0, '/home/b109/XFH/MSCDT-master/MSCDT/MSCDT')
sys.path.insert(0, '/home/b109/XFH/MSCDT-master/MSCDT')
from MSCDT.train_pipeline import train_pipeline

import MSCDT.archs
import MSCDT.data
import MSCDT.models
import MSCDT.losses
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
