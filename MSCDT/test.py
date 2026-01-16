# flake8: noqa
import os.path as osp
import sys
import os
from basicsr.test import test_pipeline
sys.path.insert(0,'/home/b109/XFH/MSCDT-master/MSCDT')
print(sys.path)
import MSCDT.archs
import MSCDT.data
import MSCDT.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
