## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
#import deblur_utils as utils
from basicsr.models import build_model
from natsort import natsorted
from glob import glob
#from kdsrgan.archs.TA_arch import BlindSR_TA
from skimage import img_as_ubyte
#from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')

parser.add_argument('--input_dir', default='/home/b109/XFH/MSCDT-master/MSCDT_1/dataset/test/fuse', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./experiments/train_MSCDTS2/models/net_g_17000.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='HIDE', type=str, help='Test Dataset')

args = parser.parse_args()
ckpt_allname = args.ckpt.split("/")[-1]
####### Load yaml #######
yaml_file = './options/test_MSCDTS2.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

#model_restoration = BlindSR_TA(**x['network_g'])
#model_restoration = build_model(x)

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 8
dataset = args.dataset
result_dir  = os.path.join(args.result_dir, dataset)
os.makedirs(result_dir, exist_ok=True)

gt_f18_path = '/home/b109/XFH/MSCDT-master/MSCDT-demotionblur/dataset/test/F18'
gt_g68_path = '/home/b109/XFH/MSCDT-master/MSCDT-demotionblur/dataset/test/G68'
lg_path = '/home/b109/XFH/MSCDT-master/MSCDT-demotionblur/dataset/test/fuse'
with torch.no_grad():
    #for inp_file_,gt_file_ in tqdm(zip(inp_files,gt_files)):
    for num in sorted(os.listdir(lg_path)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        
        gt_f18_img_path = gt_f18_path + '/' + full + '/' + num
        gt_g68_img_path = gt_g68_path + '/' + full + '/' + num
        lq_img_path = lg_path+ '/' + num
        lq_data = loadmat(lq_img_path)
        data = lq_data['data']
        lq_img = np.zeros((3, 256, 256), dtype=np.float32)
        lq_img[0, :, :] = data
        lq_img[1, :, :] = data

        mat_data_F18 = loadmat(gt_f18_img_path)
        data_F18 = mat_data_F18['data_F18']
        mat_data_G68 = loadmat(gt_g68_img_path)
        data_G68 = mat_data_G68['data_G68']
        gt_img = np.zeros((3, 256, 256), dtype=np.float32)

        gt_img[0, :, :] = np.array(data_F18)
        gt_img[1, :, :] = np.array(data_G68)

        # Unpad images to original dimensions
        pre_F18 = restored[0, :, :].cpu().numpy()
        pre_G68 = restored[1, :, :].cpu().numpy()

        assert 0
        
        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(inp_file_)[-1])[0]+'.png')), img_as_ubyte(restored))

        
