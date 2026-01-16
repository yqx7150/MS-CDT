import sys
import functools
import matplotlib.pyplot as plt
import torch
import numpy as np                     
import os
import cv2
import logging
from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio as psnr,structural_similarity as ssim,mean_squared_error as mse

def normalize(image):
    """
    Normalize a given image array to the range [0, 1].
    
    Parameters:
        image (np.ndarray): Input image array.
    
    Returns:
        np.ndarray: Normalized image array scaled to [0, 1].
    """
    image_min = image.min()
    image_max = image.max()
    normalized_image = (image - image_min) / (image_max - image_min)
    return normalized_image.astype(np.float64)

def calculate_psnr(img1, img2, max_pixel):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        assert 0

    else:
        psnr_val = 10 * np.log10((max_pixel ** 2) / np.sqrt(mse))
        

        return psnr_val

def calculate_ssim(img1, img2,data_range):
    ssim_val, _ = ssim(img1, img2, full=True,data_range=1)
    return ssim_val

def calculate_nrmse(img1, img2, mean_pixel):
    mse = np.mean((img1 - img2) ** 2)
    rmse = np.sqrt(mse)
    nrmse_val = rmse / mean_pixel
    return nrmse_val


def create_mask(image, value):
    mask = (image > value).astype(np.uint8)
    return mask


#img1:ori;img2:rec
def indictae(img1, img2):   
    if len(img1.shape) == 2:
    
        batch = img1.shape[0]   
        psnr0 = np.zeros(batch)
        ssim2 = np.zeros(batch)
        ssim0 = np.zeros(batch)
        mse0 = np.zeros(batch)

        max_pixel = np.mean(img2)
        
        mean_pixel = np.mean(img1)
        for i in range(batch):
        
            t1 = img1[i,...]
        
            t2 = img2[i,...]

            t3 = img1[i,...]
            t4 = img2[i,...]   

            psnr0[i,...] = psnr(t3,t4,data_range=1)

            ssim0[i,...] = ssim(t1,t2)
            ssim2[i,...] = ssim(t3,t4)

            mse0[i,...] = calculate_nrmse(t1,t2,mean_pixel)
        return psnr0,ssim0,mse0,ssim2
        
    else:
        assert 0




rec_path = '/home/b109/XFH/MSCDT-master/MSCDT-LBP/results/result/LBP_120_1waniter/FDG/res_img_1'
ori_path = '/home/b109/XFH/MSCDT-master/MSCDT_1/dataset/test/FDG'
result_psnr = []
result_ssim = []
result_ssim1 = []
result_nrmse = []

logging.basicConfig(filename='/home/b109/XFH/MSCDT-master/MSCDT-LBP/results/result/LBP_120_1waniter/FDG/FDG_psnr.log',filemode='w',level=logging.DEBUG, format='%(message)s')

for patient_num in sorted(os.listdir(ori_path)):

    ori_file_path = os.path.join(ori_path, patient_num)
    ori_mat_data = loadmat(ori_file_path)
    ori_img = ori_mat_data['data']

    ori_img = ori_img.reshape(( 256, 256))

    rec_file_path = os.path.join(rec_path, patient_num)
    rec_mat_data = loadmat(rec_file_path)
    rec_img = rec_mat_data['data']
    
    rec_img = rec_img.reshape((256, 256))
    metric_data = dict()
    metric_data['img1'] = normalize(ori_img)
    metric_data['img2'] = normalize(rec_img)


    print(metric_data['img1'].mean(),metric_data['img2'].mean())
    psnr0, ssim0, mse0,ssim2 = indictae(metric_data['img1'],metric_data['img2'])

    psnr1 = psnr0.mean()
    ssim1 = ssim0.mean()
    mse1 = mse0.mean()
    ssim3 = ssim2.mean()

    result_nrmse.append(mse1)
    result_ssim.append(ssim1)
    result_psnr.append(psnr1)
    result_ssim1.append(ssim3)

    
    output = 'Step:{}    PSNR:{}    SSIM:{}    NRMSE:{}'.format(patient_num, np.round(psnr1,3), np.round(ssim1,4), np.round(mse1,5))
    print(output) 
    logging.info(output)


p1 = sum(result_psnr)
l1 = len(result_psnr)
p2 = sum(result_ssim)
l2 = len(result_ssim)
p3 = sum(result_nrmse)
l3 = len(result_nrmse)
p4 = sum(result_ssim1)
l4 = len(result_ssim1)

# Calculate final average values
avg_psnr = p1 / l1 if l1 > 0 else 0
avg_ssim = p2 / l2 if l2 > 0 else 0
avg_nrmse = p3 / l3 if l3 > 0 else 0
avg_ssim1 = p4 / l4 if l4 > 0 else 0
# Print and log the final averages
final_output = 'Final Averages -> PSNR: {:.3f}, SSIM: {:.4f}, NRMSE: {:.5f}'.format(avg_psnr, avg_ssim, avg_nrmse)
print(final_output)
logging.info(final_output)
        
   

