# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 10:26:43 2024

@author: 28622
"""

import os
from src.PINN import PhysicsInformedNN
from utils.utils import model_data_collect, save_mat
from train import model_train
from predict import model_predict
from plot import result_plot, contourf_plot, error_plot
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # 数据准备
    ref_image_path = "./speckle_image/real/circle/nuclear_ring_ref.bmp"
    def_image_path = "./speckle_image/real/circle/nuclear_ring_def.bmp"
    roi_path = "./speckle_image/real/circle/roi.bmp"
    
    RG, DG, ROI, IX, IY, XY_roi = model_data_collect(ref_image_path, def_image_path, roi_path)
    layers = [2, 50, 50, 50, 2]
    
    model = PhysicsInformedNN(RG, DG, ROI, layers, IX, IY, XY_roi)

    # 训练阶段
    pretrain_epoch = [1000, 0]
    train_epoch = [1000, 0]
    gray1 = 3
    gary2 = 1
    print("train start")
    model_train(model, IX, IY, pretrain_epoch, train_epoch, gray1, gary2, new_lr=0.0001)
    print("train over")

    # 输出求解结果
    u, v = model_predict(model)
    u_roi = u[XY_roi[:, 0], XY_roi[:, 1]]; v_roi = v[XY_roi[:, 0], XY_roi[:, 1]];
    umin = np.min(u_roi); umax = np.max(u_roi)
    vmin = np.min(v_roi); vmax = np.max(v_roi)
    
    # 画出位移云图
    result_plot(
        u, v, 
        u_min=umin, u_max=umax, v_min=vmin, v_max=vmax, 
        string='', layout = [1,2], WH=[4,4]
        )
    
    # 画出位移等值图
    contourf_plot(
        u, v, N=10, 
        u_min=umin, u_max=umax, v_min=vmin, v_max=vmax, 
        string='', layout = [1,2], WH=[4,4]
        )
    
if __name__ == "__main__":
    main()

