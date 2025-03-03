# -*-coding:utf-8 -*-

import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet

# todo 需要封装为函数
if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    save_dir = "images/predict"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model_drive.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob('../DRIVE-SEG-DATA/Test_Images/*.png')
    # 遍历素有图片
    for i, test_path in enumerate(tests_path):
        # 保存结果地址
        save_res_path = os.path.join(save_dir, os.path.basename(test_path))
        # 读取图片
        img = cv2.imread(test_path)
        origin_shape = img.shape
        # print(origin_shape)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (512, 512))
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # 保存图片
        pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_res_path, pred)
        print("{}: {}的预测结果已经保存在{}".format(i+1, test_path, save_res_path))