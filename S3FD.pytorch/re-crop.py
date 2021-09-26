#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

from data.config import cfg
from s3fd import build_s3fd
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr

parser = argparse.ArgumentParser(description='s3fd evaluatuon fddb')
parser.add_argument('--model', type=str,
                    default='sfd_face.pth', help='trained model')
                    # default='weights/s3fd.pth', help='trained model')
parser.add_argument('--thresh', default=0.1, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()


use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


# FDDB_IMG_DIR = os.path.join(cfg.FACE.FDDB_DIR, 'images')
# FDDB_FOLD_DIR = os.path.join(cfg.FACE.FDDB_DIR, 'FDDB-folds')
# FDDB_RESULT_DIR = os.path.join(cfg.FACE.FDDB_DIR, 's3fd')
# FDDB_RESULT_IMG_DIR = os.path.join(FDDB_RESULT_DIR, 'images')

# if not os.path.exists(FDDB_RESULT_IMG_DIR):
#     os.makedirs(FDDB_RESULT_IMG_DIR)


def detect_face(net, img, thresh):
    height, width, _ = img.shape
    x = to_chw_bgr(img)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    y = net(x)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    bboxes = []
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            box = []
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(np.int)
            j += 1
            box += [pt[0], pt[1], pt[2] - pt[0], pt[3] - pt[1], score]
            bboxes += [box]

    return bboxes


if __name__ == '__main__':
    args
    net = build_s3fd('test', cfg.NUM_CLASSES)
    net.load_state_dict(torch.load(args.model))
    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    #transform = S3FDBasicTransform(cfg.INPUT_SIZE, cfg.MEANS)

#     img = Image.open(img_file)
    img = Image.open("examples_face-align/test.bmp")
    
    if img.mode == 'L':
        img = img.convert('RGB')
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bboxes = detect_face(net, img, args.thresh)
    

    x1, y1, w, h, score = bboxes[0]
    x1 = x1 - 0.1*w
    # y1 = y1 - 0.1*h
    w = w*1.2
    h = h*1.05
    x1, y1, x2, y2 = int(x1), int(y1), int(x1 + w), int(y1 + h)
    if(x1 < 0):
        x1 = 0
    if(y1 < 0):
        y1 = 0
    '''
    640*512 for IR image
    '''
    if(x2 > 640):
        x2 = 640
    if(y2 > 512):
        y2 = 512
    '''
    1920*1080 for IR image
    '''
#     if(x2 > 1920):
#         x2 = 640
#     if(y2 > 1080):
#         y2 = 512

    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print(x1, x2, y1, y2, score)
    face = img[y1:y2, x1:x2]

    # face = cv2.resize(face, (128,128), interpolation = cv2.INTER_CUBIC)
    face = cv2.resize(face, (256,256), interpolation = cv2.INTER_CUBIC)
    
    cv2.imwrite("examples_face-align/test_re-crop.bmp", face)
