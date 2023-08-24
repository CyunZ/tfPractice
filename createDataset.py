import numpy as np
import os
import cv2

imgs = []
labels = []

paths = os.listdir('000001')
for fpath in paths:
    ppath = os.path.join('000001',fpath)
    img = cv2.imread(ppath)
    imgs.append(img)
    labels.append([0,0,0,0,0,1])

imgs = np.array(imgs)
labels = np.array(labels)
np.savez('GDataSet.npz',imgs=imgs,labels=labels)
    
