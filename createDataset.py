import numpy as np
import os
import cv2

trainImgDir = 'trainImg'
testImgDir = 'testImg'
#文件夹名即label
pathNames = ['000001','050001','005001','000501','000051']
labelMap = {
            '000001':[0,    0,  0,  0,  0,  1],
            '050001':[0,    0.5,0,  0,  0,  1],
            '005001':[0,    0,  0.5,0,  0,  1],
            '000501':[0,    0,  0,  0.5,0,  1],
            '000051':[0,    0,  0,  0,  0.5,1]
            }

train_imgs = []
train_labels = []

test_imgs = []
test_labels = []

#训练用的数据集
for pathName in pathNames:
    pathRoot = os.path.join(trainImgDir,pathName) 
    paths = os.listdir( pathRoot)
    for fpath in paths:
        ppath = os.path.join(pathRoot,fpath)
        img = cv2.imread(ppath)
        train_imgs.append(img)
        train_labels.append(labelMap[pathName])

imgs = np.array(train_imgs)
labels = np.array(train_labels)
np.savez('TrainDataSet.npz',imgs=imgs,labels=labels)
    
#测试用的数据集
for pathName in pathNames:
    pathRoot = os.path.join(testImgDir,pathName) 
    paths = os.listdir(pathRoot)
    for fpath in paths:
        ppath = os.path.join(pathRoot,fpath)
        img = cv2.imread(ppath)
        test_imgs.append(img)
        test_labels.append(labelMap[pathName])

imgs = np.array(test_imgs)
labels = np.array(test_labels)
np.savez('TestDataSet.npz',imgs=imgs,labels=labels)
    
