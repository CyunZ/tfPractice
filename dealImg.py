import cv2
import os

pathNames = ['000001','050001','005001','000501','000051']
originalDir = 'original'

# saveDir = 'processed'

trainImgDir = 'trainImg'
testImgDir = 'testImg'


# for pathName in pathNames:
#     dir = os.path.join(saveDir,pathName)
#     if not os.path.exists(dir):
#         os.makedirs(dir)

# for pathName in  pathNames:
#     pathRoot = os.path.join(originalDir,pathName) 
#     paths = os.listdir( pathRoot)
#     for index, fpath in enumerate(paths):
#         ppath = os.path.join(pathRoot,fpath)
#         img = cv2.imread(ppath)
#         img = cv2.resize(img,(96,96))
#         # print(os.path.join(saveDir,pathName,index+'.jpg'))
#         cv2.imwrite( os.path.join(saveDir,pathName,str(index)+'.jpg') ,img)


for pathName in pathNames:
    dir = os.path.join(trainImgDir,pathName)
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = os.path.join(testImgDir,pathName)
    if not os.path.exists(dir):
        os.makedirs(dir)

#每类分配几张图做测试
testCount = 5

for pathName in  pathNames:
    pathRoot = os.path.join(originalDir,pathName) 
    paths = os.listdir( pathRoot)
    count = len(paths)
    trainCount = count - testCount
    for index, fpath in enumerate(paths):
        ppath = os.path.join(pathRoot,fpath)
        print('处理:'+ ppath)
        img = cv2.imread(ppath)
        img = cv2.resize(img,(96,96))
        if index < trainCount:        
            cv2.imwrite( os.path.join(trainImgDir,pathName,str(index)+'.jpg') ,img )
        else :            
            cv2.imwrite( os.path.join(testImgDir,pathName,str(index)+'.jpg') ,img )

print('处理完成')