import cv2
import os

pathNames = ['000001','500001']
originalDir = 'original'

saveDir = 'processed'


for pathName in pathNames:
    dir = os.path.join(saveDir,pathName)
    if not os.path.exists(dir):
        os.makedirs(dir)

for pathName in  pathNames:
    pathRoot = os.path.join(originalDir,pathName) 
    paths = os.listdir( pathRoot)
    for index, fpath in enumerate(paths):
        ppath = os.path.join(pathRoot,fpath)
        img = cv2.imread(ppath)
        img = cv2.resize(img,(96,96))
        # print(os.path.join(saveDir,pathName,index+'.jpg'))
        cv2.imwrite( os.path.join(saveDir,pathName,str(index)+'.jpg') ,img)



