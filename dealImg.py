import cv2

bgrImage = cv2.imread('2.jpg')
bgrImage = cv2.resize(bgrImage,(96,96))
cv2.imwrite('2_96x96.jpg',bgrImage)