import cv2
import os
os.mkdir("dataresizealiem")
for p in os.listdir("/u01/liemhd/dataset/reader/id/new/jpg"):
	img = cv2.imread("/u01/liemhd/dataset/reader/id/new/jpg/"+p)
	height, width = img.shape[:2]
	img = cv2.resize(img,(int(width*24/height),24))
	cv2.imwrite("dataresizealiem/"+p,img)
