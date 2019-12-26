from __future__ import print_function
import cv2
import time
import numpy as np
from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils
import dlib
import math

 
#url='http://192.168.43.124:8080/shot.jpg'



def detect(image,hog):
	orig = image.copy()
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(8, 8))
		#padding=(8, 8), scale=1.05)
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# show some information on the number of bounding boxes

	return pick	


def implem_grid(im_src,x_grid,y_grid):
	for l in range(0,len(x_grid)):
		im_src[x_grid[l],y_grid[l],:] = [255,0,0]
	return im_src	


n = 0
avg=0
na = 2
ma = 2

img_height, img_width = 50*na,50*ma
#grid=np.zeros((img_height,img_width,3))
grid=np.zeros((1000/na,1000/ma,3))
fy = (1000/na)/img_height
fx = (1000/ma)/img_width
grid[:,:,:]=255
    
for i in range(0,ma):
    x1=i*(50*fx)
    cv2.line(grid,(x1,0),(x1,(img_height - 1)*fx + fx-1),(255,0,0),2)

for i in range(0,na):
    y1=i*(50*fy)
    cv2.line(grid,(0,y1),((img_width - 1)*fy + fy-1,y1),(255,0,0),2)
y=0

cap = cv2.VideoCapture("sit_3.mp4")


pts_dst = np.array([[208, 263],[471, 204],[354, 475],[677,355]])
pts_src = np.array([[150, 50],[400, 50],[150, 300],[400,300]])
#pts_src = np.array([[100, 50],[350, 50],[100, 300],[350,300]])

while(cap.isOpened()):

	#imgResp=urllib.request.urlopen(url)
	#imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
	#im_src=cv2.imdecode(imgNp,-1)
	#ret = True
	ret,im_src = cap.read()
	if ret == True :
    
		im_src = cv2.resize(im_src, (800, 500))
		first = cv2.resize(im_src, (800, 500))

		
		start = time.time()
		rgb = cv2.cvtColor(im_src, cv2.COLOR_BGR2RGB)
		if y==0:
			hog = cv2.HOGDescriptor()
			x_grid = []
			y_grid = []
			hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
			h, status = cv2.findHomography(pts_src, pts_dst) 
			

			hinv = np.linalg.inv(h)

			
			im_grid = cv2.warpPerspective(grid, h, (first.shape[1],first.shape[0]))
			
			for i in range(0,first.shape[0]):
				for j in range(0,first.shape[1]):
					if im_grid[i,j,0] == 255 and im_grid[i,j,1] == 0 and im_grid[i,j,2] == 0:
						x_grid.append(i)
						y_grid.append(j)
			y=1
		
		
		bbox = detect(im_src,hog)
		positions_x = []
		positions_y = []
		trackers = []
		lbl = 1
		startX = 0
		endX = 0
		startY = 0
		endY = 0

		for i in range(len(bbox)):
			
			startX = bbox[i,0]
			startY = bbox[i,1]
			endX = bbox[i,2]
			endY = bbox[i,3]
			rect = dlib.rectangle(startX, startY, endX, endY)
			
			cv2.rectangle(im_src, (startX, startY), (endX, endY),(0, 255, 0), 2)
			point = np.ones([3,1])
			point[0,0] = int(startX/2 + endX/2)
			point[1,0] = int(startY/10 + endY*9/10)
			point_2nd = np.dot(hinv,point)
		
			xarr = int(point_2nd[1])
			yarr = int(point_2nd[0])
			x1 = (int(xarr/50) )* 50
			x2 = x1 + 50
			y1 = (int(yarr/50) )*50
			y2 = y1 + 50
		
			grid[x1:x2,y1:y2,:] = [255,0,0]
			
		im_src = implem_grid(im_src,x_grid,y_grid)
		cv2.imshow('xx',grid)
		cv2.imshow('x',im_src)
		done = time.time()
		elapsed = done - start
		avg += elapsed
		n+=1
    #print(elapsed) 
    # Press Q on keyboard to  exit
		if cv2.waitKey(25) & 0xFF == ord('q'):
			print(n/avg)
			print("fps")
			break
 
  # Break the loop
	else:
		print(n/avg)
		print("fps")		
		break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()	
 
