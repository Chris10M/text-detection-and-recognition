import cv2
import numpy as np 


img = cv2.imread('t1.jpg',0)
#cv2.imshow('test',img)

r = 500.0 / img.shape[1]
dim = (500, int(img.shape[0] * r))

#img2grayscale=  cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
img2grayscale=  cv2.resize(img,None,fx=0.25,fy=0.25,interpolation=cv2.INTER_AREA)
cv2.imshow('testshrink',img2grayscale)
img2grayscale = cv2.GaussianBlur(img2grayscale,(3,3),0)
cv2.imshow('Gaussian',img2grayscale)
edges = cv2.Canny(img2grayscale,50,100)
cv2.imshow('edges',edges)


cv2.waitKey(0)
cv2.destroyAllWindows()