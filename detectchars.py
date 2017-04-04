#!/usr/bin/python

import time

import sys
import os

import cv2
import numpy as np


def detect_letter_stage_1( img ):
	erc1 = cv2.text.loadClassifierNM1('trained_classifierNM1.xml')
	er1 = cv2.text.createERFilterNM1(erc1)

	erc2 = cv2.text.loadClassifierNM2('trained_classifierNM2.xml')
	er2 = cv2.text.createERFilterNM2(erc2)

	regions = cv2.text.detectRegions(gray,er1,er2)
	rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions] 
	rects.extend(rects);

	no_overlap_rects=cv2.groupRectangles(rects,2,0.2)
	no_overlap_rects=tuple(map(tuple,no_overlap_rects[0]))

	return no_overlap_rects


def canny_with_blur(img,threshold1=100 , threshold2=200):
	blur = cv2.GaussianBlur(img,(3,3),0)
	canny = cv2.Canny(blur,threshold1,threshold2)

	return canny

def resize_img(img,height , width):
	r = float(height) / img.shape[1]
	dim = (width,int(img.shape[0]*r))
	resized = cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
	return resized

def edge_present(img , no_overlap_rects):
	canny_img = canny_with_blur(img,200,300)

	img2 ,contours , hierarchy = cv2.findContours(canny_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	print len(contours)

	cloneimg = np.zeros(img.shape,np.uint8)

	erc1 = cv2.text.loadClassifierNM1('trained_classifierNM1.xml')
	er1 = cv2.text.createERFilterNM1(erc1)

	erc2 = cv2.text.loadClassifierNM2('trained_classifierNM2.xml')
	er2 = cv2.text.createERFilterNM2(erc2)

	regions = cv2.text.detectRegions(gray,er1,er2)
	rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions] 
	rects.extend(rects);

	no_overlap_rects=cv2.groupRectangles(rects,2,0.2)
	no_overlap_rects=tuple(map(tuple,no_overlap_rects[0]))


	for i in range(len(contours)):
		cnt = contours[i]
		x,y,w,h = cv2.boundingRect(cnt)
		if x-10>0:
			x = x-10
		if y-10>0:	
			y = y-10
		w = w+20
		h = h+20
		mask = np.zeros([h,w],np.uint8)
		mask[0:h,0:w] = img[y:y+h,x:x+w]
		regions = cv2.text.detectRegions(mask,er1,er2)
		rects = [cv2.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
		rects.extend(rects);

		no_overlap_rects=cv2.groupRectangles(rects,2,0.2)
		no_overlap_rects=tuple(map(tuple,no_overlap_rects[0])) 
		
		for rect in no_overlap_rects:
			cv2.rectangle(mask, rect[0:2], (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)

		img[y:y+h,x:x+w] = mask[0:h,0:w];	
		#resizedmask  = resize_img(mask,400,400)
		
		cv2.imshow("c result", mask)
		#cv2.waitKey(0)
		
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		#cv2.drawContours(bimg,[cnt],0,(255,255,255),1)	

	cv2.imshow("d",cloneimg)				
	cv2.imshow("s",img)	

	cv2.waitKey(0)

	return no_overlap_rects			

	
img  = img = cv2.imread('t1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray=  resize_img(gray,1000,1000)

no_overlap_rects = detect_letter_stage_1( gray )
no_overlap_rects = edge_present(gray , no_overlap_rects)
'''
for rect in no_overlap_rects:
  	cv2.rectangle(gray, rect[0:2], (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)

cv2.imshow("Text detection result", gray)

cv2.waitKey(0)
'''

