'''HOG FEATURE'''
from __future__ import division
import cv2
from skimage.feature import hog
from scipy.misc import imresize
import numpy as np 
  

#import matplotlib.pyplot as plt

def rem_zeros(img):
    x,y = img.shape
    for i in range(x):
        for j in range(y):
            if img[i,j]==0 :
                img[i,j] = img[i,j]+1
            
    return img
    
def HOG_1(image):
    r,c = image.shape
    if r!=64 or c!=64:
        image = imresize(image,(64,64))
    #cv2.imshow('input',image)        
    #print 'r and c',r,c
    
    #cv2.imshow('Resize',image)    
    fd, hog_image = hog(image, orientations=20, pixels_per_cell=(32,32),cells_per_block=(1, 1) ,visualise=True,feature_vector=True)
    #print fd.shape
    #cv2.imshow('HOG',hog_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return fd
#divide image into rgb componnent    
def colour_map(img):
    return  img[:,:,0],img[:,:,1],img[:,:,2]

    
        
def color_div_feat(blue,green,red):
    shape_x,shape_y = blue.shape
    number_pix = shape_x*shape_y
    green_new = rem_zeros(green)
    red_new = rem_zeros(red)
    green_new = np.float64(green_new)
    blue = np.float64(blue)
    red_new = np.float64(red_new)
    b_g = np.true_divide(blue,green_new)
    g_r = np.true_divide(green,red_new)

    b_g = np.multiply(b_g,255)
    g_r = np.multiply(g_r,255)


    h1,h2 = np.histogram(b_g, bins=20, range=None, normed=False, weights=None, density=None)
    h3,h4 = np.histogram(g_r, bins=20, range=None, normed=False, weights=None, density=None) 

    h1_n = np.true_divide(h1,number_pix) 
    h3_n = np.true_divide(h3,number_pix)
    return h1_n,h3_n                    
                                    
                                                                    
def hog_call(image):
    blue,green,red = colour_map(image)
    img2= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feat = HOG_1(img2)
    feat2 = HOG_1(blue)
    feat = np.concatenate((feat,feat2))
    feat2 = HOG_1(green)
    feat = np.concatenate((feat,feat2))
    feat3 = HOG_1(red)
    feat = np.concatenate((feat,feat3))
    #print feat.shape
    feat2,feat3 = color_div_feat(blue,green,red)
    feat = np.concatenate((feat,feat2))
    feat = np.concatenate((feat,feat3))
     
    return feat #feat is the feature of image
    
