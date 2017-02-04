'''adding salt and pepper noise'''
import cv2
import numpy as np

def noise_addition(image):
    img = image
    x,y,z = img.shape
    for i in range(x):
        for j in range(y):
            temp = np.random.random(size=None)
            if temp > 0.98: 
                img[i,j,:] = [255,255,255] 
                
            if temp <0.03:
                img[i,j,:] = [0,0,0]   
    
    
    #cv2.imshow('output',img)
    return img
    
'''    
def start(path):
    img = cv2.imread("C:\\Users\\4way\\input3.jpg")
    cv2.imshow('input',img)
    noise_addition(img)
    
    
start()
'''