from __future__ import division
import saltandpepper1 as sp
import Gist_feat_last
import HOG_feat2
import cv2
import numpy as np
import random
import math
import os
import csv
import json
import pandas as pd

'''create the afine transform of the input image and write the feature of those images '''




with open('Properties.json', 'r') as fp:
    data = json.load(fp)

'''search_in = pd.read_csv(data["ImageCsv"],sep=',',header=None)
search_in = np.asarray(search_in)'''
feat_gist = data["PositiveGist"]
feat_hog = data["PositiveHog"]
# m_a = data["MongoUrl"]

# c = MongoClient(m_a)   #taking instance of mongo client
# mer4 = data["ImageDatabase"]
# db = c[mer4]    #Main DataBase
# db2_g = db.Gabor
# db2_h = db.Hog
# db_count = db.Image_count
#--padd the boundraies--#    
def padd(img):
    #img = cv2.imread(path)
    x,y,z = img.shape #x is number of rows and y is no of columns
    #--take 30% of rows--#    
    temp_x = round(0.4*x)
    temp_y = round(0.4*y)
    #--take the regions of start and end which needs to be padd along x axis--#
    x_region_end = img[:,(y-2):,:]
    x_region_start = img[:,:2,:]
    #--padd the region in x axis--#
    for i in range(50):
        
        img = np.concatenate((img, x_region_start), axis=1)
        img = np.concatenate((x_region_end,img),axis=1)
    
    #--padding for y axis--#
    y_region_end = img[(x-1):,:,:]
    y_region_start = img[:1,:,:]
    for i in  range(50):
        img = np.concatenate((y_region_start,img),axis=0)
    #vis = np.concatenate((y_region_start,vis2),axis=0)
        img = np.concatenate((img,y_region_end),axis=0)
     
    #cv2.imshow('final  output',img)
    return img    
    
    
#--return anumber between 0-0.3--#
def random_1():
    
    z = random.random()*0.6       
    return z

def count_rows(arr):
    rows,col = arr.shape
    #print 'rows and columns',rows,col
    num_zeros=0
    for i in range(rows):
        if np.all(arr[i,:]==0):
            num_zeros = num_zeros+1
     
    ret = (num_zeros/rows)*100
    #print 'ret',num_zeros
    return ret              
#--change intensity by changing gama values--#
def adjust_gamma(image, gamma=1.0):
	
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
#---add feature to database--#	
def add_feature_to_database(name,feature,type_feat):
    if feature == "Gabor":

        temp = db2_g.find_one({"_id":name})
        if temp is not None:
            ms = temp["feature"]
            ms_new = np.concatenate(ms,feature)
            db2_g.update_one({"_id":name},{"$set":{"feature":ms_new}})
        else:
            d_temp = {"_id":name,"feature":feature}
            res = db2_g.insert_one(d)
    if feature == "Hog":
        temp = db2_h.find_one({"_id":name})
        if temp is not None:
            ms = temp["feature"]
            ms_new = np.concatenate(ms,feature)
            db2_h.update_one({"_id":name},{"$set":{"feature":ms_new}})
        else:
            d_temp = {"_id":name,"feature":feature}
            res = db2_h.insert_one(d)

#-- function to genreate affine transform--#    
def generate_affine(img,name):
    for i in range(50):
    
        print i
        #padd the image
        img1 = padd(img)
        #cv2.imshow('after concatat',img1)
        gamma_1 = random_1()


        gamma_main = random.randrange(1,3,1)
        gamma_last = gamma_main + gamma_1
        noise_addition = adjust_gamma(img1, gamma=gamma_last)#change the gammmaa of the image randomly
        noise_ad2 = sp.noise_addition(noise_addition) #add the salt and peppr noise to the image
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        (h, w) = img1.shape[:2]
        rows2,cols2,ch2 = img1.shape #ch=3 rows = hieght and col is width
        
        #creating the rotation matrix
        x=random.randrange(-5,5,1)
        
        y= random.random()
        z=x+y
        x=math.radians(z)

        sin=math.sin(x)
        cos=math.cos(x)
        rotation = [[cos,sin],[-sin,cos]] #2x2 rotation matrix
        #shear matrices
        
        shear_fact_x = -0.3+random_1() #take random values for both x and y
        shear_fact_y = -0.3+random_1()
        shear_x = [[1,shear_fact_x],[0,1]]
        shear_y = [[1,0],[shear_fact_y,1]]
        shear_new = np.array(shear_x,dtype=np.float32)
        rotation_new = np.array(rotation,dtype=np.float32)
        
    
        #multiplying 2D matrices
        afine_t= np.dot(shear_new, rotation_new)
        shear_new = np.array(shear_y,dtype=np.float32)
        afine_t2 = np.dot(afine_t,shear_new)
        #afine_t = shear_new
        pts2 = np.dot(pts1,afine_t2)#new points use for warping the image
        #print pts2
        #cv2.imshow('input',img)
        s11,s12,s13 = noise_ad2.shape
        for r in range(s11):
            for c in range(s12):
                if np.all(noise_ad2[r,c,:]==0):
                    
                    noise_ad2[r,c,0]=noise_ad2[r,c,0]+1
                    noise_ad2[r,c,1]=noise_ad2[r,c,1]+1
                    noise_ad2[r,c,2]=noise_ad2[r,c,2]+1
        
        M = cv2.getAffineTransform(pts1,pts2)
        dst = cv2.warpAffine(noise_ad2,M,(cols2,rows2))#get afine of image
        #cv2.imshow('imshow',dst)
        s1,s2,s3 = dst.shape
        #removing the extra black areas from the images so that only logos is visible.

        count = 0
        count2 = 0
        for r in range(s1):
            num = count_rows(dst[r,:,:])
            if num >= 30:
                count=count+1
                #break
            else:
                #count = count+1
                break
        for c in range(s2):
            num2 = count_rows(dst[:,c,:])
            if(num2>=30):
                count2 =count2+1
                #break
            else:
                #count2 = count2+1
                break
        
        r1  = s1-1
        while(r1>=0):
            num3 = count_rows(dst[r1,:,:])
            if(num3 >= 30):
                #break
                r1=r1-1
            else:
                #r1=r1-1
                break
        c1 = s2-1
        while(c1>=0):
            num4 = count_rows(dst[:,c1,:])
            if (num4 >=30):
                #break
                c1=c1-1
            else:
                #c1 = c1-1 
                break       
       # print 'first and  last also col1 and col2',r1,count,count2,c1
    
        dst2 = dst[count:r1,count2:c1]
        count=r1=c1=count2=0
        s1,s2,s3 = dst2.shape
       
        #add the gist and hog feature of the image
        if s2 !=0 and s1 !=0:
                    
#        cv2.imshow('noise',dst2)
        #dst2 remove black.
            gist_feat = Gist_feat_last.singleImage2(dst2)
            hog_feat = HOG_feat2.hog_call(dst2)
            gist_feat = np.concatenate((gist_feat,[1]))
            hog_feat = np.concatenate((hog_feat,[1]))
            '''add_feature_to_database(name,gist_feat,"Gabor")
            add_feature_to_database(name,hog_feat,"HOG")'''

            with open(feat_gist+name+'.csv', 'a') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(gist_feat)
            with open(feat_hog+name+'.csv', 'a') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(hog_feat)

    
    return i

def modify(name,search_in):
    x,y = search_in.shape
    for i in range(x):
        if name == search_in[i,0]:
            search_in[i,1]=search_in[i,1]+1
            np.savetxt(data["ImageCsv"], search_in, delimiter=",")
            return 0
    
    temp = np.asarray([name,1])

    with open(data["ImageCsv"], 'a') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(temp)


def modify_database(name):
    c = db_count.find_one({"name":name})
    if c is not None:
        temp = c["count"]
        db_count.update_one({"name":name},{"$set":{"count":temp+1}})
    else:
        d = {"name":name,"count":1}
        res = db_count.insert_one(d)
        


#execution starts here   
def single_affine(img,name):
    #img = cv2.imread(path)
    #modify_database(name)
    #cv2.imwrite("/root/ideaswirw/imageprocessing/DataBase/1.png",img)
    #print 'image is',img
    print img.shape
    gist_feat = Gist_feat_last.singleImage2(img)
    
    hog_feat = HOG_feat2.hog_call(img)
    
    gist_feat = np.concatenate((gist_feat,[1]))

    hog_feat = np.concatenate((hog_feat,[1]))
    
    with open(feat_gist+name+'.csv', 'a') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(gist_feat)
    with open(feat_hog+name+'.csv', 'a') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(hog_feat)
    '''
    add_feature_to_database(name,gist_feat,"Gabor")
    add_feature_to_database(name,hog_feat,"HOG")
    '''
    ret = generate_affine(img,name)
    
    return ret

