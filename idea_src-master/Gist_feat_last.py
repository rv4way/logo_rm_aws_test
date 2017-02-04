'''------------------- Program to calculate gist feature of an image-------------------------'''

#--creating Gist features--#
from __future__ import division
import cv2
import numpy as np
import numpy.matlib
from scipy.misc import imresize
import math
import sys
import csv 
import os
#--creating linspace function of matlab for future use--#
'''def linspace(a, b, n=100):
    if n < 2:
        return b
    diff = (float(b) - a)/(n - 1)
    return [diff * i + a  for i in range(n)]
'''        
def downN(x,N):
    #averaging over non-overlapping square image blocks
    #input x and output y
    x1,y1 = x.shape     #taking dimesntions of input x
    z1 = 1
    #print "value of z",z1
    nx = np.fix(np.linspace(0,x1,num=N+1))   #linspace function same as that of matkab adn scilab
    #print 'nx',nx
    ny = np.fix(np.linspace(0,y1,num=N+1))
    #print 'ny',ny
    y = np.zeros((N,N))    #create a 3D array y filled with zeros
    '''---check for nan--'''
    for xx in range(N):
        for yy in range(N):
            #t2=np.mean(x[nx[xx]:nx[xx+1],ny[yy]:ny[yy+1]],0)
            t2=np.mean(x[int(nx[xx]):int(nx[xx+1]), int(ny[yy]):int(ny[yy+1])],0)
         #   print 't1',t2.shape
          #  print t2
            #vai1= np.ma.shape(t2)
            #if(vai1[0] !=0):
            
                
            #print t2,type(vai1)
            temp1=np.mean(t2,0)
            v = temp1
            #print "v",v
            #if math.isnan(v):
                #print "nan"
            
                    
            y[xx][yy]=v      #same as that of scilab
    #print y
    return y


#--function to prefilter the image--#
def prefilt(img,fc=4):
    #fc=4 default
    # Input images are double in the range [0, 255];
    #For color images, normalization is done by dividing by the local luminance variance.
    #print 'in prefilt :',type(fc)
    w=5
    s1 = fc/np.sqrt(np.log(2))
    #img  = np.float64(img) #double precision value
    
    img = np.log(img+1)
    #print 'image size',img.shape
    img = np.resize(img, (256, 256))   #same as scilab  : resMat = resize_matrix(img, w, w,img);
    #img = np.pad(img, (w,w), 'symmetric')
    #print 'image size',img.shape
    sn,sm = np.shape(img)   #taking number of rows and cols of img
    #print 'in prefilt',sm
    c=1
    N=1
    n = np.amax([sn ,sm])   #same as that of n = max([sn sm]); (scilab)

    #print 'n',n
    n = n + np.mod(n,2)
    #print 'n',n
    #print "prefilt ddebug :",n-sn,n-sm
    #img = np.resize(img, (n-sn ,n-sm)) #same as that of : img = resize_matrix(img, n-sn ,n-sm,img)
    #img = np.pad(img, (n-sn ,n-sm), 'symmetric')
    #print 'img 2nd resize',img.shape
    # creating array temp which is equivalent to -n/2:n/2-1 in scilab
    x1= -n/2
    x2 = n/2-1
    temp=[]
    x1 = int(x1)
    x2= int (x2)
    for i in range(x1,x2+1):
        temp.append(i)
        
    temp = np.asarray(temp) #convert temp to numpy array
    fx,fy = np.meshgrid(temp,temp)   #same as that of fx,fy = meshgrid(-n/2:n/2-1)
    
    fx = np.asarray(fx)
    fy = np.asarray(fy)
    x = np.power(fx,2)+np.power(fy,2) 
    x1=-x
    s1_n = s1**2
    x3=np.divide(x1,s1_n)  #true divide (fx.^2+fy.^2)/(s1^2)
    #x=-x3
    x2=np.exp(x3)
    gf = np.fft.fftshift(x2)
    
    '''equivalent to scilab code :
         vec = [1 1 c N];
         gf = ones(vec(1),vec(2),vec(3),vec(4)).*.gf'''
    #gf = np.repmat(gf,(c,N))
    #gf = np.matlib.repmat(gf, c, N)
    gf = np.tile(gf,(1,1,c,N))
    #gf = np.resize(gf,(256,256))
    #print 'gffff',gf.shape
    #gf_s1,gf_s2 = gf.shape
    temp1 = np.fft.fft2(img)
    #temp1 = np.resize(temp1,(gf_s1,gf_s2))  #added
    #img = np.resize(img,(gf_s1,gf_s2)) #added
    temp2 = np.multiply(temp1,gf)
    temp3 = np.fft.ifft2(temp2)
    #temp3_n = np.squeeze(temp3)
    #print 'real!!!',np.real(temp3)
    output=img - np.real(temp3)   #equivalent to output = img - real(fftw(temp_v,1,2))
    #print 'output',output
    
    
    #--LOCAL contrast normallisation--#
    temp1 = np.mean(output,1)
    #cv2.imshow('prefily',temp1)
    temp2 = np.power(temp1,2)
    ''' SCILAB code is:
        vai1 =fft2(vai4);
        vai=fftw(vai1,1,2); //inverse fourier 2D transform

        vec1 = [1 1 c 1];
        localstd0 = sqrt(abs(vai));
        localstd = ones(vec1(1),vec1(2),vec1(3),vec1(4)).*.localstd0;
        equivalent python code'''
    temp3 = np.fft.fft(temp2)
    temp2 = np.multiply(temp3,gf[:][:][0][:])
    temp1 = np.fft.ifft(temp2)
    temp3 = np.absolute(temp1)
    temp2 = np.sqrt(temp3)
    localstd = np.tile(temp2,(1,1,c,1))
    output = np.true_divide(output,(0.2+localstd))   # equivalent  output1 = (output./(.2+localstd));
    #print "OUtput b4 cutting ",output
    #output = np.squeeze(output) #remove extra dimensions
    output1 = output[:,:,w:((sn-w)),w:((sm-w))]
    #output1 = output[w:((sn-w))][w:((sm-w))][:][:]
    #print "shape of output",output1.shape
    #print 'output 1',output
    #cv2.imshow('in1',output1)
    return output1
    
#--gist gabor--#
def gistgabor(img,param):
    #input is the image (2D image (B/W))and its paramters
    #outout global feature
    
    #image single conversion
    img = np.float32(img) 
    
    w = param['numberBlocks']
    G = param['G']
    be = param['boundaryExtension']
    #print "G",G.shape
    if img.ndim ==2 :  #condtition will work
        c=1
        N=1
        nrows, ncols = img.shape
        #print '2'
        #print "rows and columns",nrows,ncols
    if img.ndim ==3:
        nrows,ncols,c = img.shape
        N=c
        #print "3"
    if img.ndim ==4:
        nrows,ncols,c,N = img.shape
        #print "N and c",N,c
        #--image reshape--#
        
        img = np.reshape(img, (nrows,ncols,c*N), order="F")
        N = c*N
        #print "4"
    ny,nx,Nfilters=G.shape
    W=w*w
    temp1 = W*Nfilters
    #print "Temp1 , N",ny,nx,Nfilters
    g = np.zeros((temp1,N))  #creating g which is filled with zeros
    #g_temp=[]
    #x,y=np.shape(g)
    #print "co ordinates of g",x,y
    
    #img = np.resize(img,(be,be))  #img = resize_matrix(img, be ,be)
    img = np.pad(img,(be+5,be+5),'symmetric') #img be be throws an error
    #print "after resize",img.shape
    img = np.fft.fft2(img)  #2d fourier transform
    #print "image_b4!!",img.ndim
    #img = np.float32(img)  #single precision conversion
    #cor_x,cord_y = img.shape
    '''for val in range(cor_x):
        img[val][:] = np.real(img[val][:])'''
    
    #print "image!!!",img.ndim    
    k=0
    for n in range(Nfilters):
       # print "k",k
        temp = G[:,:,n]            
        #print "dims of temp",temp.ndim
        
        #print 'temp',temp
        #v1 = [1,1,N]
        #temp2 =np.kron( np.ones((v1[0],v1[1],v1[2])),temp)
        temp2 = np.matlib.repmat(temp, 1, N)
        temp_1 = np.multiply(img,temp2) #here
        #print "temp1",temp_1
        temp_1 = np.squeeze(temp_1)
        #print "temp1",np.ndim(temp_1)
        ig = np.absolute(np.fft.ifft2(temp_1))
        #print 'iiiiii',ig
        v=downN(ig,w)
        #x_cord,y_cord = v.shape
        #print 'v_temp ', v
        #v = np.squeeze(v)
        #print "v shape :",v.shape
        '''for vik in range(0,4):
            #print "value of v",v[vik][0][:]
            g=np.append(g,v[vik][0][:])
          '''
        v_temp = np.reshape(v,(W,N))    
        #print 'v_temp ', v
        #print "v_temp",v_temp.shape
        g[k:k+W,:] = v_temp    
        #g[k+1:k+W,:] = matrix(v, [W N])
        k = k + W
           
    return g
#--function to resize image--#
def imresizecrop(img, M, METHOD):
    #Output an image of size M(1) x M(2)
    #print 'in resize',M
    if METHOD =='':
        METHOD = 'bilinear'
    if np.ndim(M) == 1:
        M = [M[0],M[0]]
    #print "in resize function"    
    #print 'shape of image',img.shape
    vai_1 ,vai_2 = img.shape #check for shape if image is too large then scaling is 0
    #temp1=[M[0]/np.ma.size(img,0) ,M[1]/np.ma.size(img,1)]
    #print 'M[0], vai_1',M[0],vai_1
    #print 'not zero',M[0]/vai_1
    temp_1 = [M[0]/vai_1]
    temp_2 = [M[1]/vai_1] 
    #print 'temp 1,temp2 ',temp_1,temp_2
    scaling= np.concatenate((temp_1,temp_2)).max()
    #print 'scaling',scaling
    #print 'M',M
    vai = [np.ma.size(img,0),np.ma.size(img,1)]
    #print 'vai3',vai
    vai2 = np.multiply(vai,scaling)
    #print 'vai2',vai2
    vai = np.asarray(vai2)
    #print 'final',vai
    newsize = np.around(vai)
    newsize = np.int32(newsize)
    #if newsize == []:
     #   newsize = [256,256]
    #cv2.imshow('HIIII',img)
    #img = np.resize(img,newsize)
    img = imresize(img, newsize, interp='bilinear', mode=None)
    #cv2.imshow('LAWL',img)
    if scaling ==0:
        print 'image greater then 256'
        sys.exit(0)
        #break
    else:
        
        nr,nc = img.shape
    #print 'NZ',nr,nc
        x1 = (nr-M[0])/2
        x2 = (nc-M[1])/2
        sr = math.floor(x1)
        sc = math.floor(x2)
        #remove this then no zeros features are detected
        #img = img[sr+1:(sr+M[0]+1),sc+1:(sc+M[1]+1)]
        img = img[int(sr+1):int(sr+M[0]+1), int(sc+1):int((sc+M[1]+1))]
        
    return img
    
    
#--function to compute gabor features--#    
def CreateGabor(or1 ,n):
    '''function to create features here or1 == number of orientation per scale
    and n is the image size
    output is the transfer function for a jet of gabor feature
    Precomputes filter transfer functions. All computations are done on the
    Fourier domain. 

    If you call this function without output arguments it will show the
    tiling of the Fourier domain.'''
    
    or1 = np.asarray(or1, dtype=None, order=None)
    n = np.asarray(n,dtype=None, order=None)
    Nscales =  np.ma.size(or1, axis=0)
    #print Nscales
    Nfilters = np.sum(or1, axis=0, dtype=None, out=None, keepdims=False) # same as that of Nfilters = sum(or1)
    #print 'google',Nfilters,Nscales
    if np.ma.size(n, axis=0)==1:
        n1 = n[0]
        n = [n1,n1]
    n = np.asarray(n,dtype=None, order=None)    
    l=0
    w, h = 4, 32
    param = [[0 for x in range(w)] for y in range(h)]
    #param is a 2D MATRIX HERE 
    for i in range(Nscales):
        or2 = or1[i]
        #param[l].append([])
        for j in range(or2):
            
            param[l][:]=[.35 ,.3/math.pow(1.85,(i-1)) ,math.pow(16*or1[i],2/math.pow(32,2)) ,np.pi/(or1[i])*(j-1)]
            l=l+1
            #assiging value to param
    param = np.asarray(param)
    #print 'yo',param
    #print 'praram over'
    #Frequencies:
    temp_1 = n[1]
    temp_2 = n[0]
    #print temp_1
    x1 = -(temp_1/2)
    x2 = (temp_1/2)-1
    x3 = -(temp_2/2)
    x4 = (temp_2/2)-1
    n1=[]
    n2=[]
    x1 = int(x1)
    x2 = int(x2)
    x3 = int(x3)
    x4 = int(x4)
    for i in range(x1,x2+1):
        n1.append(i)
    for i in range(x3,x4+1):
        n2.append(i)
    #print n1
    #--conversion to numpy array--#
    n1 = np.asarray(n1)
    n2 = np.asarray(n2)
    #--using meshgrid--#
    fx,fy = np.meshgrid(n1,n2)  #same as that of  [fx, fy] = meshgrid(-n(2)/2:n(2)/2-1, -n(1)/2:n(1)/2-1)
    #--calcultating fr--#             
    fx1 = np.power(fx,2)
    fy1 = np.power(fy,2)               
    temp =  fx1+fy1
    temp2 = np.sqrt(temp) #taking square root
    fr=np.fft.fftshift(temp2)
    #--calculating--t--#
    a1 = fx + 1j*fy #creating complex number
    x = np.angle(a1, deg=0) #calculating its angle
    t=np.fft.fftshift(x)
    
    G=np.zeros((n[0],n[1],Nfilters))
    #print "G",G[:,:,0].shape
    #print "param",param
    for i in range(Nfilters-1):
        tr=t+param[i,3] 
        
        tr=tr+2*np.pi*(tr<-(np.pi))-2*(np.pi)*(tr>(np.pi))
        #print 'tr',tr
        
        temp2  = -10*param[i,0]*(fr/n[1]/param[i,1]-1)**2-2*param[i,2]*np.pi*tr**2   #same as that of temp3 = -10*param(i,1)*(fr/n(2)/param(i,2)-1).^2-2*param(i,3)*(%pi)*tr.^2;
        #print 't2',temp2
        x = np.exp(temp2) #x = exp(temp3);
    
    
        #print 'x',x
        G[:,:,i]=x
        
   # print "shape of dimension of G :",G
    return G

#--return the gist of image--#
def LMgist(D,HOMEIMAGES,param,HOMEGIST=''):
    if( HOMEIMAGES!='' and HOMEGIST!= ''):
        precomputed =1
    else:
        precomputed =0    
     
    #--checking type of image--#
    if type(D) == np.ndarray:
        #print 'image is good'
        Nscenes = np.ma.size(D, axis=2) #take the size along the 2nd axis
        #print (Nscenes)
        typeD=2
    param['boundaryExtension'] = 32 
    
    if(param == {}):  #--Empty parameters--#
        param  = {'imagesize':[256, 256],'orientationPerScale':[8, 8, 8, 8],'numberBlocks':4,'fc_prefilt':4,'boundaryExtension':32}
        #print param['orientationPerScale']
        param['G'] = CreateGabor(param['orientationPerScale'],[32])
    else:
        #temp_8 = param['imagesize']+2*param['boundaryExtension']
        temp = param['imagesize']
        temp2 = 2*param['boundaryExtension']
        value = temp[0]+temp2
        value2 = [value]
        param['G'] = CreateGabor(param['orientationPerScale'],value2)
    
    #t_v = param['G'] #(uint8(param.G(:,:,1)*255),'param1.jpg'       
    #t_v = t_v[:,:,10]*255
    #t_v =np.uint8(t_v)
    #cv2.imshow('hey',t_v)
    Nfeatures = np.ma.size(param['G'],2)*np.power(param['numberBlocks'],2)           
    #print type(Nfeatures)," ",type(Nscenes)
    gist = [] #list of gist vectors
    
    todo = 1
        
    img = D
    img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY ) #image resized
    #print "jusst before resize",img
    #cv2.imshow('Gray Scale',img)
    img = imresizecrop(img, param['imagesize'], 'bilinear')
    img = np.asarray(img)
    #cv2.imshow('after imresize',img)
    img = img-np.amin(img[:])
    
    #print img
    img = 255*img/np.amax(img[:])   #return 0's    
    #print 'img',img.shape
    '''cv2.imshow('input',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    #print "shape of image : ",img        
    #cv2.imshow('hey',img)
    output= prefilt(img, param['fc_prefilt'])
    #print 'output',output  #correct working uptohere
    #cv2.imshow('output',output)
    output = np.squeeze(output)
    #cv2.imshow('Prefilt Output1',output)
    #print 'hiiiiii',out_new.shape
    g = gistgabor(output,param)
        
        
    #gist = g
    '''for i in g:
        if(i ==0):
            g=np.delete(g,i) #remove zeros from the array
    '''        
    gist = g
    gist = np.squeeze(gist)
    #print "final :",g
    return gist,param

#---code flow start from here ---#
#---use this to load images from a dir in root dir---#
def allFeat():
    
    i =0
    rootdir = "E:\\Done Fea"   #logos_feature is the folder contatining sub dir in which companies logos are present
    for subdir, dirs, files in os.walk(rootdir):
        #print dirs
        #print files    
        for sub1 in dirs:  #taking the sub directories
            #print sub1
            count =0
            newdir = rootdir+"\\"+sub1  
            i =0
    
            for subdir1, dirs1, files1 in os.walk(newdir):
                if 'Correct' in subdir1 or 'False' in subdir1 or 'Others' in subdir1:
                    
                    for subdir2,dirs2,files2 in os.walk(subdir1):
                        print subdir2
                        for subs,dirs,files3 in os.walk(subdir2):
                            
                            for file1 in files3:          #every file in subdirectry
                                if '.jpg' in file1 or '.png' in file1:   #checking the extension of file.
                                    count = count+1
                                    newdir1 = subdir2+"\\"+file1   #making the image full path
                                    print newdir1
                                    img=cv2.imread(newdir1)  #reading the iamge
                                    #cv2.imshow('image1',img)
                                    param  = {'imagesize':[256, 256],'orientationPerScale':[8, 8, 8, 8],'numberBlocks':4,'fc_prefilt':4} #settign parameter
                                    gist,param = LMgist(img,'',param)
                                    with open(subdir2+"\\"+'feature.csv', 'a') as myfile:
                                        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                                        wr.writerow(gist)
                                    print count
                
    print 'done'


def singleImage2(img):
#--use this to load a single image--#
#test _logo_200_524r.png
#test 2 _logo_200_510r
    #img = cv2.imread(path)
#--create parameters--#
    param  = {'imagesize':[256, 256],'orientationPerScale':[8, 8, 8, 8],'numberBlocks':4,'fc_prefilt':4}
#print param['imagesize']

    #cv2.imshow('Input',img)#C:\Use
    gist,param = LMgist(img,'',param)
    #gist = np.asarray(gist)

    #print "at last :",gist
    #print "shape :",len(gist)
    
    #print 'done'
    return gist 
    
#g = singleImage('_logo_200_510r.png')
def single_image():
    rootdir = 'E:\\New2'
    list1 = os.listdir(rootdir)
    count =j = 0
    for i in list1:
        new_dir = rootdir+'\\'+i
        print new_dir
        img_read =cv2.imread(new_dir)
        feature_gist = singleImage2(img_read)
        count = count+1
        if count>1000:
            count=0
            j=j+1
        path1 = 'E:\\DataBase\\NegativeSelection\\Gist'
        file = path1+'\\Negatvie'+str(j)+'.csv'
        with open(file, 'a') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(feature_gist)
        print 'done with',i