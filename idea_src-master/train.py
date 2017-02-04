from __future__ import division #import division from future
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd   #use pip install pandas to install this
import os
from sklearn.externals import joblib  #save the data
import random,json
import sys
''' this will create the classifier for the new image that is comming in from the server using 
the negative search approach'''

'''----------------------------------Negative search algo 2---------------------------------''
we do not delete the old false data we instead keep 70% of it and add 70%of new false data.
to accomodate with the biasing we increase append the true data at the data.'''
#download dataset and test both the algos
'''re training the classifier'''
with open('Properties.json', 'r') as fp:
    data = json.load(fp)

feat_true = data["train_tr"]
feat_false=data["train_fl"]
neg_dirg = data["RetrainNegativeGist"]
neg_dirh = data["RetrainNegativeHog"]
classi = data["Classifier"]
root_dir = data["Train"]
#classifier_loc = "E:\\Classifier_Gist2"
rec_control = 1
#return the flag values
def random_list(length,perc,l1):
    count = length*perc #number of elements
    x = l1
    
    h = random.randrange(0,length)
    if h in x:
        return random_list(length,perc,x)
    else:
        x.append(h)
        if len(x)==count:
            return x
        else:
            return random_list(length,perc,x)
#split the train and test data
def split_new(training_data1,labels,split):
    x,y = training_data1.shape
    new_row = int(0.8*x)
    training_data = np.zeros((new_row,y))
    label = np.zeros(new_row)
    false_true=false_false = int(0.5*new_row)
    f = []
    count =0
    for i in labels:
        if i==1:
            count=count+1
        else:
            break
    count2=count3=0        
    h = random.sample(range(count),false_true)
        
    
    for i in h:
        training_data[count2,:] = training_data1[i,:]
        label[count2] = 1
        count2 = count2+1

    h2 = random.sample(range(count,int(x)),false_true)
    for i in h2:
        training_data[count2,:] = training_data1[i,:]
        count2 = count2+1
    h.append(h2)
    f = h
    #print 'devision is',count2,count3  
    return training_data,label,f
#count the number of false positive and return them
def count_zeros(arr,test):
    '''--arr is the predicted array and test is the test set
    --'''
    count_z =count3 =  0
    counter2 = 0
    x,y = test.shape
    ret = np.zeros((x,y))
    shape = arr.shape
    
    for i in arr:
        if i==0:
            count_z = count_z+1
            count3 =count3+1
        elif i==1:
            ret[counter2,:] = test[count3,:]
            
            count3=count3+1
            counter2 = counter2+1
    #print 'number of zeros',counter2,shape
    results = (count_z/shape[0])*100
    #print 'ret value',results
    ret = np.asarray(ret[:counter2,:])
    return results,ret

def re_train(new_false_new,new_false_old,train_set,labels,shape_y,new_comp2,num12,f_type):
    '''--new false contain the feature of images which gaves false positives
    trainset is the old training setand labels are there corres labels
    new comp2 is company name and num12 is number of classifier--'''
    #print 'number of false positive',new_false.shape
    global rec_control
    rec_control = rec_control+1
    num12 = num12+1
    rows ,columns = train_set.shape #previous training_set
    print rec_control
    flags = flag_prev = []
    prev_false = np.zeros((rows,columns))
    prev_true = np.zeros((rows,columns))
    if new_false_new.ndim ==1:
        new_row=new_false_new.shape
    else:
        
        new_row,new_col = new_false_new.shape #shape of new false
    count1=count2=count3=0

    #need to check the oldfalse data
    for i in labels:
        if i ==0 and count1<rows :
            
            prev_false[count2,:] = train_set[count1,:]
            #print train_set[count1,:shape_y]
            count2=count2+1
            count1 = count1+1
        elif i==1 and count1<rows:
            
            prev_true[count3,:] =  train_set[count1,:]
            #print train_set[count1,:shape_y]
            count3=count3+1
            count1 = count1+1
    '''old true and false matrices'''
    
    prev_true = np.asarray(prev_true[:count3,:]) #old true value        
    prev_false = np.asarray(prev_false[:count2,:]) #new true value
    #taking 30%old data and 70%new data randomly
    #print 'prev',prev_false.shape
    true_row,true_false = prev_true.shape
    count1,row1 = prev_false.shape
    if rec_control ==2: #if it is the first iteration
        f_old = round(0.7*(count1))
        f_new = round(0.7*(new_row))
        #print 'new and old',int(f_old+f_new),count1
        #print 'new_false',f_old+f_new
        false_count=0
        false_data = np.zeros((int(f_old+f_new),shape_y))
        x4 = 0.7*count1
        flag_prev = random.sample(range(count1), int(x4))
        for i in flag_prev:
            false_data[false_count,:] = prev_false[i,:]
            false_count=false_count+1
        x4 = 0.7*new_row
        flags = random.sample(range(new_row), int(x4))
    
    
        for i in flags:
            false_data[false_count,:] = new_false_new[i,:]
            false_count = false_count+1
        
        print 'false data succ'
        #print 'false_data_shape',false_data.shape
        f_x,f_y = false_data.shape 
        diff = int(f_x)-int(true_row)
        print 'diffence is ',diff 
        
        if diff > 0:
            try:
                perc = diff/true_row
                print 'perc is ',perc
            except Exception,e:
                print('Error on line {}',str(sys.exc_info()[-1].tb_lineno), type(e), e)
            if perc>1 :
                p1 = int(perc)
                
                for i in range(p1+2):
                    
                    prev_true = np.concatenate((prev_true,prev_true))
                
            else:
                
                prev_true = np.concatenate((prev_true,prev_true))
        
        elif diff<0:
            print 'triming data'
            diff =  abs(diff)
            prev_true = np.asarray(prev_true[:int(true_row)-diff,:])
            
            
        
        p_tx,pt_y = prev_true.shape
        
        if int(f_x) < int(p_tx):
            prev_true = prev_true[:f_x,:]
            #print 'small'
        #print 'true_data',prev_true.shape
        true_row2,col_tru2 = prev_true.shape
        new_train = np.concatenate((prev_true,false_data))
        #print 'newtrain done'
        '''for i in range(true_row):
        new_train[i,:] = prev_true[i,:]
    
    for i in range(true_row,int(f_old+f_new)):
        new_train[i,:] = false_data[i,:]'''
    else: #for all other iteration
        false_data= np.concatenate((prev_false,new_false_new)) #add the previous false data to new false data
        f_x,f_y = false_data.shape 
        diff = int(f_x)-int(true_row)
        print '2nd'
        if diff>0:
            perc = diff/true_row
            p1 = int(perc)
            print "prec34",p1
	    for i in range(p1+2):
                print 'loop here '
                prev_true = np.concatenate((prev_true,prev_true))
                        
        p_tx,pt_y = prev_true.shape
        print 'true data'
        if int(f_x) < int(p_tx):
            prev_true = prev_true[:f_x,:]
            #print 'small'
        print 'true_data_af',prev_true.shape
        true_row2,col_tru2 = prev_true.shape #concatenated true data
        new_train = np.concatenate((prev_true,false_data))
    #print 'false',false_data.shape
    #print 'true',prev_true.shape        
    #print 'total',new_train.shape
    labels_new = []    
    for i in range(true_row2):
        labels_new.append(1)
    for i in range(int(f_x)):
        labels_new.append(0)
        
    labels_new = np.asarray(labels_new)#new labels
    print 'labels and data',labels_new.shape,new_train.shape 
    #X_train,y_train,f = split_new(new_train, labels_new, 0.80)
    
    est = RandomForestClassifier(n_estimators =10,max_features='auto',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0,max_leaf_nodes=None,n_jobs=1)
    #--fitting data and labels--#
    est.fit(new_train,labels_new)
    ##--making the falsetestdata--##
    print "datafitted"
    if rec_control ==2:
        test_new = np.zeros((new_row,shape_y))
        temp = 0
        for i in range(new_row):
            if i in flags :
                lol=1
            else:
                test_new[temp,:] = new_false_old[i,:]
                temp = temp+1
        test_new = np.asarray(test_new[:temp,:])
    else:   
        test_new = np.asarray(new_false_old)
    save_location = classi+f_type+'/'+str(new_comp2)+'_'+str(num12)+".pkl"        
    joblib.dump(est, save_location,compress=9)
    
    test_x,test_y = test_new.shape
    if test_x == 0:
        return num12
    else:
    
    
        predcition_now = est.predict(test_new)
        predcition_now = np.asarray(predcition_now)
        #print predcition_now
        num,newfalse = count_zeros(predcition_now,test_new)
    
        #newfalse =np.concatenate((newfalse,test_new))
        #print 'score of Nsa',num,newfalse.shape
    
        #print 'prediction is ',predcition_now
    
        #save the classifier
        if num>99.99 or rec_control == 20:
            return rec_control #return  0 if we get the accuracy
        else:
            newfalse = np.asarray(newfalse)
            return re_train(newfalse,new_false_old,new_train,labels_new,shape_y,new_comp2,num12,f_type) #else recompute
    

def evac_correct(dataset):
    rows,columns = dataset.shape
    labels = dataset[:,columns-1]
    count = 0  
    for i in labels:
        if i ==1:
            count=count+1
        else:
            break
    data_corr = dataset[:count+1,0:columns-1]        
    labels = np.zeros(count)
    
    return data_corr,labels    


#--function to re train the classifier and make new classifier captures the falsepositive we getfrom1st classifier--#

def re_train_prefilt(save_location,new_comp2,train_set,labels,shape_y,num12,f_type): #Ecalculate all the false positive
    '''save_loacation == location of 1st clasifier
    new_comp2==name of company
    train_set,trainlabels == oldtrain set and labels
    shape_y =shape of each feature vector
    num12 == number of classifier'''
    global rec_control
    num12=num12+1
    clf = joblib.load(save_location) #load the [revious classifier
    root_dir1 = root_dir +f_type

    list1 = os.listdir(root_dir1) #root dir contain the negative data
    rows ,columns = train_set.shape
    
    new_false=np.zeros((2,shape_y)) #list to contain false data
    #print list1
    for list2 in list1:
        l1 = list2.split(".")
        if l1[0] != new_comp2:  #if its the diffrent brand then 
            #print l1[0]
            new_dir = root_dir1+"/"+list2
            if (f_type in new_dir):
                data_test = pd.read_csv(new_dir)
                data_test = np.asarray(data_test)
            #test_data,labels_1 = evac_correct(data_test)test data is correct images and labels is there corresponding images
                x,y = data_test.shape 
                temp = np.zeros((x,y))
                #print 'prediction',data_test
                predict = clf.predict(data_test) #we predict the result
                #print 'predicted'
            #print 'predict b4',predict
                counter=0
                counter4=0
                for j in predict:
                    if j==1:
                        temp[counter4,:]=data_test[counter,:]   #newfalse is the new false data
                        counter= counter+1
                        counter4= counter4+1
                    else:
                        counter = counter+1
            
                temp = temp[:counter4,:]
                new_false = np.concatenate((new_false,temp)) #initial false data       
        else:
            continue
    
    #number of false data we need to consider       
    new_false = np.asarray(new_false[2:,:])
    
    #new false contain the brands which gaves the false positive to the classifier.
    try:
        new_row,new_col = new_false.shape#new false positive
       # print 'here'
        #print new_row,new_col
        x = re_train(new_false,new_false,train_set,labels,shape_y,new_comp2,num12,f_type) #trainset is old training set and labels are previous labels
        rec_control = 1
        return x
    except Exception,e:
        print('Error on line {}',str(sys.exc_info()[-1].tb_lineno), type(e), e)
        rec_control = 1
        return 0
    #need to check the oldfalse data
    
#create the first classifier                  
def classifier(training_data,data_false,directory,priori_prob,shape_y,f_type):
    #function to compute the classifier for a label and then save it
    '''this function creates the initial 1st classifier.'''
    try :
        print '\n'

        # print type(training_data),type(data_false),type(directory),type(priori_prob),type(shape_y), type(f_type)
        #print len(training_data),len(data_false),directory,priori_prob,shape_y,f_type
        #print 'Company is :',directory
        #for i in range(4):    
            #training_dataf= np.asarray(training_dataf)
        #print 'shape of false data',data_false.shape
        #print 'shape of training_dataf',training_dataf.shape
            
        training_dataf= np.asarray(data_false) 
        #training_dataf = training_dataf[:count1,:] #false training data so that both havesame size
        
        r1,c1 = training_data.shape
        r2,c2 = training_dataf.shape   
        label_true = []
        label_false = []
        #--creating labels for true and false data--#


        for m in range(r1):
            label_true.append(1)
        
        for n in range(r2):
            label_false.append(0)
            
        label_true = np.asarray(label_true)
        label_false = np.asarray(label_false) 
        #print 'b4imputer'
        #--removing nans by the medians--#


        imp = Imputer(strategy = 'median')
        imp.fit(training_data)
        training_data = imp.transform(training_data)
        
        imp.fit(training_dataf)
        print "here "
        training_dataf = imp.transform(training_dataf)
        

        #print 'after'
        #--final training data---#
        final_training = np.concatenate((training_data,training_dataf))
        temp3,temp4 = final_training.shape
        
        #----------creating labels for final_training------------#
        final_labels= np.concatenate((label_true,label_false))
        
        #print 'shape of ifnal ddata',final_labels.shape,final_training.shape
        #--generating testing and training data randomly--#
        #X_train, X_test, y_train, y_test = train_test_split(final_training, final_labels, train_size=0.80, random_state=42)        
        #X_train,y_train,f2 = split_new(final_training,final_labels,0.8) #split the training and testing data
        #--creating instance of random forest---#
        #print 'final training'
        X_train = final_training
        y_train = final_labels
        temp1,temp2 = X_train.shape
        #print 'teri makk',temp1, temp2
        est = RandomForestClassifier(n_estimators =10,max_features='auto',max_depth=None,min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0,max_leaf_nodes=None,n_jobs=1)
        #--fitting data and labels--#
        est.fit(X_train,y_train)      #make trees from trainning data and labels
        #x_train is the training data, and y_train are there labels.
        #print 'score',est.score(X_test,y_test)
        Location = classi+f_type
        #print 'Location',Location
        try :
            os.stat(Location)
        except :
            os.mkdir(Location)
                  
                   
        
        
        save_location = Location+'/'+directory+'_'+str(0)+'.pkl'
        #print 'shape',test_data.shape
        joblib.dump(est, save_location,compress=9)#only save the classifier not the data..
        #0 is sent to check the recusrion depth
        ret = re_train_prefilt(save_location,directory,X_train,y_train,shape_y,0,f_type)
        #print 'return value',ret

    except Exception as e:
        print('Error on line {}',str(sys.exc_info()[-1].tb_lineno), type(e), e)

def randomForest(data_correct,data_false,name,shape_y,f_type):
    
    row_correct,column_correct = data_correct.shape
    
    row_false,column_false = data_false.shape
    training_data = data_correct

    classifier(training_data,data_false,name,0.5,shape_y,f_type)




def load_new(name,f_type):
    '''extract the correct and false feature of the image'''
    feature_corr = pd.read_csv(feat_true+f_type+'/'+name+'.csv',sep=',',header=None)
    feature_corr = np.asarray(feature_corr)

    shape_x,shape_y = feature_corr.shape
    feature_false = pd.read_csv(feat_false+f_type+'/'+name+'.csv',sep=',',header=None)
    feature_false = np.asarray(feature_false)

    data_correct = feature_corr[:,:shape_y-1]
    data_false = feature_false[:,:shape_y-1]
    #print type(data_correct),type(data_false),type(name),type(shape_y-1),type(f_type)
    randomForest(data_correct,data_false,name,shape_y-1,f_type)
#start()
def start_1(name): #start from here
    load_new(name,'Gist')
    load_new(name,'Hog')
#load_new(name,f_type) #load the feature...

