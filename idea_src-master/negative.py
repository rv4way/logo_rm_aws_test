from __future__ import division
import os,csv,cv2,json,sys
import numpy as np
import Gist_feat_last
import HOG_feat2
import pandas as pd
import random

'''this file will generate the negative feature for the new images which are comming in from the server'''




with open('Properties.json', 'r') as fp:
    data = json.load(fp)

feat_gist = data["PositiveGist"]#location f gist and hog feature
feat_hog =  data["PositiveHog"]#take from database
saven_gist = data["NegativeGist"]
saven_hog = data["NegativeHog"]
# m_a = data["MongoUrl"]

# c = MongoClient(m_a)   #taking instance of mongo client
# mer4 = data["ImageDatabase"]
# db = c[mer4]  #making new collection
# db2_g = db.Gabor
# db2_h = db.Hog
# db2_gn = db.GaborNeg
# db2_hn = db.HogNeg
# db_count = db.Image_count

false_zero = np.asarray([0])

def add_feature_to_database(name,feature,type_feat):
    if feature == "Gabor":

    	temp = db2_gn.find_one({"_id":name})
    	if temp is not None:
        	ms = temp["feature"]
        	ms_new = np.concatenate(ms,feature)
        	db2_gn.update_one({"_id":name},{"$set":{"feature":ms_new}})
    	else:
        	d_temp = {"_id":name,"feature":feature}
        	res = db2_gn[type_feat].insert_one(d)
    if feature == "Hog":
    	temp = db2_hn.find_one({"_id":name})
    	if temp is not None:
        	ms = temp["feature"]
        	ms_new = np.concatenate(ms,feature)
        	db2_hn.update_one({"_id":name},{"$set":{"feature":ms_new}})
    	else:
        	d_temp = {"_id":name,"feature":feature}
        	res = db2_hn[type_feat].insert_one(d)



def get_neg_names_db(name):
	neg_names = []
	results = db_count.find({"name": {"$ne": name}})
	for res in results:
		temp = res["name"]
		neg_names.append(temp)
	
	return neg_names

def SubDirPath (d):
    return filter(os.path.isdir, [os.path.join(d,f) for f in os.listdir(d)])

def file_find(rootdir):
	new_dir = rootdir+'/'+'Correct'
	files_name = os.listdir(new_dir)
	num32 = len(files_name)
	num_list = range(num32)
	index = random.choice(num_list)
	dir213 = files_name[index]
	dir113 = new_dir+'/'+dir213
	#print dir1
	return dir113
'''return the random file from the already present feature files'''
def random_dir(dir_false,name): #return the random file
	try:
		rem = name+'.csv'

		dir12 = dir_false
		list1 = os.listdir(dir12)
		
		l = list1.remove(rem)
		#print 'rem',list1
		length1 = len(list1)
		# print "dir: ", type(dir12)
		r = random.randint(0,length1-1)
		#print 'r is',r
		temp1 = list1[r]
		temp = dir12+'/'+temp1
		#print 'chosen file is',temp
		return temp


	except Exception as e:
		print('Error on line ##{}' ,str(sys.exc_info()[-1].tb_lineno), type(e), e)
'''create the -ve HOG and Gist feature for training the 1st classifier'''
def create_csv(name,number):
	try:
		count = 0
		#print name, number
		for i in range(number):
			falseImage = random_dir(feat_gist,name)
	        
	        
			feat_neg = pd.read_csv(falseImage,sep=',',header=None)
			#print 'hi22'
			feat_neg = np.asarray(feat_neg)
			shape_x,shape_y = feat_neg.shape
			r324 = random.randint(0,shape_x-1)
			feature = feat_neg[r324,:shape_y-1]
			feature = np.concatenate((feature,false_zero))#-ve feature
			#print 'feature is ',feature
			with open(saven_gist+name+'.csv', 'a') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
				wr.writerow(feature)
			count=count+1
		for i in range(number):
			falseImage = random_dir(feat_hog,name)
	        
			feat_neg = pd.read_csv(falseImage,sep=',',header=None)

			feat_neg = np.asarray(feat_neg)
			shape_x,shape_y = feat_neg.shape
			r = random.randint(0,shape_x-1)
			#print 'rowis',r
			feature = feat_neg[r,:shape_y-1]
			feature = np.concatenate((feature,false_zero))
			with open(saven_hog+name+'.csv', 'a') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
				wr.writerow(feature)
			count=count+1
			#print 'times',count
	except Exception as e:
		#print('Error on line {}' ,str(sys.exc_info()[-1].tb_lineno), type(e), e)
		pass
def create_neg_database(name,number):
	names_neg = get_neg_names_db(name)
	neg_d = random.sample(names_neg, number)

	for names in neg_d:
		c2 = db2_g.find_one({"_id":name})
		feature_arr = c2["feature"]
		feat_temp = np.asarray(feature_arr)
		feat_x,feat_y = feat_temp.shape
		neg_feature = random.sample(feat_temp, 1)
		neg_feature = neg_feature[:,:feat_y-1]
		neg_feature = np.concatenate((neg_feature,[0]))
		add_feature_to_database(name,neg_feature,"Gabor")
	for names in neg_d:
		c2 = db2_h.find_one({"_id":name})
		feature_arr = c2["feature"]
		feat_temp = np.asarray(feature_arr)
		feat_x,feat_y = feat_temp.shape
		neg_feature = random.sample(feat_temp, 1)
		neg_feature = neg_feature[:,:feat_y-1]
		neg_feature = np.concatenate((neg_feature,[0]))
		add_feature_to_database(name,neg_feature,"HOG")
	


def start(name,number):
	#print 'number is ',number
	create_csv(name,number)

#start('something4',10)
