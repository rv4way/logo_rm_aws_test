from flask import Flask,jsonify,Response,request, redirect, url_for
import dummy5
import json
import afinr_crop_custom1
import negative
import train
from celery import Celery
from celery.utils.log import get_task_logger
from werkzeug.contrib.fixers import ProxyFix
import os
from werkzeug.utils import secure_filename
import cv2
import io
import numpy as np
import pandas as pd

import face_src
from operator import is_not
from functools import partial
import sys
from padding_remove import remove_padding
import re_train
'''using celery tasks queue to queue the tsks and images'''



#logger = get_task_logger('/media/rahul/42d36b39-1ad7-45d4-86bb-bf4e0a66a97f/logo aws/DataBase/test.log')#get the log file instance
logger = get_task_logger(__name__)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['mp4', '3gp', 'avi'])


'''configuring celery and redis redis is our broker. '''
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0' #url where broker service is running
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
celery = Celery(app.name, backend=app.config['CELERY_RESULT_BACKEND'],broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)
with open('Properties.json', 'r') as fp:
    data = json.load(fp)

search_in = pd.read_csv(data["ImageCsv"],sep=',',header=None)
search_in = np.asarray(search_in)

# m_a = data["MongoUrl"]

# c = MongoClient(m_a)   #taking instance of mongo client
# #mer4 = data["ImageDatabase"]
# db = c["ImageDatabase"]
# db_count = db.Image_count


#this the background task
@celery.task
def my_background_task(path1, name): #this is the celery task this will execute the tasks in the background
	
	try:
		logger.info('starting function %s'%(name))
		#print 'starting with',name
		
		img = path1
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		top, bottom = remove_padding(gray)
		left, right = remove_padding(gray.T)
		img = img[top:bottom, left:right]
		path1 = img 


		v = afinr_crop_custom1.single_affine(path1,name)

		print "Done : affine"

		logger.info('v %s'%(name))
		
		negative.start(name,v+1)
		print "Done : negative"

		logger.info('23v %s'%(name))
		
		train.start_1(name)
		print "Done : training"

		update_idcsv(name)
		print "Done :csv"

		#print 'completed with',name
		logger.info("completed %s"%(name))

		re_train.start()
		print "Done :re_train"



	except Exception,e:
		print 'exceptiom',str(e)
	return "done"


def split(path1,name):
	s1 = path1
	s4 = s1.split('path')[1]
	s4 = s4[1:]
	s4 = s4.split('>')[0]
	path1 = s4.replace('//', '\\')
	#print 'image s',str2
	s1 = name
	s4 = s1.split(':')[1].split('>')[0]
	return path1,s4

#return the result
@app.route("/image-processing/search",methods=['GET', 'POST']) #this will check if the image is present or not
def hello():
	try:
		if request.method == 'POST':	
		# check if the post request has the file part
			#header_req = request.headers.get('x-image-profile-id')

			photo = request.files['photo'] #if photo is present in the request object

			in_memory_file = io.BytesIO()
			photo.save(in_memory_file)
			data1 = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
			color_image_flag = 1

			img = cv2.imdecode(data1, color_image_flag) #img is the image

			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			top, bottom = remove_padding(gray)
			left, right = remove_padding(gray.T)
			img = img[top:bottom, left:right]

			
			v = dummy5.image_calc(img)
			v = list(set(v))
			print v
			print len(v)
			if len(v) == 0:
				v = json.dumps(v)
				ret_val={'message':'images not found','status':0,'data':v}
				return 	jsonify(**ret_val)
			else:	
				v = json.dumps(v)
				ret_val={'message':'images found','status':1,'data':v}
				return 	jsonify(**ret_val)			
		
	except Exception as e:
		print 'exception',str(e)

		ret_val={'message':'server error. database empty','status':0,'data':'No Data' }
		return 	jsonify(**ret_val)

#get status of added image
@app.route("/image-processing/logo/status",methods=['GET', 'POST'])
def get_status():
	try:
		if request.method == 'POST':
			
			header_req = request.headers.get('x-image-profile-id')
			c = check_status(str(header_req))
			#c = db_count.find_one({"name":header_req})
			if c == True:
				ret_val={'message':'image is added.','status':1,'data':header_req }
				return 	jsonify(**ret_val)
			else:
				ret_val={'message':'image not added.','status':0,'data':header_req }
				return 	jsonify(**ret_val)		
	except:
		ret_val={'message':'can not be added','status':0,'data':header_req }
		return 	jsonify(**ret_val)
#status of status face
@app.route("/image-processing/face/status",methods=['GET', 'POST'])
def get_status_face():
	try:
		if request.method == 'POST':
			
			header_req = request.headers.get('x-image-profile-id')
			#c = db_count.find_one({"name":header_req}) Replace with face data base
			c = check_face_status(header_req)
			if c is not True:
				ret_val={'message':'image is added.','status':1,'data':header_req }
				return 	jsonify(**ret_val)
			else:
				ret_val={'message':'image not added.','status':0,'data':header_req }
				return 	jsonify(**ret_val)		
	except:
		ret_val={'message':'can not be added','status':0,'data':header_req }
		return 	jsonify(**ret_val)


#add the image to thedata base
@app.route("/image-processing/logo/add",methods=['GET','POST']) #address at which to send if image is not present in data baase
def add():#add the image to our classifier
	try:
		if request.method == 'POST':
			
			#header_req = 'test637236827test'
			header_req = request.headers.get('x-image-profile-id')
			photo = request.files['photo'] #if photo is present in the request object
			in_memory_file = io.BytesIO()
			photo.save(in_memory_file)
			data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
			color_image_flag = 1
			img = cv2.imdecode(data, color_image_flag)
			#path1,name = split(file,name)
			s4 = str(header_req)
			'''s1 = name
			s4 = s1.split(':')[1].split('>')[0]'''
			#print 'name is ',name
			
			my_background_task.delay(img,s4)

			
			
	    	ret_val={'message':'image added.','status':2,'data':header_req }

	    	print str(ret_val)
	    	return 	jsonify(**ret_val)
	except Exception as e :
		print('Error on line {}',str(sys.exc_info()[-1].tb_lineno), type(e), e)
		ret_val={'message':'request cannot be processed','status':0,'data':header_req }
		return 	jsonify(**ret_val)

@app.route("/image-processing/face/add",methods=['GET','POST'])
def add_vid():
	try:
		if request.method == 'POST':
			
			header_req = request.headers.get('x-image-profile-id')
			photo = request.files['Video'] #if photo is present in the request object
			'''in_memory_file = io.BytesIO()
			photo.save(in_memory_file)
			data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)'''
			filename = secure_filename(file.filename)#filename has video
			face_src.outsource(video_path, header_req)
			ret_val={'message':'image queued for adding.','status':2,'data':header_req }
			return 	jsonify(**ret_val)
	except:
		ret_val={'message':'request cannot be processed','status':0,'data':header_req }
		return 	jsonify(**ret_val)	


def update_idcsv(name):
	search_in = pd.read_csv(data["ImageCsv"], sep = ',', header = None)
	search_in = np.asarray(search_in)
	if name in search_in[0]:
		return
		#update
	else:
		'''
		search_in = np.concatenate((search_in,name))
		with open(fdata["ImageCsv"], 'a') as myfile:
			wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
			wr.writerow(name)
		'''
		temp_csv_path = data["ImageCsv"]
		f = open(temp_csv_path, 'a')
		f.write(name)
		f.write('\n')
		f.close()

def check_status(name):
	search_in = pd.read_csv(data["ImageCsv"],sep=',',header=None)
	search_in2 = np.asarray(search_in)
	if name in search_in2:
		return True
	else:
		return False

def check_face_status(header_req):
	csv_path = '/root/ideaswire/imageprocessing/face_database/csv/id.csv'
	data = open(csv_path, 'rb')
	data = np.asarray(data)
	for x in range(len(data)):
		csv_list = data[x]
		label = csv_list[0]
		no_image = csv_list[1]
		face_header = csv_list[2]
		if face_header == header_req:
			return True
		else:
			return False

app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == "__main__":
	app.run(debug = True, host = '0.0.0.0',port = 5000)
