from flask import Flask,jsonify,Response
import dummy5
import json
import afinr_crop_custom1
import negative
import train
from celery import Celery
from celery.utils.log import get_task_logger
from werkzeug.contrib.fixers import ProxyFix

'''using celery tasks queue to queue the tsks and images'''




#logger = get_task_logger('E:\\Test1.log')#get the log file instance
app = Flask(__name__)


'''configuring celery and redis redis is our broker. '''
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0' #url where broker service is running
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, backend=app.config['CELERY_RESULT_BACKEND'],broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)




#this the background task
@celery.task
def my_background_task(path1, name): #this is the celery task this will execute the tasks in the background
	
	try:
		logger.info('starting function %s'%(name))
		v = afinr_crop_custom1.single_affine(path1,name)
		negative.start(name,v+1)
		train.start_1(name)
		logger.info("completed %s"%(name))
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
@app.route("/vaibhav/<path:post_id>") #this will check if the image is present or not
def hello(post_id):
	try:
		v = dummy5.image_calc(post_id)
		
		list1 = v.values()
		if list1[0] == []:
			return 'EmptyList'
		else:	
			return jsonify(v)
		
	except:
		print 'Err'
#add the image to thedata base
@app.route("/add_data/<path:path1>/<string:name>") #address at which to send if image is not present in data baase
def add(path1,name):#add the image to our classifier
	
		
	path1,name = split(path1,name)
	#print 'name is ',name
	#print 'path is',path1
	my_background_task.delay(path1,name)
		
	    
	
	
	return 'image is added'
	

app.wsgi_app = ProxyFix(app.wsgi_app)

if __name__ == "__main__":
    app.run(debug=True)#execution starts here