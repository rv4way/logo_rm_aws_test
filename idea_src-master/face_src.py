import os
import video_to_face
from PIL import Image
import numpy as np
import cPickle
import video_to_face
#import deepid_generate
#from annoy import AnnoyIndex
import pandas as pd
import time

#video_path = '/vai_test.mp4'
img_path = '/root/ideaswire/imageprocessing/face_database/video_image'
vector_path = '/root/ideaswire/imageprocessing/face_database/vector'
csv_path = '/root/ideaswire/imageprocessing/face_database/csv/id.csv'
param_path = '/root/ideaswire/imageprocessing/face_database/param/1-1400.txt'
deepid_path = '/root/ideaswire/imageprocessing/face_database/deepid'
annoy_path = '/root/ideaswire/imageprocessing/face_database/annoy/test.ann'


def create_vector_array(img_path, label, vector_path):
	img_size = (3,55,47)
	image_vector_len = np.prod(img_size)
	image_array = []
	label_array = []

	img_list = os.listdir(img_path)
	image_no = len(img_list)
	for x in img_list:
		x_path = os.path.join(img_path, x)
		image = Image.open(x_path)
		arr_img = np.asarray(image, dtype='float64')			
		arr_img = arr_img.transpose(2,0,1).reshape((image_vector_len, ))
		image_array.append(arr_img)
		label_array.append(label)
	image_array = np.asarray(image_array, dtype='float64')
	label_array = np.asarray(label_array, dtype='int32')
	vector = image_array, label_array

	file_name = vector_path + '/' + str(label) + '.pkl'
	cPickle_output(vector, file_name)
	return file_name, image_no

def cPickle_output(vector, file_name):

	f = open(file_name, 'wb')								
	cPickle.dump(vector, f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()

def update_csv(csv_path, label, image_no, x_profile_id):
	f = open(csv_path, 'a')
	line = str(label) + ',' + str(image_no) + ',' + str(x_profile_id)
	f.write(line)
	f.write('\n')
	f.close

def fetch_label(csv_path):
	if not os.path.exists(csv_path):
		print 'DATABASE EMPTY'
		label = 0
		return label
	else:
		csv_data = pd.read_csv(csv_path, sep = ',', header = None)
		csv_data = np.asarray(csv_data)
		last = len(csv_data)
		x = csv_data[last-1]
		last_label = x[0]
		return last_label+1

def update_annoy(deepid_path, annoy_path, csv_path):
	f = 160
	t = AnnoyIndex(f, metric = 'angular')

	csv_data = pd.read_csv(csv_path, sep = ',', header = None)
	csv_data = np.asarray(csv_data)
	update_list = []
	for x in range(len(csv_data)):
		csv_list = csv_data[x]
		label = csv_list[0]
		no_image = csv_list[1]
		if no_image >= 30:
			update_list.append(label)

	for x in range(len(update_list)):
		update_deepid = deepid_path + '/' + str(update_list[x]) + '.pkl'
		deepid_data = cPickle.load(open(update_deepid,"rb"))
		deepid_t = deepid_data[0]
		deepid_t = deepid_t[0]
		label_t = deepid_data[1]
		x_new, y_new = deepid_t.shape
		for y in range(x_new):
			deepid = deepid_t[y]
			deepid = np.array(deepid)
			label = label_t[y]
			t.add_item(label, deepid.astype(np.int32))

	t.build(1000)
	t.save(annoy_path)

def generate_deepid(vector_file, deepid_path):
	deepid, y = deepid_generate.outsource(vector_file, param_path)
	deepid_file = deepid_path + '/' + str(y[0]) + '.pkl'
	cPickle_output((deepid,y), deepid_file)

def outsource(video_path, x_profile_id):
	label = fetch_label(csv_path)
	image_path = video_to_face.video_to_image(video_path, img_path, label)
	vector_file, image_no = create_vector_array(image_path, label, vector_path)
	#generate_deepid(vector_file, deepid_path)
	update_csv(csv_path, label, image_no, x_profile_id)
	#update_annoy(deepid_path, annoy_path, csv_path)
	print 'DUE TO NO GPU IMAGE ONLY IMAGE VECTOR IS ADDED '

'''
if __name__ == '__main__':
	outsource(video_path, 'testte')
	'''