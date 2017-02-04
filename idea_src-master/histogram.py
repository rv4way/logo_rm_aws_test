import numpy as np
import cPickle
import os
import cv2
import pandas as pd

def split_img(img_arr):

	x,y,z = img_arr.shape
	x1 = int(x/2)
	y1 = int(y/2)

	front_img = img_arr[:,:y1]
	back_img = img_arr[:,y1:]

	top_img = img_arr[:x1, :]
	bottom_img = img_arr[x1:, :]

	return front_img, back_img, top_img, bottom_img


def calc_hist(img_arr):

	rows, cols = img_arr.shape[:2]
	basehsv = cv2.cvtColor(img_arr,cv2.COLOR_BGR2HSV)

	hbins = 180
	sbins = 255
	hrange = [0,180]
	srange = [0,256]
	ranges = hrange+srange

	img_hist = cv2.calcHist(basehsv,[0,1],None,[180,256],ranges)
	cv2.normalize(img_hist,img_hist,0,255,cv2.NORM_MINMAX)

	return img_hist

def add_hist(img_arr, profile_id):

	def dump_pkl(arr, file_path):
		f = open(file_path, 'wb')
		cPickle.dump(arr, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()

	histogram_path = '../DataBase/histogram'
	profile_path = os.path.join(histogram_path, str(profile_id))
	if not os.path.exists(profile_path):
		os.mkdir(profile_path)

	front_img, back_img, top_img, bottom_img = split_img(img_arr)

	file_name = str(profile_id) + '_' + str('front_hist') + '.pkl'	
	front_path = os.path.join(profile_path, file_name)
	front_hist = calc_hist(front_img)
	dump_pkl(front_hist, front_path)


	file_name = str(profile_id) + '_' + str('back_hist') + '.pkl'
	back_path = os.path.join(profile_path, file_name)
	back_hist = calc_hist(back_img)
	dump_pkl(back_hist, back_path)


	file_name = str(profile_id) + '_' + str('top_hist') + '.pkl'
	top_path = os.path.join(profile_path, file_name)
	top_hist = calc_hist(top_img)
	dump_pkl(top_hist, top_path)


	file_name = str(profile_id) + '_' + str('bottom_hist') + '.pkl'
	bottom_path = os.path.join(profile_path, file_name)
	bottom_hist = calc_hist(bottom_img)
	dump_pkl(bottom_hist, bottom_path)


	file_name = str(profile_id) + '_' + str('img_hist') + '.pkl'
	img_path = os.path.join(profile_path, file_name)
	img_hist = calc_hist(img_arr)
	dump_pkl(img_hist, img_path)


def search_hist(profile_id, img_arr):
	hist_path = '../DataBase/histogram/'

	response = []
	for x in profile_id:
		file_path = os.path.join(hist_path, x)
		score = compare_hist(file_path, img_arr)
		if score == 1:
			response.append(x)
	print response
	return response


def compare_hist(file_path, img_arr):
	def read_pkl(file_path):
		file_data = open(file_path, 'rb')
		hist = cPickle.load(file_data)
		return hist

	#front_img, back_img, top_img, bottom_img = split_img(img_arr)
	result = []

	hist_list = os.listdir(file_path)
	for x in hist_list:
		if str('front') in x:
			front = os.path.join(file_path, x)

		if str('back') in x:
			back = os.path.join(file_path, x)

		if str('top') in x:
			top = os.path.join(file_path, x)

		if str('bottom') in x:
			bottom = os.path.join(file_path, x)

		if str('img') in x:
			img_hist = os.path.join(file_path, x)

	front_hist = read_pkl(front)
	back_hist = read_pkl(back)
	top_hist = read_pkl(top)
	bottom_hist = read_pkl(bottom)
	img_hist = read_pkl(img_hist)

	#front_hist_comp = calc_hist(front_img)
	#back_hist_comp = calc_hist(back_img)
	#top_hist_comp = calc_hist(top_img)
	#bottom_hist_comp = calc_hist(bottom_img)
	img_hist_comp = calc_hist(img_arr)

	distance = cv2.compareHist(img_hist, img_hist_comp,0)
	result.append(distance)

	distance = cv2.compareHist(front_hist, img_hist_comp,0)
	result.append(distance)

	distance = cv2.compareHist(back_hist, img_hist_comp,0)
	result.append(distance)

	distance = cv2.compareHist(top_hist, img_hist_comp,0)
	result.append(distance)

	distance = cv2.compareHist(bottom_hist, img_hist_comp,0)
	result.append(distance)

	count = 0
	for x in result:
		if x > float(0) and x <= float(1):
			count += 1

	if count == 5:
		return int(1)
	else:
		return int(0)

'''
if __name__ == '__main__':
	path = '/home/rahul/Dropbox/Processed Logo Images/Logos Phase 1/3M/Correct/3M_Logo_K_13mm_jpg_logo_300_570r_1.png'
	img_arr = cv2.imread(path)
	search_hist(['sjjhjshd'], img_arr)
	#split_img(img_arr)
	#add_hist(img_arr, 'sjjhjshd')
'''