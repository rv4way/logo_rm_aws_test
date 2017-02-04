import cv2
from scipy.misc import imresize
import os
import random

def video_to_image(video_file, img_dir, label):
	count = 0
	img_arr = []
	
	cap = cv2.VideoCapture(video_file)
	success, image = cap.read()
	
	while success:
		success, image = cap.read()
		rows,cols, x = image.shape
		print '******************************',image.shape

		M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
		dst = cv2.warpAffine(image,M,(cols,rows))

		img_arr.append(dst)
		if len(img_arr) >=200:
			break

	img_path = img_dir + '/' + str(label)
	if not os.path.exists(img_path):
		os.mkdir(img_path)

	for x in range(30):
		temp_face_crop = random.choice(img_arr)
		croped_face = face_detection(temp_face_crop)
		cv2.imwrite(img_path + '/'  + str(label) + '_' + str(count) + '.jpg', croped_face)
		count = count + 1
	return img_path
		
def face_detection(image_array):

	face_cascade = cv2.CascadeClassifier("/root/ideaswire/imageprocessing/opencv/data/haarcascades/haarcascade_frontalface_default.xml")

	gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	count  = 0
	for (x,y,w,h) in faces:
		cv2.rectangle(image_array,(x,y),(x+w,y+h),(255,0,0),0)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = gray[y:y+h, x:x+w]
		temp = image_array[y:y+h,x:x+w]
		croped_face = imresize(temp, (47,55))
	return croped_face