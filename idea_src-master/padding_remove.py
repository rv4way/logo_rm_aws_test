import cv2
import numpy as np 


def remove_padding(gray):


	diffrences = []
	sum_diff = []

	for x in range(len(gray)-1):
		#print gray[x] - gray[x+1],x,x+1
		diffrences.append(gray[x] - gray[x+1])
	
	print len(diffrences)
	for x in diffrences:
		sum_diff.append(sum(x))

	# print sum_diff
	for ind,w in enumerate(sum_diff):
		if w!=0:
			top =ind
			break
	for ind,w in enumerate(sum_diff[::-1]):
		if w!=0:
			bot =ind
			break
	bot = len(gray) - bot
	
	return top, bot


	



if __name__ == '__main__':
	
	img_org = cv2.imread("/home/rahul/Desktop/tasks/padding/image/toi.png")
	gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

	#cal_padding(img_org)
	top, bottom = remove_padding(gray)
	left, right = remove_padding(gray.T)
	#image_roi(img_org)


	cv2.imshow('img', img_org[top:bottom, left:right ])
	cv2.imshow('img_org', img_org)

	if cv2.waitKey(0):
		cv2.destroyAllWindows()
	print "Done"
	cv2.imwrite("/home/rahul/Desktop/tasks/padding/image/new.png", img_org[top:bottom, left:right ])

