#Python Functions File for Functions for Data Augmentation

from scipy.misc import imresize as imrs
from scipy.misc import imrotate as imro
import numpy as np 
import cv2


#Function to scale an image 
def Scale(Input,Target):

	shape = Input.shape
	N = shape[3]

	result_RGB = []
	result_Depth = []

	for i in range(N):
		img = Input[:,:,:,i]
		target = Target[:,:,i]
		factor = np.random.uniform(1,1.5)
		
		res_rgb = imrs(img,factor)
		res_depth = imrs(target, factor)

		res_depth = res_depth/factor 

		res_rgb = res_rgb[0:shape[0],0:shape[1]]
		res_depth = res_depth[0:shape[0],0:shape[1]]
		result_RGB.append(res_rgb)
		result_Depth.append(res_depth)

	Output = [ np.stack(result_RGB,axis =3) , np.stack(result_Depth,axis=2) ]

	return Output


#Function to Rotate an image

def Rotate(Input,Target):
	
	shape = Input.shape
	N = shape[3]

	result_RGB = []
	result_Depth = []

	for i in range(N):
		img = Input[:,:,:,i]
		target = Target[:,:,i]
		
		angle = np.random.uniform(-5,+5)	

		res_rgb = imro(img, angle)
		res_depth = imro(target, angle) 

		result_RGB.append(res_rgb)
		result_Depth.append(res_depth)
	

	Output = [ np.stack(result_RGB,axis =3) , np.stack(result_Depth,axis=2) ]

	return Output



#Function to Translate an image

def Crop(Input,Target):	
	rgb_size = [304,228]
	depth_size = [74,55]

	shape = Input.shape
	N = shape[3]

	result_RGB = []
	result_Depth = []

	for i in range(N):
		img = Input[:,:,:,i]
		target = Target[:,:,i]
		
		for x in range(5):	
			row_rand = np.random.randint(0,shape[0]-rgb_size[0])
			col_rand = np.random.randint(0,shape[1] - rgb_size[1])

			res_rgb = img[row_rand:row_rand+304 , col_rand:col_rand+228,:]
			
			res_depth = target[row_rand:row_rand+304 , col_rand:col_rand+228]
			res_depth = imrs(res_depth , [depth_size[0] , depth_size[1]])

			
			
			result_RGB.append(res_rgb)
			result_Depth.append(res_depth)
		
	Output = [ np.stack(result_RGB,axis =3) , np.stack(result_Depth,axis=2) ]

	return Output


#Function to randomly color an image
def Color(Input,Target):
	
	shape = Input.shape
	N = shape[3]

	result_RGB = []

	for i in range(N):
		img = Input[:,:,:,i]
		target = Target[:,:,i]
		
		vec = np.random.uniform(.8,1.2,3)	

		res_rgb = img * vec
		
		result_RGB.append(res_rgb)
	

	Output = [ np.stack(result_RGB,axis = 3) , Target ]

	return Output


#Function to flip an image

def Flip(Input,Target):
	
	shape = Input.shape
	N = shape[3]

	result_RGB = []
	result_Depth = []

	for i in range(N):
		img = Input[:,:,:,i]
		target = Target[:,:,i]

		res_rgb = cv2.flip(img,1)
		res_depth = cv2.flip(target,1) 

		result_RGB.append(res_rgb)
		result_Depth.append(res_depth)
	

	Output = [ np.stack(result_RGB,axis =3) , np.stack(result_Depth,axis=2) ]

	return Output






















