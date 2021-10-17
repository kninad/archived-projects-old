#Preproceesing for images

import numpy as np 
from scipy.misc import imresize 

def IMGResize(filename="images.npy",height = 304, width = 228):

	temp = np.load(filename)
	shape = np.shape(temp)

	result = []
	count = 0

	for i in range(shape[3]):
		
		if(count%10 == 0):
			print str(count) + "images done"

		img = temp[:,:,:,i]
		
		img_res = imresize(img,[height,width])
		
		result.append(img_res)
		#result = np.concatenate((result,img_corr),axis=3)
		count+=1

	return np.stack(result,axis=3)



def DepthResize(filename="depths.npy",height = 74, width = 55):

	temp = np.load(filename)
	shape = np.shape(temp)

	result = []
	count = 0

	for i in range(shape[2]):

		if(count%10 == 0):
			print str(count)+ "depths resized"

		img = temp[:,:,i]
		
		img_res = imresize(img,[height,width])

		result.append(img_res)
		
		count+=1

	return np.stack(result,axis=2)


def PreprocessRGB(filename="ResizedImages.npy"):	
	temp = np.load(filename)
	shape = np.shape(temp)
	
	#result = np.zeros([ shape[0],shape[1],shape[2],1])
	result = []
	count = 0

	for i in range(shape[3]):
		if(count%10 == 0):
			print str(count)+ "images done"
		img = temp[:,:,:,i]
		
		mean0 = np.mean(img[:,:,0])
		mean1 = np.mean(img[:,:,1])
		mean2 = np.mean(img[:,:,2])

		std0 = np.std(img[:,:,0])
		std1 = np.std(img[:,:,1])
		std2 = np.std(img[:,:,2])

		img_corr = (img - np.asarray([mean0,mean1,mean2])) / [std0,std1,std2]
		result.append(img_corr)

		#img_corr = img_corr.reshape((shape[0],shape[1],shape[2],1))

		#result = np.concatenate((result,img_corr),axis=3)
		count+=1
	return np.stack(result,axis=3)
	#return result[:,:,:,1:]

RGB_corr = PreprocessRGB()
np.save('PreProcessedImages.npy',RGB_corr)