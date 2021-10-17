import numpy as np

import random

#Function to extract weak positions from RGB-E maps wrt to DepE(Depth Edge) maps, RGBE and DepE are both edge maps

WeakPositions(RGBE,DepE,N=20):
	#Dropping Histogram Idea as ranking is not relevant in detection of strong edges. Exact values may suffice. Also due to Noise 
	#limit calculations will be erroneous.
	#Calc Histograms
	#RGB_hist = cv2.calcHist([RGBE],[0],None,[256],[0,256])
	#Dep_hist = cv2.calcHist([DepE],[0],None,[256],[0,256])
	#We need to sample positions from RGB Image which are strong pixel values but simultaneously not strong values in DEP

	RGB_mask0 = RGBE>150  #Locations where there are strong edges of anytype
	Dep_mask0 = DepE<100  #Locations where there are no Depth Edges

	RGB_mask1 = RGBE<100  #Locations where there are no RGB edges

	Neg_mask_type0 = RGB_mask0 * Dep_mask0 #Positions where there are no depth edges but strong Edges of NonDepth Type
	Neg_mask_type1 = RGB_mask1 * Dep_mask0 #Position where there are no depth edges as well as no RGB edges

	list_type0 = Neg_mask_type0.nonzero()
	list_type1 = Neg_mask_type1.nonzero()

	len0 = len(list_type0[0])
	len1 = len(list_type1[0])

	type0 = []
	type1 = []
	sample0 = random.sample(range(0,len0+1),10)   #Generates random samples without replacement
	sample1 = random.sample(range(0,len0+1),10)   #Generates random samples without replacement
	
	for temp in range(10):
		i = sample0[temp]

		rowindex = list_type0[0][i]
		colindex = list_type0[1][i]
		val = DepE[rowindex,colindex]

		type0.append([rowindex,colindex,val])

		j = sample1[temp]

		rowindex = list_type1[0][j]
		colindex = list_type1[1][j]
		val = DepE[rowindex,colindex]

		type1.append([rowindex,colindex,val])

	ouptut = [type0,type1]
	return ouptut	



StrongPositions(RGBE,DepE,N=20):


	Dep_mask0 = DepE>180  #Locations where there are strong Depth Edges

	list_type0 = Dep_mask0.nonzero()

	len0 = len(list_type0[0])

	type0 = []
	sample0 = random.sample(range(0,len0+1),20)   #Generates random samples without replacement
	
	for temp in range(20):
		i = sample0[temp]

		rowindex = list_type0[0][i]
		colindex = list_type0[1][i]
		val = DepE[rowindex,colindex]

		type0.append([rowindex,colindex,val])

	return type0

