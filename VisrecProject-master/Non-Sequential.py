import numpy as np
import keras
from keras.models import model_from_json
from keras.models import Sequential , Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.optimizers import SGD, Adam
from keras.layers import concatenate #for concatenating tensors
import json
from PIL import Image
import tensorflow as tf

input_rows = 304
input_cols = 228
input_depth = 1
lamda = 0.5;
n = 100;

def loss_function(_true, _pred):

	print("Inside Loss function ")
	print _true.shape
	print _pred.shape

	n = np.size(_true)

	#Ensure only positive values are in predicted
	_pred = _pred.clip(0.01,999999999)

	_true = _true.clip(0.01,999999999)


	_t = np.log(_true)
	_p = np.log(_pred)

	first = np.subtract(_t, _p)
	second = np.subtract(_p, _t)

	f_sq = np.square(first)

	f_sum = f_sq.sum()
	s_sum = second.sum()
	s_sum = s_sum * s_sum;
	f_sum = f_sum/n
	s_sum = lamda * (s_sum/(n * n))

	return (f_sum - s_sum)

def Construct_Coarse_Model(input_rows=304, input_cols=228, input_depth=3):  #Functional API Construction
	
	print("Building Coarse Scale Model")
	input_image_coarse = Input(shape=(input_rows,input_cols,input_depth))
	
	Conv1_coarse = Conv2D(96,(11,11),strides=4)(input_image_coarse) #96 feature maps, 11*11 kernels, 4 stride
	MaxPool1_coarse = MaxPooling2D((2,2))(Conv1_coarse)

	Conv2_coarse = Conv2D(256,(5,5))(MaxPool1_coarse)
	MaxPool2_coarse = MaxPooling2D((2,2))(Conv2_coarse)

	Conv3_coarse = Conv2D(384,(3,3))(MaxPool2_coarse)
	
	Conv4_coarse = Conv2D(384,(3,3))(Conv3_coarse)

	Conv5_coarse = Conv2D(256,(3,3))(Conv4_coarse)

	Flat = Flatten()(Conv5_coarse)

	Dense1_coarse = Dense(4096)(Flat)

	Dropout1 = Dropout(0.5)(Dense1_coarse)

	Dense2_coarse = Dense(74*55)(Dropout1)
	Reshape_coarse = Reshape((74,55))(Dense2_coarse)

	Coarse_model = Model(input_image_coarse,Reshape_coarse)
	print("Built Coarse Scale Model")
	return Coarse_model


def train(model, X_train, Y_train):
	print("Starting Training")
	model.fit(X_train, Y_train, batch_size=32, epochs=20)
	print("Model trained")

	model_json = model.to_json()
	with open("model.json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("model.h5")

	#score = model.evaluate(X_train, Y_train)
	#print("Score is {}\n".format(score))


X_train = np.load("ResizedImages.npy")

#Need to Swap the axis here
X_train = np.swapaxes(X_train,0,3)
X_train = np.swapaxes(X_train,3,2)
X_train = np.swapaxes(X_train,2,1)

Y_train = np.load("ResizedDepths.npy")
Y_train = np.swapaxes(Y_train,0,2)
Y_train = np.swapaxes(Y_train,1,2)

Coarse_Scale_Model = Construct_Coarse_Model()

print("Compiling Model")
Coarse_Scale_Model.compile(loss=loss_function, optimizer="SGD")

#Coarse_Scale_Model.compile(loss='mean_squared_error', optimizer="SGD")
print("Model Compiled")

train(Coarse_Scale_Model,X_train,Y_train)