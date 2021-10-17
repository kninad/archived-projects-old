import numpy as np
import keras
from keras.models import model_from_json
from keras.models import Sequential , Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers.convolutional import ZeroPadding2D
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

def define_cost(self, pred, y0, m0):
        bsize = self.bsize
        npix = int(np.prod(test_shape(y0)[1:]))
        y0_target = y0.reshape((self.bsize, npix))
        y0_mask = m0.reshape((self.bsize, npix))
        pred = pred.reshape((self.bsize, npix))

        p = pred * y0_mask
        t = y0_target * y0_mask

        d = (p - t)

        nvalid_pix = T.sum(y0_mask, axis=1)
        depth_cost = (T.sum(nvalid_pix * T.sum(d**2, axis=1))
                         - 0.5*T.sum(T.sum(d, axis=1)**2)) \
                     / T.maximum(T.sum(nvalid_pix**2), 1)

        return depth_cost

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
	#f_sum = np.sum(first,out=None)
	#s_sum = np.sum(second,out=None)
	s_sum = s_sum * s_sum;
	f_sum = f_sum/n
	s_sum = lamda * (s_sum/(n * n))

	return (f_sum - s_sum)


def Construct_Fine_Model(input_rows=304,input_cols=228,input_depth=3):
	#Fine Scale Model
	print("Building Fine Scale Model")
	#input_image_fine = Input(shape=(input_rows,input_cols,input_depth))
	input_image_fine = Input(shape=(304,228,3))
	feature_map_coarse = Input(shape=(74,55,1))

	#ZeroPad = ZeroPadding2D(padding=(1,1))(input_image_fine)

	Conv1_fine = Conv2D(63,(9,9),strides=2)(input_image_fine)
	#Conv1_fine = Conv2D(63,(9,9),strides=2)(ZeroPad)
	MaxPool1_fine = MaxPooling2D((2,2))(Conv1_fine)

	Concatenate_1 = concatenate([MaxPool1_fine,feature_map_coarse])

	#We have to introduce zero padding layers here to maintain shape
	ZeroPad1 = ZeroPadding2D(padding=(2,2))(Concatenate_1)	
	#Conv2_fine = Conv2D(64,(5,5))(Concatenate_1)
	Conv2_fine = Conv2D(64,(5,5))(ZeroPad1)

	ZeroPad2 = ZeroPadding2D(padding=(2,2))(Conv2_fine)	

	Conv3_fine = Conv2D(1,(5,5))(ZeroPad2)

	Fine_Model = Model([input_image_fine,feature_map_coarse],Conv3_fine)
	print("Built Fine Scale Model")
	return Fine_Model

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
Y_train = Y_train.reshape([1449,74,55,1])

Coarse_Output = np.random.rand(1449,74,55,1)

Fine_Scale_Model = Construct_Fine_Model()

print("Compiling Model")
Fine_Scale_Model.compile(loss=loss_function, optimizer="SGD")

#Coarse_Scale_Model.compile(loss='mean_squared_error', optimizer="SGD")
print("Model Compilied")

train(Fine_Scale_Model,[X_train,Coarse_Output],Y_train)