from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

#batch_size, num_classes and epochs
batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

#construct a convolutional neuron network
#if you change the architecture, you should change the following code
#####################################################################

model = Sequential()
model.add(Conv2D(10,(3,3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

####################################################################


#model compile:loss function, optimiazer and metrics
model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
              
#visualize the architecture of CNN
model.summary()          

#load the well-trained parameters, including weights and bias
model.load_weights('mnist_demo1.h5')

#obtain the model prediction in testing set
model_prediction = model.predict(x_test)

#try to find the differencees between  model_prediction and y_test to find the miss judgement
###################################################################



	
#print(model_prediction[0,:])
#print(len(model_prediction))
#print(np.argmax(y_test[0,:]))
d = {}

for i in range(len(model_prediction)):
	a = model_prediction[i,:]
	b = np.argmax(a)
	c = np.argmax(y_test[i,:])
	if (b!=c):
		d[c] = x_test[i]

e = [2,0,1,7,2,1,4,3,0,0,6,4]
for j in range(12):
	ax = plt.subplot(1,12,j+1)
	plt.imshow(d[e[j]].reshape(28,28))
	plt.gray
	

plt.show()







#################################################################
				
				
				
				

